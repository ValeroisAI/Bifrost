"""
Çıkarım motoru: sürekli batch'leme (continuous batching).

Tek GPU iş parçacığı döngüsü:
  1. Kuyruktaki yeni istekleri kabul et: istemi parça parça prefill et (B=1), ilk tokeni örnekle,
     cache'ini çalışan batch'e ekle.
  2. Çalışan tüm diziler için tek bir decode adımı (B = aktif istek sayısı).
  3. Biten/iptal edilen dizileri batch'ten çıkar.
Farklı uzunluktaki diziler aynı adımda ilerler. DeltaNet durumu sabit boyutludur; dikkat katmanlarının
cache'i konum etiketli olduğundan hizalama gerekmez.
"""

import logging
import queue
import threading
import time
import uuid
from typing import Callable, List, Optional

import torch

from .model import HeimdallLM

log = logging.getLogger("heimdall.engine")


class Request:
    def __init__(self, ids: List[int], max_tokens: int = 256, temperature: float = 0.7, top_p: float = 1.0,
                 top_k: int = 0, stop_ids=(), callback: Optional[Callable] = None) -> None:
        self.id = uuid.uuid4().hex
        self.ids, self.max_tokens = list(ids), max_tokens
        self.temperature, self.top_p, self.top_k = temperature, top_p, top_k
        self.stop_ids = set(stop_ids)
        self.callback = callback or (lambda tok, fin: None)   # callback(token | None, finish_reason | None)
        self.out: List[int] = []
        self.cancelled = False
        self.done = False
        self.t_submit = time.time()
        self.t_first: Optional[float] = None


class Engine:
    def __init__(self, model: HeimdallLM, max_batch: int = 32, max_context: int = 8192,
                 prefill_chunk: int = 2048) -> None:
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.max_batch, self.max_context, self.prefill_chunk = max_batch, max_context, prefill_chunk
        self.pending: "queue.Queue[Request]" = queue.Queue()
        self.active: List[Request] = []
        self.cache: Optional[dict] = None
        self.stats = {"requests": 0, "tokens_generated": 0, "prefill_tokens": 0, "errors": 0}
        self._stop = False
        self._thread = threading.Thread(target=self._loop, name="heimdall-engine", daemon=True)

    # ------------------------------------------------------------------ dış API
    def start(self) -> "Engine":
        self._thread.start()
        return self

    def shutdown(self) -> None:
        self._stop = True
        self._thread.join(timeout=5)

    def submit(self, req: Request) -> Request:
        self.pending.put(req)
        return req

    @property
    def queued(self) -> int:
        return self.pending.qsize()

    # ------------------------------------------------------------------ döngü
    def _loop(self) -> None:
        with torch.inference_mode():
            while not self._stop:
                try:
                    self._admit()
                    if self.active:
                        self._decode()
                except Exception:  # bir hata tüm servisi düşürmesin: aktif istekleri hatayla bitir
                    log.exception("motor adımı başarısız")
                    self.stats["errors"] += 1
                    for r in self.active:
                        r.done = True
                        r.callback(None, "error")
                    self.active, self.cache = [], None
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()

    def _admit(self) -> None:
        block = not self.active
        while len(self.active) < self.max_batch:
            try:
                req = self.pending.get(timeout=0.05) if block else self.pending.get_nowait()
            except queue.Empty:
                return
            block = False
            if req.cancelled:
                continue
            self.stats["requests"] += 1
            ids = req.ids[-(self.max_context - 1):] or [0]
            cache = self.model.new_cache(1)
            x = torch.tensor([ids], device=self.device)
            logits = None
            for s in range(0, x.size(1), self.prefill_chunk):
                logits = self.model(x[:, s:s + self.prefill_chunk], cache=cache, last_only=True)
            self.stats["prefill_tokens"] += len(ids)
            tok = self._sample(logits[:, -1], [req])[0]
            self.cache = cache if self.cache is None else HeimdallLM.cache_merge([self.cache, cache])
            self.active.append(req)
            self._emit(req, tok, len(ids))
            self._prune()

    def _decode(self) -> None:
        last = torch.tensor([[r.out[-1]] for r in self.active], device=self.device)
        logits = self.model(last, cache=self.cache)[:, -1]
        toks = self._sample(logits, self.active)
        for r, t, n in zip(self.active, toks, self.cache["lens"]):
            self._emit(r, t, n)
        self._prune()

    def _emit(self, req: Request, tok: int, ctx_len: int) -> None:
        if req.t_first is None:
            req.t_first = time.time()
        if req.cancelled:
            req.done = True
            return
        if tok in req.stop_ids:
            req.done = True
            req.callback(None, "stop")
            return
        req.out.append(tok)
        self.stats["tokens_generated"] += 1
        req.callback(tok, None)
        if len(req.out) >= req.max_tokens or ctx_len + 1 >= self.max_context:
            req.done = True
            req.callback(None, "length")

    def _prune(self) -> None:
        keep = [i for i, r in enumerate(self.active) if not r.done]
        if len(keep) == len(self.active):
            return
        self.active = [self.active[i] for i in keep]
        self.cache = HeimdallLM.cache_select(self.cache, keep) if keep else None

    def _sample(self, logits: torch.Tensor, reqs: List[Request]) -> List[int]:
        """Satır başına sıcaklık / top-k / top-p; tek senkronizasyon."""
        logits = logits.float()
        if all(r.temperature <= 0 for r in reqs):
            return logits.argmax(-1).tolist()
        b, v = logits.shape
        dev = logits.device
        temp = torch.tensor([max(r.temperature, 1e-5) for r in reqs], device=dev)
        top_p = torch.tensor([r.top_p for r in reqs], device=dev)
        top_k = torch.tensor([r.top_k if r.top_k > 0 else v for r in reqs], device=dev)
        greedy = torch.tensor([r.temperature <= 0 for r in reqs], device=dev)
        probs = torch.softmax(logits / temp[:, None], dim=-1)
        sp, si = probs.sort(dim=-1, descending=True)
        drop = (sp.cumsum(-1) - sp > top_p[:, None]) | (torch.arange(v, device=dev)[None] >= top_k[:, None])
        sp = sp.masked_fill(drop, 0.0)
        pick = torch.multinomial(sp / sp.sum(-1, keepdim=True), 1)
        tok = torch.where(greedy, logits.argmax(-1), si.gather(-1, pick).squeeze(-1))
        return tok.tolist()
