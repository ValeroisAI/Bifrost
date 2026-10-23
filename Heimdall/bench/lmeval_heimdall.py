"""
Eğitilmiş Heimdall checkpoint'ini lm-evaluation-harness ile standart testlerde ölçer (sıfır-örnek).

    pip install lm-eval            # bir kez
    python bench/lmeval_heimdall.py --ckpt runs/hafta/model/last.pt                       # varsayılan görevler
    python bench/lmeval_heimdall.py --ckpt runs/hafta/model/last.pt --tasks arc_easy,piqa --limit 200   # hızlı deneme
    python bench/lmeval_heimdall.py --ckpt runs/hafta/model/last.pt --out runs/hafta/lmeval.json

Beklenti: 337 M parametre / 3.7 B token base model; küçük public modeller (Pythia, GPT-2) 100× fazla token görmüştür.
Çoğu görevde şansa yakın çıkması NORMALDİR; sonuçlar "şans düzeyi"yle birlikte yazdırılır. wikitext word_perplexity
tokenizer'dan bağımsızdır (kelime başına), yani public modellerle karşılaştırılabilir TEK sayıdır; token başına kayıplar
(8K sözlük) karşılaştırılamaz.

Uygulama lm-eval'in HF modeliyle aynı kuralları izler: bağlam sonundaki boşluk devama taşınır, tam metin ve bağlam ayrı
kodlanır, rolling'de <eos> öneki. Sağ-dolgulu toplu iş (nedensel model: dolgu önceki konumları etkilemez).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heimdall.io import DEFAULT_TOKENIZER, load_model  # noqa: E402

CHANCE = {"arc_easy": 0.25, "arc_challenge": 0.25, "hellaswag": 0.25, "piqa": 0.5, "winogrande": 0.5, "boolq": 0.5,
          "openbookqa": 0.25, "sciq": 0.25}
DEFAULT_TASKS = "arc_easy,arc_challenge,hellaswag,piqa,winogrande,lambada_openai,wikitext"


def make_lm(model, tok: Tokenizer, dev, max_len: int = 2048, batch_size: int = 16):
    from lm_eval.api.model import LM

    eos = tok.token_to_id("<eos>")
    amp = torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")

    def encode_pair(context: str, continuation: str):
        """lm-eval HF kuralı: bağlamın sondaki boşluğu devama geçer; devam = tam kodlamanın bağlamdan sonrası."""
        n = len(context) - len(context.rstrip())
        if n:
            context, continuation = context[:-n], context[-n:] + continuation
        whole = tok.encode(context + continuation).ids
        ctx = tok.encode(context).ids
        if whole[:len(ctx)] == ctx and len(whole) > len(ctx):
            return ctx, whole[len(ctx):]
        return ctx, tok.encode(continuation).ids                 # sınırda birleşme olduysa ayrı kodla

    class HeimdallEval(LM):
        def __init__(self):
            super().__init__()

        @torch.no_grad()
        def _score(self, seqs):
            """seqs: [(ctx_ids, cont_ids)] → [(toplam logp, açgözlü mü)]. Uzunluğa göre sıralı, sağ dolgulu toplu iş."""
            items = []
            for i, (c, k) in enumerate(seqs):
                ids = ((c or [eos]) + k)[-(max_len + 1):]
                items.append((i, ids, len(k)))
            out = [None] * len(seqs)
            for b in range(0, len(items), batch_size):
                chunk = sorted(items[b:b + batch_size], key=lambda it: -len(it[1]))
                width = max(len(ids) - 1 for _, ids, _ in chunk)
                x = torch.zeros(len(chunk), width, dtype=torch.long)
                for j, (_, ids, _) in enumerate(chunk):
                    x[j, :len(ids) - 1] = torch.tensor(ids[:-1])
                with amp:
                    logits = model(x.to(dev))
                for j, (i, ids, k) in enumerate(chunk):
                    n = len(ids) - 1
                    lp = F.log_softmax(logits[j, n - k:n].float(), dim=-1)
                    tgt = torch.tensor(ids[-k:], device=dev)
                    out[i] = (float(lp.gather(-1, tgt[:, None]).sum()), bool((lp.argmax(-1) == tgt).all()))
            return out

        def loglikelihood(self, requests, disable_tqdm: bool = False):
            seqs = []
            for req in requests:
                c, k = encode_pair(*req.args)
                seqs.append((c, k if k else [eos]))
            return self._score(seqs)

        def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
            from lm_eval.utils import get_rolling_token_windows, make_disjoint_window
            out = []
            for req in requests:
                ids = tok.encode(req.args[0]).ids
                wins = list(map(make_disjoint_window, get_rolling_token_windows(token_list=ids, prefix_token=eos,
                                                                              max_seq_len=max_len, context_len=1)))
                scored = self._score([(c, k) for c, k in wins])
                out.append(sum(s for s, _ in scored))
            return out

        @torch.no_grad()
        def generate_until(self, requests, disable_tqdm: bool = False):
            out = []
            for req in requests:
                ctx, kw = req.args
                until = kw.get("until", []) or []
                until = [until] if isinstance(until, str) else until
                cache = model.new_cache(1)
                with amp:
                    logits = model(torch.tensor([(tok.encode(ctx).ids or [eos])[-max_len:]], device=dev), cache=cache,
                                   last_only=True)[0, -1]
                    gen, text = [], ""
                    for _ in range(kw.get("max_gen_toks", 128)):
                        nxt = int(logits.argmax())
                        if nxt == eos:
                            break
                        gen.append(nxt)
                        text = tok.decode(gen)
                        if any(u in text for u in until):
                            text = text[:min(text.index(u) for u in until if u in text)]
                            break
                        logits = model(torch.tensor([[nxt]], device=dev), cache=cache)[0, -1]
                out.append(text)
            return out

    return HeimdallEval()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tasks", default=DEFAULT_TASKS)
    p.add_argument("--limit", type=int, help="görev başına örnek sınırı (hızlı deneme)")
    p.add_argument("--fewshot", type=int, default=0)
    p.add_argument("--max-len", type=int, default=2048, help="eğitim uzunluğu 2048")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--bootstrap", type=int, default=1000, help="standart hata için bootstrap sayısı")
    p.add_argument("--out")
    a = p.parse_args()

    import lm_eval
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(a.ckpt, dev)
    tok = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    t0 = time.time()
    res = lm_eval.simple_evaluate(model=make_lm(model, tok, dev, a.max_len, a.batch), tasks=a.tasks.split(","),
                                  limit=a.limit, num_fewshot=a.fewshot, bootstrap_iters=a.bootstrap)
    table = {t: {k: round(v, 4) for k, v in r.items() if isinstance(v, (int, float))} for t, r in res["results"].items()}
    print(f"\n{model.num_params() / 1e6:.0f}M param | {a.ckpt} | {a.fewshot}-shot | limit {a.limit} | {time.time() - t0:.0f} sn\n")
    for t, r in table.items():
        extra = f"  (şans: {CHANCE[t]:.0%})" if t in CHANCE else ""
        print(f"{t:18s} " + "  ".join(f"{k.split(',')[0]}={v}" for k, v in r.items() if "stderr" not in k) + extra)
    meta = {"ckpt": a.ckpt, "fewshot": a.fewshot, "limit": a.limit, "max_len": a.max_len, "results": table, "chance": CHANCE}
    if a.out:
        Path(a.out).write_text(json.dumps(meta, indent=2))
        print(f"\nYazıldı: {a.out}")


if __name__ == "__main__":
    main()
