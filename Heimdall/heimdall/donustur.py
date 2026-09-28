"""
donustur.py — Hazır bir Transformer'ı (Llama / Qwen2 / Qwen3 / SmolLM ailesi) sabit bellekli modele dönüştürür.

Fikir: 1000 kat hesap farkını sıfırdan eğitimle kapatamayız, ama o hesap açık ağırlıklarda zaten harcanmış.
Her dikkat katmanı şu katmana dönüştürülür (öğretmenin q/k/v/o ağırlıkları aynen kullanılır):

    o_t = o_pencere + γ_t ⊙ (ρ ⊙ o_hafıza − o_pencere)

    o_pencere : softmax dikkat, yalnız son W token + ilk S "çapa" token (StreamingLLM). t < W iken öğretmenle birebir aynı.
    o_hafıza  : kapılı delta kuralı. Token j, pencereden çıktığı anda (t = j + W) hafızaya yazılır; hafıza ve pencere
                ayrık, hiçbir token iki kez sayılmaz.  S = αS + β k̂(v − αSᵀk̂)ᵀ,  k̂ = norm(F_k k),  q̂ = norm(F_q q)
    γ_t       : pencere dışına düşen dikkat kütlesinin tahmini; girdiye ve pencere içi log-normalizöre (lse) bakar.
                t < W iken 0.

Çıkarım belleği: katman başına W+S anahtar + sabit boyutlu S matrisi. Bağlam uzunluğundan bağımsız.

Eğitim iki aşamalıdır (yalnız yeni parametreler, model boyutunun ~%2'si):
    A) Katman katman: her katman öğretmenin gizli durumlarıyla beslenir, öğretmenin dikkat çıktısını taklit eder.
    B) Uçtan uca: öğretmenin token olasılıklarına KL damıtma.

    # model ve veri otomatik iner (Hugging Face), GPU varsa bf16 ile GPU'da çalışır:
    python -m heimdall.donustur --model HuggingFaceTB/SmolLM2-135M --wikitext veri/wikitext --out donusum
"""

import argparse
import contextlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernels import RMSNorm, delta_rule, rope


class TeacherConfig:
    def __init__(self, hf: dict) -> None:
        self.d = hf["hidden_size"]
        self.n_layers = hf["num_hidden_layers"]
        self.n_heads = hf["num_attention_heads"]
        self.n_kv = hf.get("num_key_value_heads", self.n_heads)
        self.head_dim = hf.get("head_dim") or self.d // self.n_heads
        self.inter = hf["intermediate_size"]
        self.rope_theta = hf.get("rope_theta", 10000.0)
        self.eps = hf.get("rms_norm_eps", 1e-6)
        self.vocab = hf["vocab_size"]
        self.tie = hf.get("tie_word_embeddings", False)
        self.qk_norm = hf.get("model_type") == "qwen3"
        self.attn_bias = hf.get("attention_bias", hf.get("model_type") == "qwen2")
        if hf.get("rope_scaling"):
            raise NotImplementedError("rope_scaling henüz desteklenmiyor")


class ConvAttention(nn.Module):
    def __init__(self, tc: TeacherConfig) -> None:
        super().__init__()
        self.tc, self.h, self.hkv, self.dh = tc, tc.n_heads, tc.n_kv, tc.head_dim
        self.q_proj = nn.Linear(tc.d, self.h * self.dh, bias=tc.attn_bias)
        self.k_proj = nn.Linear(tc.d, self.hkv * self.dh, bias=tc.attn_bias)
        self.v_proj = nn.Linear(tc.d, self.hkv * self.dh, bias=tc.attn_bias)
        self.o_proj = nn.Linear(self.h * self.dh, tc.d, bias=False)
        if tc.qk_norm:
            self.q_norm = RMSNorm(self.dh, tc.eps)
            self.k_norm = RMSNorm(self.dh, tc.eps)
        self.converted = False
        self.mode = "teacher"          # teacher | student | layer (aşama A) | window (yalnız pencere, karşılaştırma)
        self.aux = None

    def convert(self, window: int, sinks: int, archival: int) -> None:
        h, dh = self.h, self.dh
        self.window, self.sinks = window, sinks
        self.fq = nn.Parameter(torch.eye(dh).repeat(h, 1, 1))
        self.fk = nn.Parameter(torch.eye(dh).repeat(h, 1, 1))
        self.ab = nn.Linear(self.tc.d, 2 * h)
        self.mix = nn.Linear(self.tc.d, h)
        for lin in (self.ab, self.mix):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        nn.init.constant_(self.mix.bias, -2.0)
        self.mix_lse = nn.Parameter(torch.zeros(h))
        self.mem_scale = nn.Parameter(torch.ones(h))
        self.A_log = nn.Parameter(torch.empty(h).uniform_(1.0, 16.0).log())
        dt = torch.exp(torch.empty(h).uniform_(math.log(1e-3), math.log(1e-1)))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        mask = torch.ones(h)
        mask[:archival] = 0.0
        self.register_buffer("decay_mask", mask, persistent=False)
        self.converted = True

    def new_params(self):
        return [p for n, p in self.named_parameters() if not n.startswith(("q_proj", "k_proj", "v_proj", "o_proj",
                                                                           "q_norm", "k_norm"))]

    def _qkv(self, x):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.h, self.dh)
        k = self.k_proj(x).view(b, t, self.hkv, self.dh)
        v = self.v_proj(x).view(b, t, self.hkv, self.dh)
        if self.tc.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        rep = self.h // self.hkv
        k, v = k.repeat_interleave(rep, dim=2), v.repeat_interleave(rep, dim=2)
        pos = torch.arange(t, device=x.device)
        qr = rope(q.transpose(1, 2), pos, self.tc.rope_theta)
        kr = rope(k.transpose(1, 2), pos, self.tc.rope_theta)
        return q, k, v, qr, kr

    def _full(self, qr, kr, v):
        return F.scaled_dot_product_attention(qr, kr, v.transpose(1, 2), is_causal=True).transpose(1, 2)

    def _window_attn(self, qr, kr, v):
        """Pencere + çapa softmax dikkati, blok-yerel: bellek O(T·(2W+S)). Döndürür o [B,T,H,D], lse [B,H,T]."""
        b, h, t, d = qr.shape
        w, s = self.window, min(self.sinks, t)
        pad = (-t) % w
        n = (t + pad) // w
        vt = v.transpose(1, 2)
        qb, kb, vb = (F.pad(z.float(), (0, 0, 0, pad)).view(b, h, n, w, d) for z in (qr, kr, vt))
        prev = lambda z: F.pad(z, (0, 0, 0, 0, 1, 0))[:, :, :-1]
        sink = lambda z: z[:, :, None, :s].float().expand(b, h, n, s, d)
        keys = torch.cat((sink(kr), prev(kb), kb), dim=3)                           # [B,H,N,S+2W,D]
        vals = torch.cat((sink(vt), prev(vb), vb), dim=3)
        blk = torch.arange(n, device=qr.device)[:, None, None]
        qpos = blk * w + torch.arange(w, device=qr.device)[None, :, None]           # [N,W,1]
        j = torch.arange(w, device=qr.device)
        kpos = torch.cat((torch.arange(s, device=qr.device).expand(n, s),
                          (blk[:, 0] - 1) * w + j, blk[:, 0] * w + j), dim=1)[:, None, :]   # [N,1,S+2W]
        is_sink = torch.arange(s + 2 * w, device=qr.device) < s
        ok = (kpos >= 0) & (kpos <= qpos) & (is_sink | ((qpos - kpos < w) & (kpos >= s)))
        scores = (qb @ keys.transpose(-1, -2)) * self.dh ** -0.5
        scores = scores.masked_fill(~ok, float("-inf"))
        lse = torch.logsumexp(scores, dim=-1)
        o = torch.exp(scores - lse[..., None]) @ vals
        o = o.reshape(b, h, n * w, d)[:, :, :t].transpose(1, 2)
        return o, lse.reshape(b, h, n * w)[:, :, :t]

    def _hybrid(self, x, q, k, v, qr, kr, memory: bool = True):
        b, t = x.shape[:2]
        w, s = self.window, self.sinks
        i = torch.arange(t, device=x.device)
        o_win, lse = self._window_attn(qr, kr, v)
        if not memory or t <= w:
            return o_win
        qm = F.normalize(torch.einsum("bthd,hde->bthe", q.float(), self.fq), dim=-1) * self.dh ** -0.5
        km = F.normalize(torch.einsum("bthd,hde->bthe", k.float(), self.fk), dim=-1)
        a, bb = self.ab(x).float().chunk(2, dim=-1)
        g = -self.A_log.exp() * F.softplus(a + self.dt_bias) * self.decay_mask
        beta = torch.sigmoid(bb) * (i >= s).float()[None, :, None]                  # çapa tokenler yazılmaz
        shift = lambda z: F.pad(z, (0, 0) * (z.dim() - 2) + (w, 0))[:, :t]           # j, t = j + W'de yazılır
        o_mem, _ = delta_rule(qm, shift(km), shift(v.float()), shift(g), shift(beta), 64, backend="torch")
        gamma = torch.sigmoid(self.mix(x).float() + self.mix_lse * (lse.transpose(1, 2) - math.log(w)))
        gamma = gamma * (i >= w).float()[None, :, None]
        return o_win + gamma[..., None] * (self.mem_scale[:, None] * o_mem - o_win)

    def forward(self, x):
        b, t, _ = x.shape
        q, k, v, qr, kr = self._qkv(x)
        if not self.converted or self.mode == "teacher":
            o = self._full(qr, kr, v)
        elif self.mode == "layer":  # artık akış öğretmenden; öğrenci dalı yan hesaplanır
            with torch.no_grad():
                o = self._full(qr, kr, v)
            o_s = self._hybrid(x, q, k, v, qr, kr)
            self.aux = (o_s - o.float()).square().mean() / o.float().square().mean().clamp(min=1e-8)
        else:
            o = self._hybrid(x, q, k, v, qr, kr, memory=self.mode == "student")
        return self.o_proj(o.reshape(b, t, -1).to(x.dtype))


class Layer(nn.Module):
    def __init__(self, tc: TeacherConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(tc.d, tc.eps)
        self.self_attn = ConvAttention(tc)
        self.post_attention_layernorm = RMSNorm(tc.d, tc.eps)
        self.gate_proj = nn.Linear(tc.d, tc.inter, bias=False)
        self.up_proj = nn.Linear(tc.d, tc.inter, bias=False)
        self.down_proj = nn.Linear(tc.inter, tc.d, bias=False)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        h = self.post_attention_layernorm(x)
        return x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))


class ConvLM(nn.Module):
    def __init__(self, tc: TeacherConfig) -> None:
        super().__init__()
        self.tc = tc
        self.embed_tokens = nn.Embedding(tc.vocab, tc.d)
        self.layers = nn.ModuleList([Layer(tc) for _ in range(tc.n_layers)])
        self.norm = RMSNorm(tc.d, tc.eps)
        self.lm_head = nn.Linear(tc.d, tc.vocab, bias=False)
        if tc.tie:
            self.lm_head.weight = self.embed_tokens.weight

    @classmethod
    def from_pretrained(cls, path) -> "ConvLM":
        from safetensors.torch import load_file
        p = Path(path)
        model = cls(TeacherConfig(json.loads((p / "config.json").read_text())))
        state = {}
        for f in sorted(p.glob("*.safetensors")):
            for k, v in load_file(str(f)).items():
                state[k.removeprefix("model.").replace("mlp.", "")] = v.float()
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [m for m in missing if not (model.tc.tie and m == "lm_head.weight")]
        assert not missing and not unexpected, (missing, unexpected)
        return model

    def convert(self, window: int = 64, sinks: int = 4, archival_frac: float = 0.25, keep_global=()) -> None:
        for i, layer in enumerate(self.layers):
            if i not in keep_global:
                layer.self_attn.convert(window, sinks, int(archival_frac * self.tc.n_heads))

    def set_mode(self, mode: str) -> None:
        for layer in self.layers:
            layer.self_attn.mode = mode

    def new_params(self):
        return [p for layer in self.layers if layer.self_attn.converted for p in layer.self_attn.new_params()]

    def forward(self, ids):
        x = self.embed_tokens(ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x)).float()

    def aux_loss(self):
        auxes = [l.self_attn.aux for l in self.layers if l.self_attn.aux is not None]
        return sum(auxes) / len(auxes)


# --------------------------------------------------------------------------- veri ve ölçüm
def eval_windows(path, t: int, n: int, sep: int):
    """Makale başından başlayan, en az t+1 uzunluklu pencereler (bağlam gerçekten büyür)."""
    data = np.fromfile(path, dtype=np.uint16).astype(np.int64)
    starts = [0] + [i + 1 for i in np.nonzero(data == sep)[0][:-1]]
    ends = list(np.nonzero(data == sep)[0])
    wins = [torch.from_numpy(data[s:s + t + 1]) for s, e in zip(starts, ends) if e - s > t]
    return torch.stack(wins[:n])


@torch.no_grad()
def evaluate(model, wins, mode: str, buckets):
    model.set_mode(mode)
    nll = torch.zeros(wins.size(1) - 1, device=wins.device)
    for w in wins:
        logits = model(w[None, :-1])
        nll += F.cross_entropy(logits[0], w[1:], reduction="none")
    nll /= wins.size(0)
    out = {"ppl": math.exp(nll.mean().item())}
    for a, b in buckets:
        out[f"{a}-{b}"] = math.exp(nll[a:b].mean().item())
    return out


def fetch_model(name: str) -> str:
    if Path(name).exists():
        return name
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=name, allow_patterns=["*.json", "*.safetensors"])


def prepare_wikitext(model_dir: str, out_dir: str, train_tokens: int = 20_000_000, sep: int = 0):
    """wikitext-103'ü indirip öğretmenin tokenizer'ıyla .bin dosyalarına çevirir (makaleler sep ile ayrılır)."""
    import re

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {"test": ["test-00000-of-00001"], "train": ["train-00000-of-00002", "train-00001-of-00002"]}
    if all((out / f"wiki_{k}.bin").exists() for k in files):
        return str(out / "wiki_train.bin"), str(out / "wiki_test.bin")
    tok = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
    for split, names in files.items():
        ids = []
        for name in names:
            path = hf_hub_download("Salesforce/wikitext", f"wikitext-103-raw-v1/{name}.parquet", repo_type="dataset")
            cur = []
            for line in pq.read_table(path).column("text").to_pylist() + [" = END = \n"]:
                if re.match(r"^ = [^=].* = \n$", line) and cur:
                    ids.extend(tok.encode("".join(cur)).ids + [sep])
                    cur = []
                cur.append(line)
            if split == "train" and len(ids) >= train_tokens:
                break
        np.array(ids, dtype=np.uint16 if tok.get_vocab_size() < 65536 else np.uint32).tofile(out / f"wiki_{split}.bin")
        print(f"wikitext {split}: {len(ids):,} token", flush=True)
    return str(out / "wiki_train.bin"), str(out / "wiki_test.bin")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF model klasörü ya da HF adı (ör. HuggingFaceTB/SmolLM2-135M)")
    p.add_argument("--train", help="eğitim .bin (öğretmen tokenizer'ı)")
    p.add_argument("--test", help="test .bin (makaleler --sep ile ayrılmış)")
    p.add_argument("--wikitext", help="wikitext-103'ü bu klasöre hazırla ve kullan")
    p.add_argument("--wikitext-tokens", type=float, default=20e6)
    p.add_argument("--device", default=None, help="cuda (ROCm dahil) | cpu; boşsa otomatik")
    p.add_argument("--out", default="donusum")
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--sinks", type=int, default=4)
    p.add_argument("--archival-frac", type=float, default=0.25)
    p.add_argument("--keep-global", type=int, nargs="*", default=[], help="dönüştürülmeyecek katmanlar")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--steps-a", type=int, default=100)
    p.add_argument("--steps-b", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--eval-len", type=int, default=2048)
    p.add_argument("--eval-n", type=int, default=12)
    p.add_argument("--sep", type=int, default=0, help="makale ayırıcı token")
    p.add_argument("--threads", type=int, default=0)
    args = p.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    args.model = fetch_model(args.model)
    if args.wikitext:
        args.train, args.test = prepare_wikitext(args.model, args.wikitext, int(args.wikitext_tokens), args.sep)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    gpu = device.type == "cuda"
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if gpu else contextlib.nullcontext()

    model = ConvLM.from_pretrained(args.model)
    model.convert(args.window, args.sinks, args.archival_frac, set(args.keep_global))
    for q in model.parameters():
        q.requires_grad_(False)
    params = model.new_params()
    new_ids = {id(q) for q in params}
    for q in model.parameters():
        if gpu and id(q) not in new_ids:
            q.data = q.data.to(torch.bfloat16)   # donmuş öğretmen ağırlıkları bf16: yarı bellek, hızlı matmul
    for q in params:
        q.requires_grad_(True)
    model.to(device)
    n_new = sum(q.numel() for q in params)
    tc = model.tc
    kv_teacher = lambda n: tc.n_layers * 2 * tc.n_kv * tc.head_dim * n * 2
    state = sum(tc.n_heads * tc.head_dim ** 2 * 4 + 2 * tc.n_kv * tc.head_dim * (args.window + args.sinks) * 2
                for i in range(tc.n_layers) if i not in args.keep_global)
    print(f"Öğretmen {sum(q.numel() for q in model.parameters()) / 1e6:.0f}M param, {tc.n_layers} katman | yeni param "
          f"{n_new / 1e6:.2f}M | pencere {args.window} + {args.sinks} çapa | bellek: öğretmen 8K'da "
          f"{kv_teacher(8192) / 2**20:.0f} MB, 1M'de {kv_teacher(2**20) / 2**30:.1f} GB → dönüşüm {state / 2**20:.1f} MB (sabit)",
          flush=True)

    wins = eval_windows(args.test, args.eval_len, args.eval_n, args.sep).to(device)
    L = args.eval_len
    buckets = [(a, min(b, L)) for a, b in ((0, args.window), (args.window, 256), (256, 1024), (1024, L)) if a < min(b, L)]
    data = np.memmap(args.train, dtype=np.uint16 if model.tc.vocab < 65536 else np.uint32, mode="r")
    rng = np.random.default_rng(0)

    def batch():
        s = rng.integers(0, len(data) - args.seq_len - 1, args.batch)
        return torch.from_numpy(np.stack([data[i:i + args.seq_len] for i in s]).astype(np.int64)).to(device)

    results = {"args": vars(args), "eval_windows": int(wins.size(0))}
    t0 = time.time()
    with amp:
        results["ogretmen"] = evaluate(model, wins, "teacher", buckets)
        results["yalniz_pencere"] = evaluate(model, wins, "window", buckets)
        results["donusum_egitimsiz"] = evaluate(model, wins, "student", buckets)
    for k in ("ogretmen", "yalniz_pencere", "donusum_egitimsiz"):
        print(f"{k:22s} " + " | ".join(f"{b}: {v:.2f}" for b, v in results[k].items()), flush=True)

    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    for stage, steps in (("A", args.steps_a), ("B", args.steps_b)):
        model.set_mode("layer" if stage == "A" else "student")
        for g in opt.param_groups:
            g["lr"] = args.lr
        for step in range(1, steps + 1):
            for g in opt.param_groups:  # kosinüs düşüş
                g["lr"] = args.lr * 0.5 * (1 + math.cos(math.pi * step / steps))
            x = batch()
            with amp:
                if stage == "A":
                    model(x)
                    loss = model.aux_loss()
                else:
                    with torch.no_grad():
                        model.set_mode("teacher")
                        t_logp = F.log_softmax(model(x), dim=-1)
                        model.set_mode("student")
                    s_logp = F.log_softmax(model(x), dim=-1)
                    loss = F.kl_div(s_logp, t_logp, log_target=True, reduction="batchmean") / x.size(1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if step % 10 == 0 or step == steps:
                print(f"  aşama {stage} adım {step}/{steps} | kayıp {loss.item():.4f} | "
                      f"{step * x.numel() / (time.time() - t0):,.0f} tok/s", flush=True)
        t0 = time.time()
        key = f"donusum_asama_{stage}"
        with amp:
            results[key] = evaluate(model, wins, "student", buckets)
        results[key]["egitim_token"] = (args.steps_a + (args.steps_b if stage == "B" else 0)) * args.batch * args.seq_len
        print(f"{key:22s} " + " | ".join(f"{b}: {v:.2f}" for b, v in results[key].items()), flush=True)
        (out / "sonuc.json").write_text(json.dumps(results, indent=2))
    torch.save({n: q for n, q in model.named_parameters() if q.requires_grad}, out / "donusum_param.pt")
    print(f"Kaydedildi: {out}", flush=True)


if __name__ == "__main__":
    main()
