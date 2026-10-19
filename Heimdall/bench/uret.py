"""
Eğitilmiş Heimdall modelinden metin üretir (sabit/küçük cache ile, token token).

    python bench/uret.py --ckpt runs/gece/last.pt
    python bench/uret.py --ckpt runs/gece/last.pt --prompt "def fibonacci(n):\n" --max-new 200 --temp 0.7
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heimdall.io import DEFAULT_TOKENIZER, load_model  # noqa: E402

DEFAULTS = ["def fibonacci(n):\n", "The most important reason why the sky is blue is", "class Stack:\n    def __init__(self):\n"]


def sample(logits, temp, top_k, top_p):
    if temp <= 0:
        return int(logits.argmax())
    logits = logits.float() / temp
    if top_k:
        kth = logits.topk(min(top_k, logits.numel())).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = torch.softmax(logits, -1)
    if top_p < 1.0:
        sp, si = probs.sort(descending=True)
        sp = sp.masked_fill(sp.cumsum(-1) - sp > top_p, 0.0)
        probs = torch.zeros_like(probs).scatter(-1, si, sp)
    return int(torch.multinomial(probs / probs.sum(), 1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt", action="append", help="birden fazla verilebilir (\\n kaçışı çalışır)")
    p.add_argument("--max-new", type=int, default=150)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(args.ckpt, dev)          # ağırlıklar kayıtlı tipte (eğitimdeki gibi); matmul bf16 autocast
    tok = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    eos = tok.token_to_id("<eos>")
    amp = torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")
    print(f"{model.num_params() / 1e6:.1f}M param, bits={model.cfg.bits}, düzen {model.cfg.layout}")
    for prompt in [s.encode().decode("unicode_escape") for s in args.prompt] if args.prompt else DEFAULTS:
        ids = torch.tensor([tok.encode(prompt).ids], device=dev)
        out, printed = [], ""
        print("=" * 70 + "\n" + prompt, end="", flush=True)
        with torch.no_grad(), amp:
            cache = model.new_cache(1)
            logits = model(ids, cache=cache, last_only=True)[0, -1]
            t0 = time.time()
            for _ in range(args.max_new):
                nxt = sample(logits, args.temp, args.top_k, args.top_p)
                if nxt == eos:
                    break
                out.append(nxt)
                text = tok.decode(out)
                if not text.endswith("�"):
                    print(text[len(printed):], end="", flush=True)
                    printed = text
                logits = model(torch.tensor([[nxt]], device=dev), cache=cache)[0, -1]
        n = max(len(out), 1)
        print(f"\n[{len(out)} token, {n / (time.time() - t0):.0f} tok/s, cache {model.cache_bytes(cache) / 2**20:.1f} MB]")


if __name__ == "__main__":
    main()
