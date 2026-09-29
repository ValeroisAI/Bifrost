"""
bifrost_eval.py — Dönüştürülmüş modelin ölçümü.

  Şifre bulma (uzun bağlam hafızası): uzun anlamsız metnin içine bir sayı gizlenir, sonda sorulur.
    python -m heimdall.bifrost_eval sifre --model HuggingFaceTB/SmolLM2-135M --conv kosular/x/donusum_param.pt \
        --lengths 1000 4000 16000 64000 --n 5
  Standart benchmarklar (lm-eval-harness; pip install lm-eval):
    python -m heimdall.bifrost_eval lmeval --model HuggingFaceTB/SmolLM2-135M --conv ... --tasks arc_easy,hellaswag
  --conv verilmezse öğretmen (orijinal model) ölçülür; --mode window yalnız pencereyi ölçer.
"""

import argparse
import json
import random
import time

import torch
import torch.nn.functional as F

from .donustur import ConvLM, fetch_model

FILLER = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "


def load(model_name, conv=None, mode=None, device=None):
    from pathlib import Path

    from tokenizers import Tokenizer

    path = fetch_model(model_name)
    model = ConvLM.load_converted(path, conv) if conv else ConvLM.from_pretrained(path)
    if mode:
        model.set_mode(mode)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if dev.type == "cuda":
        model = model.to(torch.bfloat16)
    return model.to(dev).eval(), Tokenizer.from_file(str(Path(path) / "tokenizer.json")), dev


@torch.no_grad()
def passkey(model, tok, dev, length: int, depth: float, seed: int):
    rng = random.Random(seed)
    key = str(rng.randint(10000, 99999))
    head = "There is important info hidden inside a lot of irrelevant text. Find it and memorize it.\n"
    needle = f" The pass key is {key}. Remember it. {key} is the pass key. "
    tail = "\nWhat is the pass key? The pass key is"
    n_fill = max(0, length - len(tok.encode(head + needle + tail).ids))
    unit = tok.encode(FILLER).ids
    filler = (unit * (n_fill // len(unit) + 1))[:n_fill]
    cut = int(len(filler) * depth)
    ids = tok.encode(head).ids + filler[:cut] + tok.encode(needle).ids + filler[cut:] + tok.encode(tail).ids
    out, cache = model.generate(torch.tensor([ids], device=dev), max_new=8)
    text = tok.decode(out)
    return key in text, len(ids), text.strip(), ConvLM.cache_bytes(cache)


def run_passkey(args) -> None:
    model, tok, dev = load(args.model, args.conv, args.mode, args.device)
    res = []
    for length in args.lengths:
        for depth in args.depths:
            hits, t0 = 0, time.time()
            for s in range(args.n):
                ok, n_tok, text, mem = passkey(model, tok, dev, length, depth, s)
                hits += ok
            res.append({"uzunluk": n_tok, "derinlik": depth, "basari": hits / args.n, "cache_mb": mem / 2**20,
                        "sure_s": round(time.time() - t0, 1), "ornek": text})
            print(f"{n_tok:>7} token | derinlik {depth:.0%} | başarı {hits}/{args.n} | cache {mem / 2**20:.1f} MB"
                  f" | örnek çıktı: {text!r}", flush=True)
    if args.out:
        open(args.out, "w").write(json.dumps(res, indent=2, ensure_ascii=False))


def make_lm(model, tok, dev, max_len: int = 4096):
    from lm_eval.api.model import LM

    class BifrostLM(LM):
        def __init__(self):
            super().__init__()

        def _logp(self, ids):
            x = torch.tensor([ids[-(max_len + 1):]], device=dev)
            return F.log_softmax(model(x[:, :-1]).float(), dim=-1)[0], x[0, 1:]

        @torch.no_grad()
        def loglikelihood(self, requests):
            out = []
            for req in requests:
                ctx, cont = req.args
                c, k = tok.encode(ctx).ids or [0], tok.encode(cont).ids
                logp, tgt = self._logp(c + k)
                lp, tg = logp[-len(k):], tgt[-len(k):]
                out.append((float(lp.gather(-1, tg[:, None]).sum()), bool((lp.argmax(-1) == tg).all())))
            return out

        @torch.no_grad()
        def loglikelihood_rolling(self, requests):
            out = []
            for req in requests:
                ids, total = [0] + tok.encode(req.args[0]).ids, 0.0
                for s in range(0, len(ids) - 1, max_len):
                    logp, tgt = self._logp(ids[s:s + max_len + 1])
                    total += float(logp.gather(-1, tgt[:, None]).sum())
                out.append(total)
            return out

        @torch.no_grad()
        def generate_until(self, requests):
            out = []
            for req in requests:
                ctx, kw = req.args
                until = kw.get("until", []) or []
                until = [until] if isinstance(until, str) else until
                cache = model.new_cache()
                logits = model(torch.tensor([tok.encode(ctx).ids[-max_len:] or [0]], device=dev), cache, last_only=True)
                gen, text = [], ""
                for _ in range(kw.get("max_gen_toks", 256)):
                    nxt = int(logits[0, -1].argmax())
                    gen.append(nxt)
                    text = tok.decode(gen)
                    if any(u in text for u in until):
                        text = text[:min(text.index(u) for u in until if u in text)]
                        break
                    logits = model(torch.tensor([[nxt]], device=dev), cache)
                out.append(text)
            return out

    return BifrostLM()


def run_lmeval(args) -> None:
    import lm_eval

    model, tok, dev = load(args.model, args.conv, args.mode, args.device)
    res = lm_eval.simple_evaluate(model=make_lm(model, tok, dev, args.max_len), tasks=args.tasks.split(","),
                                  limit=args.limit, num_fewshot=args.fewshot)
    table = {t: {k: v for k, v in r.items() if isinstance(v, (int, float))} for t, r in res["results"].items()}
    print(json.dumps(table, indent=2))
    if args.out:
        open(args.out, "w").write(json.dumps(table, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("sifre", "lmeval"):
        a = sub.add_parser(name)
        a.add_argument("--model", required=True)
        a.add_argument("--conv", help="donusum_param.pt (yoksa orijinal model)")
        a.add_argument("--mode", choices=["student", "window", "teacher"])
        a.add_argument("--device")
        a.add_argument("--out")
        if name == "sifre":
            a.add_argument("--lengths", type=int, nargs="+", default=[1000, 4000, 16000])
            a.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.5])
            a.add_argument("--n", type=int, default=5)
        else:
            a.add_argument("--tasks", default="arc_easy,hellaswag,piqa")
            a.add_argument("--limit", type=int)
            a.add_argument("--fewshot", type=int)
            a.add_argument("--max-len", type=int, default=4096)
    args = p.parse_args()
    run_passkey(args) if args.cmd == "sifre" else run_lmeval(args)


if __name__ == "__main__":
    main()
