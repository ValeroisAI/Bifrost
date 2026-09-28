"""
Eğitim checkpoint'inden servis paketi üretir (bf16 safetensors + config + tokenizer).

    python -m heimdall.export --ckpt runs/temel/last.pt --out models/heimdall-temel
"""

import argparse

import torch

from .io import DEFAULT_TOKENIZER, load_model, save_bundle


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    args = p.parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = load_model(args.ckpt)
    meta = {"step": ckpt.get("step"), "tokens": ckpt.get("tokens"), "source": args.ckpt,
            "arch": ckpt.get("args", {}).get("arch"), "params": model.num_params()}
    out = save_bundle(model, args.out, args.tokenizer, meta,
                      torch.bfloat16 if args.dtype == "bf16" else torch.float32)
    print(f"Servis paketi: {out} ({meta['params'] / 1e6:.1f}M param, {args.dtype})")


if __name__ == "__main__":
    main()
