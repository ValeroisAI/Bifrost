"""
damit.py — Bilgi damıtma: büyük bir modelin bilgisini küçük Heimdall modeline aktarır.

Normal eğitimde öğrenci her pozisyonda tek bir doğru token görür (sert etiket). Damıtmada öğretmenin
tüm olasılık dağılımını görür ("Paris %90, Lyon %4, Fransa %2 ..."): hangi cevapların yakın, hangilerinin
imkânsız olduğu bilgisi. Token başına öğrenilen bilgi çok daha fazladır.

Verimlilik: öğretmen yalnız BİR KEZ çalışır. Her pozisyon için en olası k token ve log-olasılıkları diske yazılır
(k=32 → token başına ~130 bayt). Öğrenci bu dosyadan istediği kadar dönem eğitilir; öğretmen belleğe hiç alınmaz.

    # 1) öğretmen geçişi (GPU'da hızlı): top-k dağılımları diske
    python -m heimdall.damit ogretmen --model HuggingFaceTB/SmolLM2-360M --data veri/train.bin --out kd/ --tokens 50e6
    # 2) öğrenci eğitimi: damıtma kaybı + normal kayıp
    python -m heimdall.damit ogrenci --kd kd/ --test veri/test.bin --preset kucuk --out kosular/damit
    # karşılaştırma için aynı veriyle yalnız normal kayıp:  --alpha 0
"""

import argparse
import contextlib
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import PRESETS, apply_arch
from .model import HeimdallLM
from .optim import build_optimizers, set_lr, wsd


def _device(name):
    return torch.device(name or ("cuda" if torch.cuda.is_available() else "cpu"))


@torch.no_grad()
def teacher_pass(args) -> None:
    from .donustur import ConvLM, fetch_model

    dev = _device(args.device)
    gpu = dev.type == "cuda"
    model = ConvLM.from_pretrained(fetch_model(args.model)).eval()
    if gpu:
        model = model.to(torch.bfloat16)
    model.to(dev)
    data = np.memmap(args.data, dtype=np.uint16, mode="r")
    t, k = args.seq_len, args.k
    n_seq = min(int(args.tokens) // t, (len(data) - 1) // t)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ids_f = np.lib.format.open_memmap(out / "ids.npy", "w+", np.uint16, (n_seq, t + 1))
    top_i = np.lib.format.open_memmap(out / "top_i.npy", "w+", np.uint16 if model.tc.vocab < 65536 else np.int32,
                                      (n_seq, t, k))
    top_v = np.lib.format.open_memmap(out / "top_v.npy", "w+", np.float16, (n_seq, t, k))
    t0 = time.time()
    for s in range(0, n_seq, args.batch):
        e = min(s + args.batch, n_seq)
        rows = np.stack([data[i * t:i * t + t + 1] for i in range(s, e)]).astype(np.int64)
        x = torch.from_numpy(rows[:, :-1]).to(dev)
        logp = F.log_softmax(model(x).float(), dim=-1)
        v, i = logp.topk(k, dim=-1)
        ids_f[s:e], top_i[s:e], top_v[s:e] = rows, i.cpu().numpy(), v.cpu().half().numpy()
        if (s // args.batch) % 20 == 0:
            done = e * t
            print(f"  öğretmen {done / 1e6:.2f}M / {n_seq * t / 1e6:.2f}M token | {done / (time.time() - t0):,.0f} tok/s",
                  flush=True)
    for a in (ids_f, top_i, top_v):
        a.flush()
    (out / "meta.json").write_text(json.dumps({"model": args.model, "vocab": model.tc.vocab, "k": k, "seq_len": t,
                                               "n_seq": n_seq}, indent=2))
    print(f"Yazıldı: {out} ({n_seq * t / 1e6:.2f}M token, top-{k})", flush=True)


def kd_loss(logits, top_i, top_v, targets, alpha: float, temp: float = 1.0):
    """alpha · CE(öğretmen top-k dağılımı, öğrenci) + (1 − alpha) · CE(gerçek token)."""
    logp = F.log_softmax(logits.float() / temp, dim=-1)
    hard = F.nll_loss(logp.view(-1, logp.size(-1)), targets.reshape(-1))
    if alpha == 0:
        return hard, hard
    p_t = torch.softmax(top_v.float() / temp, dim=-1)       # top-k içinde yeniden normalize
    soft = -(p_t * logp.gather(-1, top_i.long())).sum(-1).mean() * temp ** 2
    return alpha * soft + (1 - alpha) * hard, hard


@torch.no_grad()
def evaluate(model, test_path, t, n, dev, amp):
    data = np.memmap(test_path, dtype=np.uint16, mode="r")
    step = max(1, (len(data) - t - 1) // n)
    tot = 0.0
    model.eval()
    for j in range(n):
        row = torch.from_numpy(np.asarray(data[j * step:j * step + t + 1], dtype=np.int64))[None].to(dev)
        with amp:
            tot += F.cross_entropy(model(row[:, :-1]).float()[0], row[0, 1:]).item()
    model.train()
    return tot / n


def student_train(args) -> None:
    dev = _device(args.device)
    gpu = dev.type == "cuda"
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if gpu else contextlib.nullcontext()
    kd = Path(args.kd)
    meta = json.loads((kd / "meta.json").read_text())
    ids = np.load(kd / "ids.npy", mmap_mode="r")
    top_i = np.load(kd / "top_i.npy", mmap_mode="r")
    top_v = np.load(kd / "top_v.npy", mmap_mode="r")
    preset = PRESETS[args.preset]
    cfg = replace(apply_arch(preset.model, args.arch), vocab_size=meta["vocab"], kernel="auto" if gpu else "torch",
                  **({"d_model": args.d_model} if args.d_model else {}),
                  **({"n_layers": args.n_layers} if args.n_layers else {}))
    torch.manual_seed(args.seed)
    model = HeimdallLM(cfg).to(dev)
    opts = build_optimizers(model, args.lr_muon or preset.lr_muon, args.lr_adam or preset.lr_adam)
    n_seq, t = ids.shape[0], meta["seq_len"]
    total = int(args.epochs * n_seq // args.batch)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.jsonl", "w")
    print(f"Öğrenci {model.num_params() / 1e6:.1f}M param ({args.arch}) | öğretmen {meta['model']} top-{meta['k']} | "
          f"{n_seq * t / 1e6:.2f}M token × {args.epochs} dönem = {total} adım | alpha {args.alpha}", flush=True)
    t0 = time.time()
    for step in range(1, total + 1):
        set_lr(opts, wsd(step / total, 0.02, 0.7))
        b = np.sort(rng.choice(n_seq, args.batch, replace=False))
        x = torch.from_numpy(ids[b].astype(np.int64)).to(dev)
        ti = torch.from_numpy(top_i[b].astype(np.int64)).to(dev)
        tv = torch.from_numpy(top_v[b].astype(np.float32)).to(dev)
        with amp:
            logits = model(x[:, :-1])
        loss, hard = kd_loss(logits, ti, tv, x[:, 1:], args.alpha, args.temp)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for o in opts:
            o.step()
            o.zero_grad(set_to_none=True)
        if step % args.log_every == 0 or step == total:
            rec = {"step": step, "tokens": step * args.batch * t, "loss": loss.item(), "ce": hard.item(),
                   "tok_s": step * args.batch * t / (time.time() - t0)}
            if args.test and (step % (args.log_every * 5) == 0 or step == total):
                rec["val"] = evaluate(model, args.test, t, args.eval_n, dev, amp)
            print(" | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
    torch.save({"config": cfg.to_dict(), "model": model.state_dict(), "step": total, "tokens": total * args.batch * t,
                "optim": [], "args": vars(args)}, out / "last.pt")
    print(f"Kaydedildi: {out / 'last.pt'}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Bilgi damıtma")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ogretmen", help="öğretmenin top-k dağılımlarını diske yaz")
    a.add_argument("--model", required=True)
    a.add_argument("--data", required=True, help="öğretmen tokenizer'ıyla uint16 .bin")
    a.add_argument("--out", required=True)
    a.add_argument("--tokens", type=float, default=10e6)
    a.add_argument("--seq-len", type=int, default=1024)
    a.add_argument("--k", type=int, default=32)
    a.add_argument("--batch", type=int, default=4)
    a.add_argument("--device")
    b = sub.add_parser("ogrenci", help="öğrenciyi damıtma dosyasından eğit")
    b.add_argument("--kd", required=True)
    b.add_argument("--test")
    b.add_argument("--out", required=True)
    b.add_argument("--preset", default="mini", choices=list(PRESETS))
    b.add_argument("--arch", default="hibrit")
    b.add_argument("--d-model", type=int)
    b.add_argument("--n-layers", type=int)
    b.add_argument("--alpha", type=float, default=0.9, help="damıtma kaybı ağırlığı (0 = normal eğitim)")
    b.add_argument("--temp", type=float, default=1.0)
    b.add_argument("--epochs", type=float, default=1.0)
    b.add_argument("--batch", type=int, default=8)
    b.add_argument("--lr-muon", type=float)
    b.add_argument("--lr-adam", type=float)
    b.add_argument("--log-every", type=int, default=20)
    b.add_argument("--eval-n", type=int, default=16)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--device")
    args = p.parse_args()
    teacher_pass(args) if args.cmd == "ogretmen" else student_train(args)


if __name__ == "__main__":
    main()
