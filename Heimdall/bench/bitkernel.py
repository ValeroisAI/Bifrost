"""
Bit çekirdeklerinin GPU'da doğruluğu ve hızı (CUDA / ROCm):  python bench/bitkernel.py

Her şekil için: bf16 F.linear (taban) | int8 Triton GEMM | paketli 1-bit | paketli 2-bit.
Çekirdek derlenmezse hata mesajı basılır ve o satır atlanır; bu, o GPU'da çekirdeğin çalışmadığını gösterir.
"""
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heimdall.bitkernel import int8_matmul, pack_weights, packed_matmul  # noqa: E402

assert torch.cuda.is_available(), "GPU gerekli"
dev = "cuda"


def timeit(fn, iters=30, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / iters


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


print(f"{torch.cuda.get_device_name(0)} | torch {torch.__version__} | hip {torch.version.hip}")
print(f"{'şekil (M,K,N)':>22s} | {'yol':14s} | {'ms':>8s} | {'bf16 ?×':>8s} | {'göreli hata':>11s} | {'GB/s ağırlık':>12s}")
shapes = [("üretim", 1, 4096, 4096), ("üretim", 8, 4096, 4096), ("üretim", 16, 2048, 8192),
          ("prefill", 2048, 2048, 2048), ("eğitim", 8192, 2048, 2048), ("eğitim", 8192, 768, 2048)]
for tag, m, k, n in shapes:
    torch.manual_seed(0)
    x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    w1 = torch.where(torch.rand(n, k, device=dev) > 0.5, 1.0, -1.0)
    w2 = torch.randint(-1, 2, (n, k), device=dev).float()
    s = (torch.rand(n, device=dev) + 0.5)
    ref1, ref2 = (x.float() @ w1.t()) * s, (x.float() @ w2.t()) * s
    base = timeit(lambda: F.linear(x, w2.to(torch.bfloat16)))
    rows = [("bf16 F.linear", base, 0.0, 2 * n * k)]
    try:  # int8: aktivasyon nicemleme dahil
        w8 = w2.to(torch.int8)
        def run8():
            sx = (x.abs().amax(-1).float() / 127).clamp(min=1e-8)
            return int8_matmul((x.float() / sx[:, None]).round().clamp(-127, 127).to(torch.int8), w8, sx, s, torch.bfloat16)
        rows.append(("int8 Triton", timeit(run8), rel(run8(), ref2), n * k))
    except Exception as e:
        print(f"{tag} {(m, k, n)}: int8 çekirdek HATA {type(e).__name__}: {str(e)[:150]}")
    for bits, w, ref in ((1, w1, ref1), (2, w2, ref2)):
        try:
            pk = pack_weights(w, bits)
            fn = lambda: packed_matmul(x, pk, s, bits, k)
            rows.append((f"paketli {bits}-bit", timeit(fn), rel(fn(), ref), pk.numel()))
        except Exception as e:
            print(f"{tag} {(m, k, n)}: paketli {bits}-bit çekirdek HATA {type(e).__name__}: {str(e)[:150]}")
    for name, t, err, wbytes in rows:
        print(f"{f'{tag} {(m, k, n)}':>22s} | {name:14s} | {t * 1e3:8.3f} | {base / t:7.2f}× | {err:11.4f} | {wbytes / t / 1e9:12.0f}")
