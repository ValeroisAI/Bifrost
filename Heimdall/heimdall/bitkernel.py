"""
bitkernel.py — 1-bit / 1.58-bit için Triton çekirdekleri (CUDA ve ROCm).

  1) int8 GEMM  (eğitim, prefill):   y = (Xq · Wᵀ) ⊙ sx       Xq: token başına int8'e nicemlenmiş aktivasyon,
     W ∈ {−1,0,+1} int8. int8 × int8 → int32 birikim. Geri geçiş düz geçiş (STE): ağırlık gradyanı yoğun kalır.
  2) Paketli GEMM (üretim, bellek bağımlı): ağırlık 1 bit (8/bayt) ya da 2 bit (4/bayt) paketlenmiş durur,
     çekirdek içinde açılır; bf16 aktivasyonla çarpılır. bf16 depolamaya göre 16× (1-bit) / 8× (2-bit) az bellek trafiği.

Doğruluk testleri Triton yorumlayıcısında (CPU) yapılır:  tests/test_bitkernel.py
Hız ölçümü GPU'da yapılır:  python bench/bitkernel.py   (bu dosya sandbox'ta GPU'da çalıştırılmadı)
Çekirdek derlenmezse / desteklenmezse (ör. int8 dot bu GPU'da yok) modül yoğun yola düşer ve bir kez uyarır.
"""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:  # triton kurulu değil
    HAS_TRITON = False

_WARNED = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        warnings.warn(msg)


if HAS_TRITON:

    @triton.jit
    def _int8_mm_kernel(x_ptr, w_ptr, sx_ptr, sw_ptr, y_ptr, M, N, K,
                        stride_xm, stride_xk, stride_wn, stride_wk, stride_ym, stride_yn,
                        HAS_SW: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        offs_m = tl.program_id(0) * BM + tl.arange(0, BM)
        offs_n = tl.program_id(1) * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.int32)
        for k in range(0, K, BK):
            kk = k + offs_k
            a = tl.load(x_ptr + offs_m[:, None] * stride_xm + kk[None, :] * stride_xk,
                        mask=(offs_m[:, None] < M) & (kk[None, :] < K), other=0)
            b = tl.load(w_ptr + offs_n[None, :] * stride_wn + kk[:, None] * stride_wk,
                        mask=(offs_n[None, :] < N) & (kk[:, None] < K), other=0)
            acc = tl.dot(a, b, acc)
        out = acc.to(tl.float32) * tl.load(sx_ptr + offs_m, mask=offs_m < M, other=0.0)[:, None]
        if HAS_SW:
            out = out * tl.load(sw_ptr + offs_n, mask=offs_n < N, other=0.0)[None, :]
        tl.store(y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn, out,
                 mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    @triton.jit
    def _packed_mm_kernel(x_ptr, p_ptr, s_ptr, y_ptr, M, N, K,
                          stride_xm, stride_xk, stride_pn, stride_ym, stride_yn,
                          BITS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        PER: tl.constexpr = 8 // BITS
        offs_m = tl.program_id(0) * BM + tl.arange(0, BM)
        offs_n = tl.program_id(1) * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, K, BK):
            kk = k + offs_k
            a = tl.load(x_ptr + offs_m[:, None] * stride_xm + kk[None, :] * stride_xk,
                        mask=(offs_m[:, None] < M) & (kk[None, :] < K), other=0.0)
            kmask = (offs_n[None, :] < N) & (kk[:, None] < K)
            byte = tl.load(p_ptr + offs_n[None, :] * stride_pn + kk[:, None] // PER, mask=kmask, other=0).to(tl.int32)
            code = (byte >> ((kk[:, None] % PER) * BITS)) & ((1 << BITS) - 1)
            if BITS == 1:
                w = (code * 2 - 1).to(tl.float32)
            else:
                w = tl.where(code == 1, 1.0, tl.where(code == 2, -1.0, 0.0))
            w = tl.where(kmask, w, 0.0).to(a.dtype)
            acc = tl.dot(a, w, acc)
        out = acc * tl.load(s_ptr + offs_n, mask=offs_n < N, other=0.0)[None, :]
        tl.store(y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn, out,
                 mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _tiles(m: int, decode: bool):
    if decode or m <= 32:
        return 16, 64, 128, 4
    if m < 256:
        return 64, 64, 64, 4
    return 128, 128, 64, 8


def int8_matmul(xq: torch.Tensor, w8: torch.Tensor, sx: torch.Tensor, sw: torch.Tensor = None,
                out_dtype=torch.float32) -> torch.Tensor:
    """xq [M,K] int8, w8 [N,K] int8, sx [M] fp32 (satır ölçeği), sw [N] fp32 (isteğe bağlı) → [M,N]."""
    m, k = xq.shape
    n = w8.size(0)
    y = torch.empty(m, n, device=xq.device, dtype=out_dtype)
    bm, bn, bk, warps = _tiles(m, False)
    grid = (triton.cdiv(m, bm), triton.cdiv(n, bn))
    _int8_mm_kernel[grid](xq, w8, sx, sw if sw is not None else sx, y, m, n, k, xq.stride(0), xq.stride(1),
                          w8.stride(0), w8.stride(1), y.stride(0), y.stride(1),
                          HAS_SW=sw is not None, BM=bm, BN=bn, BK=bk, num_warps=warps)
    return y


def pack_weights(w: torch.Tensor, bits: int) -> torch.Tensor:
    """w [N,K] ∈ {−1,+1} (bits=1) ya da {−1,0,+1} (bits=2) → uint8 [N, ceil(K/per)]; K boyunca paketlenir."""
    per = 8 // bits
    n, k = w.shape
    wf = F.pad(w.float(), (0, (-k) % per))
    if bits == 1:
        code = (wf > 0).to(torch.int32)
    else:
        code = torch.where(wf > 0, 1, torch.where(wf < 0, 2, 0)).to(torch.int32)
    shifts = torch.arange(per, device=w.device, dtype=torch.int32) * bits
    return (code.view(n, -1, per) << shifts).sum(-1).to(torch.uint8)


def unpack_weights(packed: torch.Tensor, bits: int, k: int, dtype=torch.float32) -> torch.Tensor:
    per = 8 // bits
    shifts = torch.arange(per, device=packed.device, dtype=torch.int32) * bits
    code = (packed.to(torch.int32)[..., None] >> shifts) & ((1 << bits) - 1)
    code = code.reshape(packed.size(0), -1)[:, :k]
    w = (code * 2 - 1) if bits == 1 else torch.where(code == 1, 1, torch.where(code == 2, -1, 0))
    return w.to(dtype)


def packed_matmul(x: torch.Tensor, packed: torch.Tensor, scale: torch.Tensor, bits: int, k: int) -> torch.Tensor:
    """x [M,K] (bf16/fp16/fp32), packed [N,·] uint8, scale [N] fp32 → [M,N] (x ile aynı dtype)."""
    m = x.size(0)
    n = packed.size(0)
    y = torch.empty(m, n, device=x.device, dtype=x.dtype)
    bm, bn, bk, warps = _tiles(m, m <= 32)
    grid = (triton.cdiv(m, bm), triton.cdiv(n, bn))
    _packed_mm_kernel[grid](x, packed, scale, y, m, n, k, x.stride(0), x.stride(1), packed.stride(0),
                            y.stride(0), y.stride(1), BITS=bits, BM=bm, BN=bn, BK=bk, num_warps=warps)
    return y


# --------------------------------------------------------------------------- eğitim: int8 yolu
class _Int8BitFn(torch.autograd.Function):
    """İleri: aktivasyon token başına int8, ağırlık int8 (±1/0), int32 birikim. Geri: düz geçiş (STE)."""

    @staticmethod
    def forward(ctx, x2d, weight):
        sx = (x2d.detach().abs().amax(dim=-1).float() / 127.0).clamp(min=1e-8)
        xq = (x2d.detach().float() / sx[:, None]).round().clamp(-127, 127).to(torch.int8)
        y = int8_matmul(xq.contiguous(), weight.detach().to(torch.int8).contiguous(), sx, None, x2d.dtype)
        ctx.save_for_backward(x2d, weight)
        return y

    @staticmethod
    def backward(ctx, gy):
        x2d, w = ctx.saved_tensors
        return gy @ w.to(gy.dtype), (gy.t() @ x2d.to(gy.dtype)).to(w.dtype)


@torch.compiler.disable
def bit_int8_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """x [..., K] · weightᵀ. Çekirdek çalışmazsa yoğun yola düşer."""
    shape = x.shape
    try:
        y = _Int8BitFn.apply(x.reshape(-1, shape[-1]), weight)
    except Exception as e:  # derleme/destek hatası: bir kez uyar, yoğun yol
        _warn_once("int8", f"int8 çekirdek çalışmadı ({type(e).__name__}: {str(e)[:120]}); yoğun yola dönülüyor")
        return F.linear(x, weight.to(x.dtype))
    return y.reshape(*shape[:-1], -1)


# --------------------------------------------------------------------------- üretim: paketli yol
class PackedBitLinear(nn.Module):
    """Yalnız çıkarım. Ağırlık 1/2 bit paketli; bellek bf16'nın 1/16 (ikili) ya da 1/8 (üçlü) kadar."""

    def __init__(self, weight: torch.Tensor, scale: torch.Tensor, bits: int) -> None:
        super().__init__()
        self.bits, self.in_features, self.out_features = bits, weight.size(1), weight.size(0)
        self.register_buffer("packed", pack_weights(weight, bits))
        self.register_buffer("scale", scale.float().contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2d = x.reshape(-1, shape[-1]).contiguous()
        if HAS_TRITON and x.is_cuda:
            try:
                y = packed_matmul(x2d, self.packed, self.scale, self.bits, self.in_features)
                return y.reshape(*shape[:-1], -1)
            except Exception as e:
                _warn_once("packed", f"paketli çekirdek çalışmadı ({type(e).__name__}: {str(e)[:120]}); torch yolu")
        w = unpack_weights(self.packed, self.bits, self.in_features, x.dtype)
        return (F.linear(x2d, w) * self.scale.to(x.dtype)).reshape(*shape[:-1], -1)


def pack_bit_model(model: nn.Module) -> int:
    """BitLinear katmanlarını yerinde PackedBitLinear'a çevirir (çıkarım için). Kazanılan bayt sayısını döndürür."""
    from .bit import BitLinear

    saved = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, BitLinear):
                scale = (child.base * child.scale).detach()
                packed = PackedBitLinear(child.weight.detach(), scale, child.bits)
                saved += child.weight.numel() * child.weight.element_size() - packed.packed.numel()
                setattr(parent, name, packed)
    return saved
