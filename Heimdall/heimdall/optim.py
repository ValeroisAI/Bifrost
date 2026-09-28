"""
Muon (gizli 2D matrisler) + AdamW (embedding, norm, kapı, skalerler) ve WSD öğrenme hızı çizelgesi.
Birleşik matrisler (qkv, w12) `muon_split` parçalarına bölünüp ayrı ayrı ortogonalleştirilir.
"""

import torch
import torch.nn as nn


def zeropower_ns5(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16() if g.is_cuda else g.float()
    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        s = x @ x.mT
        x = a * x + (b * s + c * (s @ s)) @ x
    if transposed:
        x = x.mT
    return x.to(g.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, weight_decay: float = 0.0) -> None:
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            mom = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "m" not in st:
                    st["m"] = torch.zeros_like(p.grad)
                buf = st["m"]
                buf.lerp_(p.grad, 1 - mom)
                u = p.grad.lerp(buf, mom)  # Nesterov
                parts = u.split(list(getattr(p, "muon_split", (p.size(0),))), dim=0)
                u = torch.cat([zeropower_ns5(s) * max(1.0, s.size(0) / s.size(1)) ** 0.5 for s in parts], dim=0)
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(u, alpha=-group["lr"])


def build_optimizers(model: nn.Module, lr_muon: float, lr_adam: float, use_muon: bool = True) -> list:
    embed_ids = {id(model.embed.weight), id(model.lm_head.weight)}
    muon, embed, rest, seen = [], [], [], set()
    for p in model.parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if id(p) in embed_ids:
            embed.append(p)
        elif p.ndim == 2 and min(p.shape) >= 64:
            muon.append(p)
        else:
            rest.append(p)
    groups = [{"params": embed, "weight_decay": 0.0}, {"params": rest, "weight_decay": 0.0}]
    if not use_muon:
        groups.append({"params": muon, "weight_decay": 0.1})
    kw = dict(lr=lr_adam, betas=(0.9, 0.95), eps=1e-8)
    try:
        adam = torch.optim.AdamW(groups, fused=torch.cuda.is_available(), **kw)
    except (RuntimeError, TypeError):
        adam = torch.optim.AdamW(groups, **kw)
    return [Muon(muon, lr=lr_muon), adam] if use_muon else [adam]


def wsd(progress: float, warmup: float = 0.01, decay_start: float = 0.75) -> float:
    """Isınma → sabit → doğrusal düşüş (progress ∈ [0, 1])."""
    if progress < warmup:
        return max(progress / warmup, 1e-3)
    if progress < decay_start:
        return 1.0
    return max(0.0, (1.0 - progress) / (1.0 - decay_start))


def set_lr(optimizers, factor: float) -> None:
    for opt in optimizers:
        for group in opt.param_groups:
            group.setdefault("base_lr", group["lr"])
            group["lr"] = group["base_lr"] * factor
