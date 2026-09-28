"""Heimdall: Gated DeltaNet + seyrek global dikkat hibrit dil modeli (CUDA / ROCm)."""

from .config import ARCHS, PRESETS, HeimdallConfig, apply_arch
from .model import HeimdallLM

__all__ = ["HeimdallConfig", "HeimdallLM", "PRESETS", "ARCHS", "apply_arch"]
__version__ = "1.0.0"
