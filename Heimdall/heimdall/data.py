"""
Veri: uint16 token dosyaları (.bin), ağırlıklı karışım, dosya sonundan validation ayrımı, arka plan yükleyici.
Dosya belirtimi: "yol.bin" ya da "yol.bin:ağırlık" (Windows sürücü harfi desteklenir).
"""

import queue
import threading
from pathlib import Path

import numpy as np
import torch


def parse_spec(spec: str):
    head, sep, tail = spec.rpartition(":")
    if sep and head and not tail.startswith(("\\", "/")):
        try:
            return head, float(tail)
        except ValueError:
            pass
    return spec, None


class Mixture:
    def __init__(self, specs, val_frac: float = 0.005, seed: int = 0) -> None:
        self.sources, weights = [], []
        for spec in specs:
            path, w = parse_spec(spec)
            data = np.memmap(path, dtype=np.uint16, mode="r")
            split = int(len(data) * (1 - val_frac))
            self.sources.append((path, data, split))
            weights.append(w)
        sizes = np.array([s[2] for s in self.sources], dtype=np.float64)
        w = np.array([x if x is not None else (1.0 if any(v is not None for v in weights) else s)
                      for x, s in zip(weights, sizes)])
        self.weights = w / w.sum()
        self.rng = np.random.default_rng(seed)

    def describe(self) -> str:
        return ", ".join(f"{Path(p).name} ({len(d) / 1e6:.1f}M tok, %{100 * w:.0f})"
                         for (p, d, _), w in zip(self.sources, self.weights))

    def batch(self, b: int, t: int) -> torch.Tensor:
        which = self.rng.choice(len(self.sources), size=b, p=self.weights)
        rows = []
        for i in which:
            _, data, split = self.sources[i]
            s = int(self.rng.integers(0, split - t - 1))
            rows.append(np.asarray(data[s:s + t + 1], dtype=np.int64))
        return torch.from_numpy(np.stack(rows))

    def val_batches(self, t: int, n: int):
        """Her kaynaktan n adet sabit validation penceresi: [(ad, tensor[n, t+1])]."""
        out = []
        for path, data, split in self.sources:
            avail = (len(data) - split - 1) // (t + 1)
            k = min(n, avail)
            if k > 0:
                arr = np.stack([np.asarray(data[split + j * (t + 1): split + (j + 1) * (t + 1)], dtype=np.int64)
                                for j in range(k)])
                out.append((Path(path).stem, torch.from_numpy(arr)))
        return out


class Loader:
    """Arka plan iş parçacığında batch hazırlar; pinned bellek + asenkron cihaza kopya."""

    def __init__(self, mix: Mixture, batch: int, seq_len: int, device: torch.device, depth: int = 4) -> None:
        self.mix, self.batch, self.seq_len, self.device = mix, batch, seq_len, device
        self.q = queue.Queue(maxsize=depth)
        self.stop = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        pin = self.device.type == "cuda"
        while not self.stop:
            xy = self.mix.batch(self.batch, self.seq_len)
            if pin:
                xy = xy.pin_memory()
            while not self.stop:
                try:
                    self.q.put(xy, timeout=0.5)
                    break
                except queue.Full:
                    continue

    def next(self):
        xy = self.q.get().to(self.device, non_blocking=True)
        return xy[:, :-1], xy[:, 1:]
