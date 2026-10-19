"""
Koşuları karşılaştırır (log.jsonl): son val loss, eşit tokende val loss, hız.

    python -m heimdall.compare runs/temel runs/temel_tf
"""

import json
import sys
from pathlib import Path


def load(run: str):
    rows = [json.loads(line) for line in open(Path(run) / "log.jsonl")]
    vals = [(r["tokens"], r["val"]) for r in rows if "val" in r]
    speed = [r["tok_s"] for r in rows if "tok_s" in r]
    return vals, (sorted(speed)[len(speed) // 2] if speed else 0.0)


def main() -> None:
    runs = {r: load(r) for r in sys.argv[1:]}
    common = min(v[-1][0] for v, _ in runs.values() if v)
    print(f"{'koşu':30s} {'son val':>9s} {'val @ ' + format(common / 1e6, '.0f') + 'M tok':>16s} {'tok/s (medyan)':>15s}")
    for name, (vals, speed) in runs.items():
        last = sum(vals[-1][1].values()) / len(vals[-1][1])
        at = [v for t, v in vals if t <= common][-1]
        print(f"{name:30s} {last:9.4f} {sum(at.values()) / len(at):16.4f} {speed:15,.0f}")


if __name__ == "__main__":
    main()
