"""
hazirla.py — Hugging Face veri kümesinden (parquet) metin indirip tokenizer'ımızla uint16 .bin dosyasına çevirir.

    python -m heimdall.hazirla --repo HuggingFaceFW/fineweb-edu --dosya sample/10BT/000_00000.parquet \
        --kolon text --out veri/fineweb.bin --max-token 500e6
    python -m heimdall.hazirla --repo codeparrot/github-code-clean --dosya data/train-00000-of-00880.parquet \
        --kolon code --out veri/kod.bin --max-token 200e6

Belgeler <eos> ile ayrılır. Dosya yarım kalırsa kaldığı belgeden devam etmez; baştan yazılır (kısa iştir).
"""

import argparse
import time
from pathlib import Path

import numpy as np

from .io import DEFAULT_TOKENIZER


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--dosya", nargs="+", required=True, help="repodaki parquet yolları")
    p.add_argument("--kolon", default="text")
    p.add_argument("--out", required=True)
    p.add_argument("--max-token", type=float, default=1e9)
    p.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    p.add_argument("--min-karakter", type=int, default=200, help="bundan kısa belgeleri at")
    p.add_argument("--batch", type=int, default=2000, help="aynı anda tokenleştirilen belge sayısı")
    args = p.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(args.tokenizer)
    eos = tok.token_to_id("<eos>")
    assert eos is not None and tok.get_vocab_size() < 65536
    limit = int(args.max_token)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")       # yarım kalan dosya "hazır" sayılmasın: bitince yeniden adlandırılır
    total, docs, t0 = 0, 0, time.time()
    with open(tmp, "wb") as f:
        for path in args.dosya:
            print(f"indiriliyor: {args.repo}/{path}", flush=True)
            local = hf_hub_download(args.repo, path, repo_type="dataset")
            pf = pq.ParquetFile(local)
            for rb in pf.iter_batches(batch_size=args.batch, columns=[args.kolon]):
                texts = [t for t in rb.column(0).to_pylist() if t and len(t) >= args.min_karakter]
                if not texts:
                    continue
                ids = []
                for e in tok.encode_batch(texts):
                    ids.extend(e.ids)
                    ids.append(eos)
                arr = np.asarray(ids, dtype=np.uint16)
                if total + len(arr) > limit:
                    arr = arr[:limit - total]
                arr.tofile(f)
                total += len(arr)
                docs += len(texts)
                if (docs // args.batch) % 10 == 0:
                    print(f"  {total / 1e6:,.0f}M token | {docs:,} belge | {total / (time.time() - t0) / 1e6:.2f}M tok/s", flush=True)
                if total >= limit:
                    break
            if total >= limit:
                break
    tmp.replace(out)
    data = np.memmap(out, dtype=np.uint16, mode="r")
    print(f"Yazıldı: {out} — {len(data) / 1e6:,.1f}M token, {docs:,} belge, {time.time() - t0:.0f} sn")
    mid = len(data) // 2
    print("Örnek (ortadan 300 token):", repr(tok.decode(data[mid:mid + 300].tolist()))[:400])


if __name__ == "__main__":
    main()
