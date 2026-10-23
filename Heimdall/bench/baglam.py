"""
Eğitilmiş Heimdall modelinin BAĞLAM davranışı: eğitim uzunluğunun (2048) ötesinde ne oluyor?

    # 1) Konuma göre kayıp: bağlam uzadıkça model ilerideki token'ları daha iyi mi, kötü mü tahmin ediyor?
    python bench/baglam.py nll --ckpt runs/hafta/model/last.pt --veri veri/kod_heldout.bin --uzunluk 8192 --n 8

    # 2) Parola arama (passkey): uzun bir metnin ortasına gömülü 5 haneli sayıyı geri getirebiliyor mu?
    python bench/baglam.py sifre --ckpt runs/hafta/model/last.pt --uzunluklar 512 1024 2048 4096 8192 --deneme 5

nll: pencereler .bin'den rastgele alınır (belgeler <eos> ile birleşik olabilir; mutlak değerden çok EĞRİNİN ŞEKLİ önemlidir).
     Kayıp 2048'den sonra YÜKSELİYORSA model eğitim uzunluğunun ötesine genellemiyor demektir.
sifre: base model için zor bir test; ≤2048'de bile düşük çıkabilir. Çıktı, her uzunlukta doğruluk ve ölçülen cache boyutunu verir
       (cache boyutu: yalnız global dikkat katmanları uzunlukla büyür; DeltaNet katmanları sabittir).
Hiçbir sonuç yayınlanmadan önce yorumlanmalıdır: tek checkpoint, küçük örnek, tek tohum.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heimdall.io import DEFAULT_TOKENIZER, load_model  # noqa: E402

FILLER = ("The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
          "The river runs to the sea. A small bird sat on the old wall. ")


def device_and_amp():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")
    return dev, amp


@torch.no_grad()
def nll_by_position(model, data, length: int, n: int, seed: int, dev, amp) -> np.ndarray:
    rng = np.random.default_rng(seed)
    total = np.zeros(length)
    for _ in range(n):
        s = int(rng.integers(0, len(data) - length - 1))
        w = torch.from_numpy(data[s:s + length + 1].astype(np.int64)).to(dev)
        with amp:
            logits = model(w[None, :-1])
        total += F.cross_entropy(logits[0].float(), w[1:], reduction="none").cpu().numpy()
        del logits
    return total / n


def buckets_for(length: int) -> list:
    edges = [0, 256, 512, 1024, 2048]
    e = 2048
    while e < length:
        e = min(e * 2, length)
        edges.append(e)
    return list(zip(edges[:-1], edges[1:]))


def cmd_nll(a) -> None:
    dev, amp = device_and_amp()
    model = load_model(a.ckpt, dev)
    data = np.memmap(a.veri, dtype=np.uint16, mode="r")
    nll = nll_by_position(model, data, a.uzunluk, a.n, a.seed, dev, amp)
    print(f"{model.num_params() / 1e6:.0f}M param | {a.n} pencere × {a.uzunluk} token | {a.veri}")
    print(f"{'konum aralığı':>16s}  {'kayıp (nats)':>12s}   ppl")
    rows = {}
    for lo, hi in buckets_for(a.uzunluk):
        v = float(nll[lo:hi].mean())
        rows[(lo, hi)] = v
        print(f"{f'{lo}-{hi}':>16s}  {v:12.4f}   {np.exp(v):7.2f}" + ("   ← eğitim uzunluğunun ötesi" if lo >= 2048 else ""))
    inside = rows.get((1024, 2048))
    beyond = [v for (lo, hi), v in rows.items() if lo >= 2048]
    if inside is not None and beyond:
        worst = max(beyond)
        print(f"\n2048 ötesi en kötü aralık / 1024-2048 aralığı: {worst / inside:.3f}×  "
              f"({'GENELLEMİYOR: kayıp yükseliyor' if worst > inside * 1.05 else 'kayıp yaklaşık korunuyor'})")


def build_prompt(tok: Tokenizer, length: int, key: str, depth: float) -> list:
    """~length token: dolgu metni, derinlik oranında gömülü parola cümlesi, sonda soru."""
    needle = tok.encode(f" The pass key is {key}. Remember it. {key} is the pass key. ").ids
    question = tok.encode(" What is the pass key? The pass key is").ids
    filler = tok.encode(FILLER).ids
    body = max(length - len(needle) - len(question), 8)
    ids = (filler * (body // len(filler) + 1))[:body]
    cut = int(depth * len(ids))
    return ids[:cut] + needle + ids[cut:] + question


@torch.no_grad()
def cmd_sifre(a) -> None:
    dev, amp = device_and_amp()
    model = load_model(a.ckpt, dev)
    tok = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    rng = np.random.default_rng(a.seed)
    depths = [0.1, 0.5, 0.9]
    print(f"{model.num_params() / 1e6:.0f}M param | {a.deneme} deneme × {len(depths)} derinlik\n")
    print(f"{'uzunluk':>8s}  " + "  ".join(f"derinlik {d:.0%}" for d in depths) + "   toplam   cache")
    for length in a.uzunluklar:
        ok = {d: 0 for d in depths}
        cache_mb = 0.0
        for _ in range(a.deneme):
            for d in depths:
                key = str(int(rng.integers(10000, 99999)))
                ids = torch.tensor([build_prompt(tok, length, key, d)], device=dev)
                cache = model.new_cache(1)
                with amp:
                    logits = model(ids, cache=cache, last_only=True)[0, -1]
                    out = []
                    for _ in range(8):
                        nxt = int(logits.argmax())
                        out.append(nxt)
                        logits = model(torch.tensor([[nxt]], device=dev), cache=cache)[0, -1]
                cache_mb = model.cache_bytes(cache) / 2 ** 20
                ok[d] += key in tok.decode(out).replace(" ", "")
        tot = sum(ok.values())
        print(f"{length:8d}  " + "  ".join(f"{ok[d]:>3d}/{a.deneme:<9d}" for d in depths)
              + f"   {100 * tot / (a.deneme * len(depths)):5.0f}%  {cache_mb:7.1f} MB", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("nll")
    n.add_argument("--ckpt", required=True)
    n.add_argument("--veri", required=True, help="uint16 .bin (eğitimde görülmemiş olması tercih edilir)")
    n.add_argument("--uzunluk", type=int, default=8192)
    n.add_argument("--n", type=int, default=8)
    n.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("sifre")
    s.add_argument("--ckpt", required=True)
    s.add_argument("--uzunluklar", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192])
    s.add_argument("--deneme", type=int, default=5)
    s.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    t0 = time.time()
    {"nll": cmd_nll, "sifre": cmd_sifre}[a.cmd](a)
    print(f"\n({time.time() - t0:.0f} sn)")


if __name__ == "__main__":
    main()
