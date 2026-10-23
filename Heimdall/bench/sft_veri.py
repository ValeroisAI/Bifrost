#!/usr/bin/env python3
"""
sft_veri.py — Üç Hugging Face SFT veri kümesini indirir, tek tip OpenAI-sohbet JSONL'ine çevirir, süzer, tekilleştirir.

    python bench/sft_veri.py --kuru                       # İNDİRMEDEN: her kaynağın ilk satırlarını okuyup ne çıkacağını gösterir
    python bench/sft_veri.py --out veri/sft               # varsayılan: yalnız d3 ≈40K örnek (d2 ve d1 kapalı; nedeni yukarıda)
    python bench/sft_veri.py --d3-max 20000 --d2-max 500 --max-token-ornek 8000

Kaynaklar (her birinin mevcut yapısı elle incelendi):
  d3  saidutta69/fable-5-premium-v2             openai_chat/train.jsonl   temizlenmiş, çoğu KISA soru-cevap (~480 token), ~%6 araç izi
  d2  MoreThought/Fable-5.1-Max-Reasoning-...   5k/train.jsonl            uzun ajan izleri AMA araç çağrıları gizlenmiş (aşağıya bak)
  d1  Manusagents/...Distillation-Dataset       data/<kategori>/<kaynak>  73 kaynağın karması; birçok "iz" klasörü BOŞ satırlardır.
                                                 --d1 klasör... ile açılır; siber güvenlik klasörleri bilinçli olarak varsayılan dışıdır.

Çıktı (--out): train_metin.jsonl / eval_metin.jsonl (yalnız system/user/assistant), train_arac.jsonl / eval_arac.jsonl
(tool_calls ve tool mesajları içerenler; varsa "tools" alanıyla), her dosya için *.meta.jsonl (satır satır: kaynak, depo,
yükleyicinin beyan ettiği lisans, tahmini token) ve manifest.json (ne kadarı neden atıldı). Satır içine meta KONMAZ:
platform fazladan alanı reddedebilir.

Önce *_metin.jsonl ile küçük bir deneme yükleyin; araç biçimi için platformun güncel dokümanına bakın (araç izlerinde
D3'ün "tools" tanımı YOKTUR; D2'de vardır).

YASAL/ETİK UYARI (script de başlangıçta yazdırır): bu kümeler başka şirketlerin modellerinin çıktılarını (damıtma) içerir.
Kümelerin üzerindeki MIT/Apache etiketi yükleyicinin beyanıdır; o çıktıların hizmet şartlarını geçersiz kılmaz. Birçok
sağlayıcının şartı, çıktıların rakip model eğitiminde kullanılmasını yasaklar. Modeli yayınlayacaksanız ya da kredi
başvurusunda göstereceksiniz, kullanmadan önce bunu kendiniz değerlendirin.
"""

import argparse
import ast
import hashlib
import json
import random
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

D1_REPO = "Manusagents/GPT-5.5-Gemini-3.1-Pro-Grok-4-Claude-Fable-5-Mythos-5-Qwen-3.7-Max-and-more-Distillation-Dataset"
SOURCES = {
    "d3": dict(repo="saidutta69/fable-5-premium-v2", files=["openai_chat/train.jsonl"], license="mit", kind="chat"),
    "d2": dict(repo="MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x", files=["5k/train.jsonl"], license="apache-2.0", kind="chat"),
    "d1": dict(repo=D1_REPO, files=[], license="mit", kind="instr"),
}
ROLES = {"system", "user", "assistant", "tool"}
PLACEHOLDER = re.compile(r"<[a-zA-Z_ ]{2,40}>")
CAVEAT = "<local-command-caveat>"
NOTICE = ("UYARI: bu kümeler başka şirketlerin modellerinden damıtılmış çıktılar içerir; MIT/Apache etiketi yükleyicinin beyanıdır "
          "ve o çıktıların hizmet şartlarını geçersiz kılmaz. Yayınlayacağınız/başvuruda göstereceğiniz model için kendiniz değerlendirin.")


# --------------------------------------------------------------------------- normalizasyon
def _text(content) -> str | None:
    """OpenAI içerik parçaları listesini ya da metni düz metne çevirir."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and p.get("type", "text") == "text" and isinstance(p.get("text"), str):
                parts.append(p["text"])
            else:
                return None                      # görüntü vb.: desteklenmez
        return "".join(parts)
    return None


def normalize_messages(ms) -> list | None:
    """Mesajları OpenAI sohbet biçimine getirir; bozuksa None. arguments her zaman JSON METNİ olur."""
    if not isinstance(ms, list) or not ms:
        return None
    out = []
    for m in ms:
        if not isinstance(m, dict) or m.get("role") not in ROLES:
            return None
        text = _text(m.get("content"))
        if text is None:
            return None
        msg = {"role": m["role"], "content": text}
        if m["role"] == "assistant":
            if m.get("reasoning_content"):
                msg["reasoning_content"] = m["reasoning_content"]
            tcs = []
            for tc in m.get("tool_calls") or []:
                fn = (tc or {}).get("function") or {}
                args = fn.get("arguments", "{}")
                if not isinstance(args, str):
                    args = json.dumps(args, ensure_ascii=False)
                if tc.get("id") in (None, "") or not fn.get("name"):
                    return None
                tcs.append({"id": str(tc["id"]), "type": "function", "function": {"name": fn["name"], "arguments": args}})
            if tcs:
                msg["tool_calls"] = tcs
        elif m["role"] == "tool":
            if m.get("tool_call_id") in (None, ""):
                return None
            msg["tool_call_id"] = str(m["tool_call_id"])
        out.append(msg)
    return out


def parse_instruction(s: str) -> list:
    """D1: 'instruction' düz metin ya da str(dict) / JSON ({'messages': [...]}) olabilir."""
    t = s.strip()
    if t[:1] in "{[":
        obj = None
        for loader in (json.loads, ast.literal_eval):
            try:
                obj = loader(t)
                break
            except Exception:
                continue
        if isinstance(obj, dict) and isinstance(obj.get("messages"), list):
            return obj["messages"]
        if isinstance(obj, list) and obj and isinstance(obj[0], dict) and "role" in obj[0]:
            return obj
    return [{"role": "user", "content": s}]


def rows_to_messages(kind: str, row: dict):
    """Ham satır → (mesajlar, tools). Başarısızsa (None, neden)."""
    if kind == "instr":
        instr, resp = row.get("instruction") or "", row.get("response") or ""
        if not isinstance(instr, str) or not isinstance(resp, str) or not instr.strip() or not resp.strip():
            return None, "bos_satir"
        ms = parse_instruction(instr) + [{"role": "assistant", "content": resp}]
        tools = None
    else:
        ms = row.get("messages")
        if isinstance(ms, str):
            try:
                ms = json.loads(ms)
            except Exception:
                return None, "json_hatasi"
        tools = row.get("tools") or None
    ms = normalize_messages(ms)
    if ms is None:
        return None, "gecersiz_mesaj"
    return (ms, tools), None


def est_tokens(ms: list) -> int:
    chars = sum(len(m["content"]) + len(m.get("reasoning_content", "")) + sum(len(t["function"]["arguments"]) + 20
                                                                              for t in m.get("tool_calls", [])) for m in ms)
    return int(chars / 3.6)


def _noisy(tc: dict) -> bool:
    """Araç adı kaybolmuş ('unknown_tool') ya da argümanları <placeholder> ile gizlenmiş çağrı."""
    return tc["function"]["name"] in ("unknown_tool", "") or bool(PLACEHOLDER.search(tc["function"]["arguments"]))


def repair_tools(ms: list, drop_noise: bool = True) -> tuple[list, bool]:
    """Eşleşmeyen araç çağrılarını ve sahipsiz araç sonuçlarını çıkarır (D2'de neredeyse her izde birkaç tane var).
    Platformların sohbet şablonları 'çağrı ↔ sonuç' eşleşmesi ister; eksik bir sonuç tüm örneği reddettirebilir."""
    calls = {tc["id"] for m in ms for tc in m.get("tool_calls", []) if not (drop_noise and _noisy(tc))}
    ok = calls & {m["tool_call_id"] for m in ms if m["role"] == "tool"}
    out, changed = [], False
    for m in ms:
        if m["role"] == "assistant" and m.get("tool_calls"):
            keep = [tc for tc in m["tool_calls"] if tc["id"] in ok]
            if len(keep) != len(m["tool_calls"]):
                changed = True
                m = dict(m)
                if keep:
                    m["tool_calls"] = keep
                else:
                    m.pop("tool_calls")
                    if not m["content"].strip() and not m.get("reasoning_content"):
                        continue                                     # içeriksiz ve çağrısız kalan mesaj atılır
        elif m["role"] == "tool" and m["tool_call_id"] not in ok:
            changed = True
            continue
        out.append(m)
    return out, changed


def filter_example(ms: list, args, stats: Counter | None = None) -> tuple[list | None, str | None]:
    """Süzgeçler; (mesajlar, None) ya da (None, atılma nedeni)."""
    n_calls = sum(len(m.get("tool_calls", [])) for m in ms)
    if not args.arac_onarma:
        ms, changed = repair_tools(ms, drop_noise=not args.arac_gurultusu)
        if changed and stats is not None:
            stats["onarilan_arac_eslesmesi"] += 1
        left = sum(len(m.get("tool_calls", [])) for m in ms)
        if n_calls and (n_calls - left) / n_calls > args.max_bozuk:
            return None, "arac_izi_bozuk"           # çağrıların çoğu gürültü/eşleşmesiz: onarılırsa geriye iz kalmaz
    # son mesaj asistanın düz cevabı olmalı: sondaki tool/araç çağrılı yarım turları kes
    if args.kesik:       # araç sonucuyla biten (kesik) izleri de al: son mesaj, cevabı olan bir araç sonucu olsun
        while ms and ms[-1]["role"] != "tool" and not (ms[-1]["role"] == "assistant" and ms[-1]["content"].strip()
                                                       and not ms[-1].get("tool_calls")):
            ms = ms[:-1]
    else:
        while ms and not (ms[-1]["role"] == "assistant" and not ms[-1].get("tool_calls") and ms[-1]["content"].strip()):
            ms = ms[:-1]
    if not ms or not any(m["role"] == "user" for m in ms):
        return None, "son_cevap_yok"
    if ms[0]["role"] == "tool" or any(m["role"] == "user" and m["content"].lstrip().startswith(CAVEAT) for m in ms):
        return None, "yerel_komut_gurultusu"
    calls = {tc["id"] for m in ms for tc in m.get("tool_calls", [])}
    answered = {m["tool_call_id"] for m in ms if m["role"] == "tool"}
    if calls != answered:
        return None, "arac_cagri_eslesmiyor"
    # roller sırası: assistant'tan önce user/tool gelmeli (ilk mesaj system olabilir)
    body = [m for m in ms if m["role"] != "system"]
    if not body or body[0]["role"] != "user":
        return None, "ilk_mesaj_user_degil"
    if args.dusunce == "sil":
        for m in ms:
            if m["role"] == "assistant":
                m["content"] = re.sub(r"<thought>.*?</thought>\s*", "", m["content"], flags=re.S)
                m.pop("reasoning_content", None)
    elif args.dusunce == "alana":
        for m in ms:
            if m["role"] == "assistant":
                t = re.match(r"\s*<thought>(.*?)</thought>\s*", m["content"], flags=re.S)
                if t:
                    m["reasoning_content"] = t.group(1).strip()
                    m["content"] = m["content"][t.end():]
    if est_tokens(ms) > args.max_token_ornek:
        return None, "cok_uzun"
    if est_tokens(ms) < args.min_token_ornek:
        return None, "cok_kisa"
    return ms, None


def process_row(src: str, row: dict, args, seen: set, stats: Counter):
    """Bir ham satırı işler. Döndürür (mesajlar, tools) ya da None; nedenleri stats'a yazar."""
    spec = SOURCES[src]
    got, why = rows_to_messages(spec["kind"], row)
    if why:
        stats["atildi:" + why] += 1
        return None
    ms, tools = got
    ms, why = filter_example(ms, args, stats)
    if why:
        stats["atildi:" + why] += 1
        return None
    h = hashlib.sha256(json.dumps(ms, sort_keys=True, ensure_ascii=False).encode()).digest()[:16]
    if h in seen:
        stats["atildi:tekrar"] += 1
        return None
    seen.add(h)
    return ms, tools


# --------------------------------------------------------------------------- kaynak okuma
def stream_remote(repo: str, path: str, n: int):
    """İlk n satırı indirmeden oku (--kuru)."""
    r = urllib.request.urlopen(f"https://huggingface.co/datasets/{repo}/resolve/main/{path}", timeout=120)
    try:
        for i, line in enumerate(r):
            if i >= n:
                break
            yield json.loads(line)
    finally:
        r.close()


def count_lines(path: str) -> int:
    n = 0
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            n += chunk.count(b"\n")
    return n


def d1_files(folders: list) -> list:
    from huggingface_hub import HfApi
    files = []
    for fo in folders:
        for e in HfApi().list_repo_tree(D1_REPO, path_in_repo=f"data/{fo}", repo_type="dataset"):
            if e.path.endswith(".jsonl"):
                files.append(e.path)
    return sorted(files)


def source_files(src: str, args) -> list:
    if src == "d2":
        return [f"{args.d2_bolum}/train.jsonl"]
    if src == "d1":
        return d1_files(args.d1)
    return SOURCES[src]["files"]


def enabled_sources(args) -> list:
    out = []
    if args.d3_max > 0:
        out.append("d3")
    if args.d2_max > 0:
        out.append("d2")
    if args.d1:
        out.append("d1")
    return out


# --------------------------------------------------------------------------- çıktı
class Writers:
    def __init__(self, out: Path):
        out.mkdir(parents=True, exist_ok=True)
        self.f = {}
        for split in ("train", "eval"):
            for kind in ("metin", "arac"):
                n = f"{split}_{kind}"
                self.f[n] = (open(out / f"{n}.jsonl", "w", encoding="utf-8"), open(out / f"{n}.meta.jsonl", "w", encoding="utf-8"))
        self.count = Counter()
        self.tokens = Counter()

    def write(self, ms: list, tools, meta: dict) -> None:
        kind = "arac" if any(m["role"] == "tool" or m.get("tool_calls") for m in ms) else "metin"
        h = int.from_bytes(hashlib.sha256(json.dumps(ms, sort_keys=True).encode()).digest()[:4], "big")
        split = "eval" if h % 100 == 0 else "train"                 # ~%1, belirleyici
        row = {"messages": ms}
        if tools:
            row["tools"] = tools
        f, m = self.f[f"{split}_{kind}"]
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        m.write(json.dumps(meta, ensure_ascii=False) + "\n")
        self.count[f"{split}_{kind}"] += 1
        self.tokens[f"{split}_{kind}"] += meta["token"]

    def close(self) -> None:
        for f, m in self.f.values():
            f.close()
            m.close()


def run(args) -> int:
    print(NOTICE, file=sys.stderr)
    rng = random.Random(args.seed)
    srcs = enabled_sources(args)
    if not srcs:
        print("hiçbir kaynak açık değil (--d3-max / --d2-max / --d1)", file=sys.stderr)
        return 2
    seen, manifest = set(), {"ayar": {k: v for k, v in vars(args).items()}, "uyari": NOTICE, "kaynaklar": {}}
    writers = None if args.kuru else Writers(Path(args.out))
    t0 = time.time()
    for src in srcs:
        spec, stats = SOURCES[src], Counter()
        target = {"d3": args.d3_max, "d2": args.d2_max, "d1": args.d1_max}[src]
        files = source_files(src, args)
        kept = 0
        if args.kuru:
            firsts = list({str(Path(f).parent): f for f in reversed(files)}.values())[::-1]   # klasör başına ilk dosya
            for path in firsts[:8]:
                for row in stream_remote(spec["repo"], path, args.kuru_satir):
                    stats["okunan"] += 1
                    stats[f"okunan[{Path(path).parent.name}]"] += 1
                    res = process_row(src, row, args, seen, stats)
                    if res:
                        kept += 1
                        stats["tut:arac" if any(m["role"] == "tool" for m in res[0]) else "tut:metin"] += 1
                        stats["tut_token"] += est_tokens(res[0])
                        if kept <= args.kuru_ornek:
                            print(f"\n--- {src} örnek {kept} (~{est_tokens(res[0])} token, {len(res[0])} mesaj)")
                            for m in res[0][:4]:
                                c = m["content"].replace("\n", " ")
                                print(f"   [{m['role']}] {c[:220]}" + (f"  (+{len(m['tool_calls'])} araç çağrısı)" if m.get("tool_calls") else ""))
        else:
            from huggingface_hub import hf_hub_download
            local = [hf_hub_download(spec["repo"], p, repo_type="dataset") for p in files]
            total = sum(count_lines(p) for p in local)
            p_acc = min(1.0, target * args.fazla_ornek / max(total, 1))
            print(f"[{src}] {total:,} satır → ≈{target:,} hedef (örnekleme olasılığı {p_acc:.3f})", file=sys.stderr)
            for path, loc in zip(files, local):
                with open(loc, encoding="utf-8") as f:
                    for line in f:
                        if kept >= target:
                            break
                        if p_acc < 1.0 and rng.random() > p_acc:
                            continue
                        try:
                            row = json.loads(line)
                        except Exception:
                            stats["atildi:json_satir"] += 1
                            continue
                        stats["okunan"] += 1
                        res = process_row(src, row, args, seen, stats)
                        if res:
                            ms, tools = res
                            tok = est_tokens(ms)
                            writers.write(ms, tools, {"kaynak": src, "depo": spec["repo"], "dosya": path, "lisans_beyani": spec["license"],
                                                       "token": tok})
                            kept += 1
                            stats["tut_token"] += tok
        stats["tutulan"] = kept
        manifest["kaynaklar"][src] = dict(stats)
        print(f"\n[{src}] okunan {stats['okunan']:,} | tutulan {kept:,} | ~{stats['tut_token'] / max(kept, 1):,.0f} token/örnek", file=sys.stderr)
        for k, v in sorted(stats.items()):
            if k.startswith("atildi:") and v:
                print(f"      {k:36s} {v:,}", file=sys.stderr)
    if writers:
        writers.close()
        manifest["cikti"] = {k: dict(ornek=writers.count[k], token=writers.tokens[k]) for k in writers.count}
        (Path(args.out) / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False, default=str))
        print(f"\nYazıldı: {args.out}  ({time.time() - t0:.0f} sn)", file=sys.stderr)
        for k, v in sorted(manifest["cikti"].items()):
            print(f"   {k:14s} {v['ornek']:>8,} örnek  ~{v['token'] / 1e6:,.1f}M token", file=sys.stderr)
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="veri/sft")
    p.add_argument("--kuru", action="store_true", help="indirmeden ilk satırları oku, istatistik ve örnek göster")
    p.add_argument("--kuru-satir", type=int, default=400, help="--kuru: dosya başına okunacak satır")
    p.add_argument("--kuru-ornek", type=int, default=2, help="--kuru: kaynak başına gösterilecek örnek")
    p.add_argument("--d3-max", type=int, default=40000, help="0 = kapalı")
    p.add_argument("--d2-max", type=int, default=0,
                   help="0 = kapalı (VARSAYILAN): D2'nin araç çağrıları gizlenmiş (5k: <file_path> yer tutucuları, 10k: unknown_tool) → araç izi olarak kullanılamaz")
    p.add_argument("--d2-bolum", default="5k", choices=["5k", "10k"], help="10k dosyası ~2 GB")
    p.add_argument("--d1", nargs="*", default=[], metavar="KLASÖR", help="örn: coding/fable5_agentic_sft (cybersecurity/* önerilmez)")
    p.add_argument("--d1-max", type=int, default=20000)
    p.add_argument("--max-token-ornek", type=int, default=16000, help="tahmini token üst sınırı; aşanlar atılır")
    p.add_argument("--min-token-ornek", type=int, default=8)
    p.add_argument("--dusunce", default="koru", choices=["koru", "sil", "alana"],
                   help="<thought>…</thought>: içerikte bırak / sil / reasoning_content alanına taşı")
    p.add_argument("--arac-gurultusu", action="store_true", help="unknown_tool ve <placeholder> argümanlı çağrıları ÇIKARMA")
    p.add_argument("--arac-onarma", action="store_true", help="eşleşmeyen araç çağrı/sonuçlarını ONARMA; eşleşmeyen örnekleri at")
    p.add_argument("--max-bozuk", type=float, default=0.2, help="çağrıların bu oranından fazlası gürültü/eşleşmesizse örneği at")
    p.add_argument("--kesik", action="store_true", help="düz cevapla bitmeyen (araç sonucuyla biten) izleri de tut")
    p.add_argument("--fazla-ornek", type=float, default=1.4, help="süzgeç kaybını telafi için örnekleme çarpanı")
    p.add_argument("--seed", type=int, default=0)
    return p


if __name__ == "__main__":
    sys.exit(run(parser().parse_args()))
