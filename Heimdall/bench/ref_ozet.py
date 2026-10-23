"""
Heimdall sonuçlarını (runs/hafta/lmeval.json) ve lm_eval CLI ile alınan referans modelleri (runs/ref_*) tek tabloda toplar.

    python bench/ref_ozet.py                       # runs/hafta/lmeval.json + runs/ref_*
    python bench/ref_ozet.py --ref 'runs/ref_*' --heimdall runs/hafta/lmeval.json

Her görev için tek ölçüt (lm-eval'in yaygın kullanımı): arc_easy acc, arc_challenge acc_norm, hellaswag acc_norm, piqa acc,
winogrande acc, lambada acc (+ppl), wikitext word_perplexity (düşük iyi; tokenizer'dan bağımsız tek karşılaştırılabilir sayı).
Eğitim token sayıları buraya YAZILMAZ: model kartlarından kontrol edip tabloya elle ekle.
"""
import argparse
import glob
import json
from pathlib import Path

ROWS = [("arc_easy", "acc"), ("arc_challenge", "acc_norm"), ("hellaswag", "acc_norm"), ("piqa", "acc"),
        ("winogrande", "acc"), ("lambada_openai", "acc"), ("lambada_openai", "perplexity"), ("wikitext", "word_perplexity")]
CHANCE = {"arc_easy": 0.25, "arc_challenge": 0.25, "hellaswag": 0.25, "piqa": 0.5, "winogrande": 0.5}


def load_results(path: Path) -> dict | None:
    data = json.loads(path.read_text())
    res = data.get("results")
    if not isinstance(res, dict):
        return None
    out = {}
    for task, r in res.items():
        out[task] = {k.split(",")[0]: v for k, v in r.items() if isinstance(v, (int, float)) and "stderr" not in k}
    return out


def find_ref(pattern: str) -> dict:
    found = {}
    for p in sorted(glob.glob(pattern)):
        P = Path(p)
        files = [P] if P.is_file() else sorted(P.rglob("results*.json"))
        if files:                                           # birden çok koşu varsa en yenisi
            r = load_results(files[-1])
            if r:
                found[P.stem.replace("ref_", "")] = r
    return found


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--heimdall", default="runs/hafta/lmeval.json")
    p.add_argument("--ref", default="runs/ref_*")
    a = p.parse_args()
    models = {}
    if Path(a.heimdall).exists():
        r = load_results(Path(a.heimdall))
        if r:
            models["Heimdall-337M"] = r
    models.update(find_ref(a.ref))
    if not models:
        raise SystemExit("sonuç bulunamadı")
    names = list(models)
    print(f"{'görev (ölçüt)':30s} {'şans':>6s} " + " ".join(f"{n[:16]:>16s}" for n in names))
    for task, metric in ROWS:
        cells = []
        for n in names:
            v = models[n].get(task, {}).get(metric)
            cells.append(f"{'-':>16s}" if v is None else f"{v:16.4f}" if metric in ("acc", "acc_norm") else f"{v:16.2f}")
        ch = CHANCE.get(task)
        print(f"{task + ' (' + metric + ')':30s} {'' if ch is None else f'{ch:6.2f}':>6s} " + " ".join(cells))
    print("\nNot: eğitim token sayıları model kartlarından elle eklenmeli; tek koşu, tek tohum, standart hata ~±1 puan.")


if __name__ == "__main__":
    main()
