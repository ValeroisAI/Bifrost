"""SFT veri birleştirici: normalizasyon, araç izi onarımı/gürültü süzgeci, tekilleştirme, bölme, uçtan uca yazma."""

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

import sft_veri as S  # noqa: E402

ARGS = S.parser().parse_args([])


def call(i, name="read_file", args='{"path": "a.py"}'):
    return {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": args}}


def trace(n_calls=3, noisy=0, drop_result=None, final=True):
    ms = [{"role": "user", "content": "fix the bug in a.py " * 3}]
    for i in range(n_calls):
        bad = i < noisy
        ms.append({"role": "assistant", "content": f"<thought>step {i}</thought>",
                   "tool_calls": [call(i, "unknown_tool" if bad else "read_file", "{}" if bad else '{"path": "a.py"}')]})
        if i != drop_result:
            ms.append({"role": "tool", "tool_call_id": f"call_{i}", "content": f"result {i} " * 5})
    if final:
        ms.append({"role": "assistant", "content": "Done: the bug was an off-by-one in a.py, fixed and tests pass."})
    return ms


def test_normalize():
    ms = S.normalize_messages([{"role": "user", "content": [{"type": "text", "text": "hi "}, {"type": "text", "text": "there"}]},
                               {"role": "assistant", "content": None, "tool_calls": [call(0, args={"path": "x"})]},
                               {"role": "tool", "tool_call_id": 0, "content": "ok"}])
    assert ms[0]["content"] == "hi there"
    assert ms[1]["tool_calls"][0]["function"]["arguments"] == '{"path": "x"}'      # dict → JSON metni
    assert ms[2]["tool_call_id"] == "0"
    assert S.normalize_messages([{"role": "user", "content": [{"type": "image_url"}]}]) is None
    assert S.normalize_messages([{"role": "robot", "content": "x"}]) is None
    assert S.normalize_messages([]) is None


def test_d1_instruction_parsing():
    d = {"messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": "q?"}]}
    for text in (str(d), json.dumps(d)):                                           # python repr ve JSON
        assert S.parse_instruction(text) == d["messages"]
    assert S.parse_instruction("plain question") == [{"role": "user", "content": "plain question"}]
    got, why = S.rows_to_messages("instr", {"instruction": "", "response": ""})
    assert got is None and why == "bos_satir"
    (ms, tools), why = S.rows_to_messages("instr", {"instruction": str(d), "response": "a!"})
    assert why is None and [m["role"] for m in ms] == ["system", "user", "assistant"]


def test_repair_and_noise():
    ok, why = S.filter_example(trace(3), ARGS, Counter())
    assert why is None and len([m for m in ok if m["role"] == "tool"]) == 3
    # eksik sonuç: o çağrı çıkarılır, geri kalan tutarlı kalır
    st = Counter()
    fixed, why = S.filter_example(trace(5, drop_result=2), ARGS, st)
    calls = {tc["id"] for m in fixed for tc in m.get("tool_calls", [])}
    results = {m["tool_call_id"] for m in fixed if m["role"] == "tool"}
    assert why is None and calls == results and "call_2" not in calls and st["onarilan_arac_eslesmesi"] == 1
    # gürültülü çağrıların çoğu → örnek atılır; azsa çıkarılıp tutulur
    assert S.filter_example(trace(5, noisy=4), ARGS, Counter())[1] == "arac_izi_bozuk"
    few, why = S.filter_example(trace(10, noisy=1), ARGS, Counter())
    assert why is None and all(not S._noisy(tc) for m in few for tc in m.get("tool_calls", []))
    # placeholder argüman
    t = trace(5)
    t[1]["tool_calls"][0]["function"]["arguments"] = '{"path": "<file_path>"}'
    assert all(tc["id"] != "call_0" for m in S.filter_example(t, ARGS, Counter())[0] for tc in m.get("tool_calls", []))


def test_final_and_limits():
    assert S.filter_example(trace(3, final=False), ARGS, Counter())[1] == "son_cevap_yok"   # araç sonucuyla biten iz
    kes = S.parser().parse_args(["--kesik"])
    ok, why = S.filter_example(trace(3, final=False), kes, Counter())
    assert why is None and ok[-1]["role"] == "tool"
    assert S.filter_example([{"role": "user", "content": "x" * 100000}, {"role": "assistant", "content": "y"}], ARGS, Counter())[1] == "cok_uzun"
    caveat = [{"role": "user", "content": S.CAVEAT + " noise"}, {"role": "assistant", "content": "ok then, fine thanks"}]
    assert S.filter_example(caveat, ARGS, Counter())[1] == "yerel_komut_gurultusu"
    th = S.parser().parse_args(["--dusunce", "alana"])
    ms = S.filter_example(trace(2), th, Counter())[0]
    assert ms[1]["reasoning_content"] == "step 0" and ms[1]["content"] == ""
    th = S.parser().parse_args(["--dusunce", "sil"])
    assert "<thought>" not in json.dumps(S.filter_example(trace(2), th, Counter())[0])


def test_end_to_end(monkeypatch_dir=None):
    tmp = Path(tempfile.mkdtemp())
    rows = []
    for i in range(300):
        rows.append(json.dumps({"messages": [{"role": "user", "content": f"question number {i} about python lists?"},
                                             {"role": "assistant", "content": f"answer {i}: use a list comprehension here."}]}))
    rows.append(rows[0])                                                           # tekrar
    rows.append(json.dumps({"messages": json.dumps(trace(4))}))                    # messages JSON METNİ olarak (D3 biçimi)
    rows.append("{bozuk json")
    src = tmp / "d3.jsonl"
    src.write_text("\n".join(rows) + "\n")
    import huggingface_hub
    orig = huggingface_hub.hf_hub_download
    huggingface_hub.hf_hub_download = lambda repo, path, repo_type=None: str(src)
    try:
        out = tmp / "out"
        args = S.parser().parse_args(["--out", str(out), "--d3-max", "1000", "--fazla-ornek", "1.0"])
        assert S.run(args) == 0
    finally:
        huggingface_hub.hf_hub_download = orig
    man = json.loads((out / "manifest.json").read_text())
    d3 = man["kaynaklar"]["d3"]
    assert d3["atildi:tekrar"] == 1 and d3["atildi:json_satir"] == 1 and d3["tutulan"] == 301
    n_text = sum(1 for _ in open(out / "train_metin.jsonl")) + sum(1 for _ in open(out / "eval_metin.jsonl"))
    n_tool = sum(1 for _ in open(out / "train_arac.jsonl")) + sum(1 for _ in open(out / "eval_arac.jsonl"))
    assert (n_text, n_tool) == (300, 1)
    for name in ("train_metin", "eval_metin", "train_arac", "eval_arac"):             # meta, satırlarla hizalı
        a = [json.loads(line) for line in open(out / f"{name}.jsonl")]
        b = [json.loads(line) for line in open(out / f"{name}.meta.jsonl")]
        assert len(a) == len(b) and all(set(x) == {"messages"} for x in a)             # satır içinde fazladan alan YOK
        assert all(m["kaynak"] == "d3" and m["lisans_beyani"] for m in b)
    assert 0 < sum(1 for _ in open(out / "eval_metin.jsonl")) < 20                      # ~%1 değerlendirme
    first = json.loads(open(out / "train_arac.jsonl").readline() if (out / "train_arac.jsonl").stat().st_size
                       else open(out / "eval_arac.jsonl").readline())
    assert any(m["role"] == "tool" for m in first["messages"])


if __name__ == "__main__":
    test_normalize()
    test_d1_instruction_parsing()
    test_repair_and_noise()
    test_final_and_limits()
    test_end_to_end()
    print("OK sft_veri: normalizasyon, D1 ayrıştırma, onarım/gürültü, son cevap/sınırlar, uçtan uca yazma")
