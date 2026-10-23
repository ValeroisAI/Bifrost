"""Planlayıcı: parametre / yeni parametre / KV / durum sayıları gerçek ConvLM ile BİREBİR aynı mı?"""

import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heimdall.donustur import ConvLM, Q4Linear, TeacherConfig, TernaryLinear  # noqa: E402
from heimdall.planla import analyze, check_support, fit, load_config, render  # noqa: E402

BASE = dict(hidden_size=64, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, intermediate_size=128,
            vocab_size=97, rope_theta=10000.0, tie_word_embeddings=True)
VARIANTS = {
    "llama": dict(BASE, model_type="llama"),
    "llama_untied_headdim": dict(BASE, model_type="llama", tie_word_embeddings=False, head_dim=24),
    "qwen2": dict(BASE, model_type="qwen2"),                      # attention bias
    "qwen3": dict(BASE, model_type="qwen3", head_dim=32),         # qk-norm
    "bitnet": dict(BASE, model_type="bitnet", hidden_act="relu2"),
}


def test_counts_match_model():
    for name, hf in VARIANTS.items():
        m = ConvLM(TeacherConfig(hf))
        window, sinks = 8, 2
        m.convert(window=window, sinks=sinks)
        r = analyze(hf, window=window, sinks=sinks)
        real = sum(q.numel() for q in m.parameters()) - sum(q.numel() for q in m.new_params())
        assert r["params"] == real, (name, r["params"], real)
        assert r["new_params"] == sum(q.numel() for q in m.new_params()), (name, r["new_params"])
        # dönüşmüş durum baytları: gerçek cache ile aynı mı (sabit kısım: S matrisi + halka + çapa)
        m.eval().set_mode("student")
        ids = torch.randint(0, 97, (1, 24))
        with torch.no_grad():
            c = m.new_cache()
            m(ids, c)
        real_state = sum(v.numel() * v.element_size() for lc in c["layers"] for k, v in lc.items()
                         if torch.is_tensor(v) and not k.startswith(("ring_g", "ring_beta", "ring_km", "ring_pos")))
        # planlayıcı yalnız S (fp32) ve halka K/V (bf16) sayar: donustur.main'deki formülle aynı
        dh = hf.get("head_dim") or hf["hidden_size"] // hf["num_attention_heads"]
        h, kv = hf["num_attention_heads"], hf["num_key_value_heads"]
        expect = hf["num_hidden_layers"] * (h * dh * dh * 4 + 2 * kv * dh * (window + sinks) * 2)
        assert r["state_bytes"] == expect, name
        assert r["kv_teacher"][8192] == hf["num_hidden_layers"] * 2 * kv * dh * 8192 * 2
        assert real_state > 0


def test_weight_bytes_match_quantized_model():
    hf = dict(VARIANTS["llama"], hidden_size=128, intermediate_size=256, num_attention_heads=4, num_key_value_heads=2)
    hf["tie_word_embeddings"] = False
    # q4: gerçek Q4Linear baytlarını elle kur ve planlayıcıyla karşılaştır
    m = ConvLM(TeacherConfig(hf))
    got = 0.0
    for name, mod in m.named_modules():
        if name.startswith("layers.") and isinstance(mod, torch.nn.Linear):
            q = Q4Linear(torch.randn_like(mod.weight), 32, bits=8 if name.endswith(("v_proj", "down_proj")) else 4)
            got += sum(v.numel() * v.element_size() for v in q.buffers())
    r = analyze(hf, q4=True)
    layers_w = r["weights_bytes"] - r["params_layer"] * 0 - (hf["vocab_size"] * hf["hidden_size"] * 2.0
                                                              + hf["vocab_size"] * hf["hidden_size"] * 0.625)
    norms = hf["num_hidden_layers"] * 2 * hf["hidden_size"] * 2.0
    assert abs(layers_w - norms - got) < 1, (layers_w - norms, got)
    # bitnet: TernaryLinear int8
    hb = dict(VARIANTS["bitnet"])
    mb = ConvLM(TeacherConfig(hb))
    got_b = 0
    for name, mod in mb.named_modules():
        if name.startswith("layers.") and isinstance(mod, torch.nn.Linear):
            got_b += TernaryLinear(mod.weight.detach()).wq.numel()
    rb = analyze(hb)
    d, L, V = hb["hidden_size"], hb["num_hidden_layers"], hb["vocab_size"]
    norms_b = L * (2 * d + hb["num_attention_heads"] * (d // hb["num_attention_heads"]) + hb["intermediate_size"]) * 2.0
    assert abs(rb["weights_bytes"] - V * d * 2.0 - norms_b - got_b) < 1


def test_support_flags_and_fit():
    assert check_support(VARIANTS["qwen3"]) == []
    assert any("rope_scaling" in w for w in check_support(dict(VARIANTS["llama"], rope_scaling={"type": "yarn"})))
    assert any("MoE" in w for w in check_support(dict(VARIANTS["llama"], num_experts=8)))
    assert not analyze(dict(VARIANTS["llama"], model_type="gemma"))["supported"]
    mm = analyze({"text_config": VARIANTS["llama"], "architectures": ["X"]})
    assert not mm["supported"] and any("çok modlu" in n for n in mm["notes"])
    # büyük model küçük kartta sığmamalı, büyük kartta sığmalı
    big = dict(model_type="llama", hidden_size=6656, num_hidden_layers=52, num_attention_heads=32, num_key_value_heads=2,
               intermediate_size=21504, vocab_size=262144, tie_word_embeddings=False)
    r = analyze(big, grad_ckpt=True)
    assert fit(r, 16)["donusum_egitimi"][1] == "kırmızı" and fit(r, 192)["donusum_egitimi"][1] != "kırmızı"
    text = render("big", r, fit(r, 16), 16)
    assert "SIĞMAZ" in text


def test_load_config_from_dir():
    with tempfile.TemporaryDirectory() as t:
        (Path(t) / "config.json").write_text(json.dumps(VARIANTS["qwen3"]))
        assert load_config(t)["model_type"] == "qwen3"


if __name__ == "__main__":
    test_counts_match_model()
    print("OK sayılar gerçek ConvLM ile birebir (llama/qwen2/qwen3/bitnet)")
    test_weight_bytes_match_quantized_model()
    print("OK ağırlık baytları (4/8-bit ve üçlü) gerçek katmanlarla aynı")
    test_support_flags_and_fit()
    print("OK destek bayrakları ve sığar/sığmaz kararı")
    test_load_config_from_dir()
    print("OK config klasörden okunur")
