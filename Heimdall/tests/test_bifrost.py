"""Bifrost dönüşümü: sabit bellekli cache, tam paralel hesapla birebir aynı mı?"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heimdall.donustur import ConvLM, TeacherConfig  # noqa: E402

HF = dict(hidden_size=64, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, intermediate_size=128,
          vocab_size=97, rope_theta=10000.0, tie_word_embeddings=True)


def _model():
    torch.manual_seed(0)
    m = ConvLM(TeacherConfig(HF))
    m.convert(window=8, sinks=2, keep_global={2})           # son katman global dikkat kalır (hibrit)
    with torch.no_grad():                                    # yeni parametreleri rastgele yap: tüm yollar çalışsın
        for q in m.new_params():
            q.add_(torch.randn_like(q) * 0.3)
    return m.eval()


def test_cache_equals_full():
    m = _model()
    ids = torch.randint(0, 97, (2, 40))
    for mode in ("student", "teacher"):
        m.set_mode(mode)
        with torch.no_grad():
            full = m(ids)
            for split in (1, 5, 13):                         # prefill uzunluğu: çapa içi, pencere içi, pencere ötesi
                c = m.new_cache()
                parts = [m(ids[:, :split], c)] + [m(ids[:, t:t + 1], c) for t in range(split, 40)]
                err = (torch.cat(parts, 1) - full).abs().max().item()
                assert err < 1e-4, (mode, split, err)


def test_memory_constant():
    m = _model()
    m.set_mode("student")
    sizes = []
    with torch.no_grad():
        for n in (50, 400):
            c = m.new_cache()
            m(torch.randint(0, 97, (1, n)), c)
            conv_bytes = sum(v.numel() * v.element_size() for layer in c["layers"][:2] for v in layer.values()
                             if torch.is_tensor(v))
            sizes.append(conv_bytes)
    assert sizes[0] == sizes[1], sizes                       # dönüştürülen katmanlar: bağlamdan bağımsız


if __name__ == "__main__":
    test_cache_equals_full()
    print("OK cache = tam hesap (öğrenci ve öğretmen modu)")
    test_memory_constant()
    print("OK dönüştürülen katman belleği sabit")
