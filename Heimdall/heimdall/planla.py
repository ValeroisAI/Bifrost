"""
planla.py — Bifrost dönüşümü bu kartta olur mu? Ağırlık indirmeden, yalnız config.json'dan hesaplar.

    python -m heimdall.planla --model Qwen/Qwen3-8B --gpu 16 --ram 32
    python -m heimdall.planla --model microsoft/bitnet-b1.58-2B-4T-bf16 --gpu 16 --seq-len 1024 --batch 1 --grad-ckpt

Çıktı: parametre sayısı, öğretmen ağırlık belleği (bf16 / --q4 / BitNet üçlü), yeni parametre sayısı, KV cache (orijinal ve
dönüşmüş), eğitim belleği tahmini, "sığar mı" kararı.

Doğruluk: PARAMETRE SAYILARI ve KV/durum baytları kesindir (testlerde gerçek ConvLM ile birebir karşılaştırılır). Eğitim
AKTİVASYON belleği ise tahmindir (±%40): ilk GPU koşusunda `tepe GPU` satırlarına göre kalibre edilmelidir.
"""

import argparse
import json
import math
from pathlib import Path

SUPPORTED = {"llama": "Llama / SmolLM", "qwen2": "Qwen2 / 2.5", "qwen3": "Qwen3", "bitnet": "BitNet b1.58 (bf16 ana ağırlık)"}
GPUS = {"9070xt": 16, "rx7900xtx": 24, "rtx4090": 24, "rtx5090": 32, "a100": 80, "h100": 80, "mi300x": 192}
GB = 2 ** 30


def load_config(name: str) -> dict:
    p = Path(name)
    if (p / "config.json").exists():
        return json.loads((p / "config.json").read_text())
    from huggingface_hub import hf_hub_download
    return json.loads(Path(hf_hub_download(name, "config.json")).read_text())


def _text_config(hf: dict) -> tuple[dict, list[str]]:
    """Çok modlu checkpoint'lerde metin kısmını al; desteklenmeyen özellikleri listele."""
    notes = []
    if "text_config" in hf:
        notes.append("çok modlu checkpoint: görüntü encoder'ı atlanıp yalnız metin decoder'ı yüklenmeli (yükleyici adaptörü YOK)")
        hf = {**hf["text_config"], "_arch": hf.get("architectures")}
    return hf, notes


def check_support(hf: dict) -> list[str]:
    """Bu kodla dönüştürülemeyecek nedenleri döndürür (boş liste = destekleniyor)."""
    why = []
    mt = hf.get("model_type")
    if mt not in SUPPORTED:
        why.append(f"model_type '{mt}' desteklenmiyor (destek: {', '.join(SUPPORTED)})")
    if any("expert" in k for k in hf) or "moe" in str(hf.get("architectures", "")).lower():
        why.append("MoE modeller desteklenmiyor")
    if hf.get("rope_scaling"):
        why.append("rope_scaling henüz desteklenmiyor (uzun bağlam modellerinde sık)")
    if hf.get("hidden_act", "silu") not in ("silu", "relu2"):
        why.append(f"aktivasyon {hf.get('hidden_act')} desteklenmiyor")
    if mt == "bitnet" and (hf.get("quantization_config") or {}).get("quantization_mode", "online") != "online":
        why.append("BitNet: yalnız bf16 ana ağırlıklı ('online') sürüm destekleniyor")
    return why


def analyze(hf: dict, *, window: int = 64, sinks: int = 4, seq_len: int = 1024, batch: int = 1, q4: bool = False,
            sensitive_bits: int = 8, grad_ckpt: bool = False, keep_global: int = 0) -> dict:
    multimodal = "text_config" in hf
    hf, notes = _text_config(hf)
    d, L = hf["hidden_size"], hf["num_hidden_layers"]
    h = hf["num_attention_heads"]
    kv = hf.get("num_key_value_heads", h)
    dh = hf.get("head_dim") or d // h
    inter, V = hf["intermediate_size"], hf["vocab_size"]
    tie = hf.get("tie_word_embeddings", False)
    mt = hf.get("model_type")
    bitnet = mt == "bitnet"
    attn_bias = hf.get("attention_bias", mt == "qwen2")
    qk_norm = mt == "qwen3"
    issues = check_support(hf)
    notes += issues

    # ---- parametreler (donustur.ConvLM ile aynı yapı)
    attn_lin = d * h * dh + 2 * d * kv * dh + h * dh * d
    attn_bias_p = (h * dh + 2 * kv * dh) if attn_bias else 0
    v_proj, down_proj = d * kv * dh, inter * d
    mlp = 3 * d * inter
    norms = 2 * d + (2 * dh if qk_norm else 0) + ((h * dh + inter) if bitnet else 0)
    layer_total = attn_lin + attn_bias_p + mlp + norms
    embed = V * d
    head = 0 if tie else V * d
    total = L * layer_total + embed + head + d
    n_conv = L - keep_global
    new_per_layer = 2 * h * dh * dh + 3 * h * d + 7 * h
    new_params = n_conv * new_per_layer

    # ---- donuk öğretmen ağırlık baytları (donustur: bf16 / Q4Linear / TernaryLinear int8)
    def lin_bytes(n_params: int, is_sensitive: bool) -> float:
        if bitnet:
            return n_params * 1.0                                         # int8 üçlü (tensör başına tek ölçek)
        if q4:
            bits = sensitive_bits if is_sensitive else 4
            return n_params * (bits / 8 + 0.125)                          # + grup başına fp16 ölçek ve sıfır noktası
        return n_params * 2.0
    per_layer_w = (lin_bytes(v_proj, True) + lin_bytes(down_proj, True)
                   + lin_bytes(attn_lin - v_proj, False) + lin_bytes(mlp - down_proj, False)
                   + (attn_bias_p + norms) * 2.0)
    embed_b = embed * 2.0
    head_b = 0.0 if tie else (head * (0.5 + 0.125) if q4 else head * 2.0)
    weights = L * per_layer_w + embed_b + head_b
    weights_bf16 = total * 2.0

    # ---- KV cache ve dönüşmüş durum (donustur.main'deki formüllerle aynı)
    kv_teacher = lambda n: L * 2 * kv * dh * n * 2
    state = n_conv * (h * dh * dh * 4 + 2 * kv * dh * (window + sinks) * 2) + keep_global * 0
    # keep_global katmanları tam KV tutar; bağlama bağlı bu kısım ayrıca eklenir
    kv_global = lambda n: keep_global * 2 * kv * dh * n * 2

    # ---- eğitim belleği: donuk ağırlık + yeni parametreler (fp32 + grad + Adam = 16 B) + aktivasyon + KL logitleri
    bt = batch * seq_len
    ws_token = (h * dh * (2 + sinks / window) * 4 * 2 + h * (2 * window + sinks) * 4 * 3          # pencere dikkati (fp32 ara değerler)
                + (h + 2 * kv) * dh * 2 * 2 + 3 * inter * 2 * 2 + 10 * d * 4)                      # q/k/v/o, FFN, norm/residual
    ws = bt * ws_token
    saved = L * bt * d * 2 + ws if grad_ckpt else L * ws
    logits = 4 * bt * V * 4                                                                          # öğretmen+öğrenci logp, softmax, grad
    train = (weights + new_params * 16 + saved + logits) * 1.15                                      # %15 parçalanma payı
    # aşama A (katman katman, diskten): şu an kodda YOK; kuramsal alt sınır
    layerwise_one = per_layer_w + new_per_layer * 16 + ws + bt * d * 2 * 2

    return dict(
        model_type=mt, layers=L, d_model=d, heads=h, kv_heads=kv, head_dim=dh, vocab=V, tie=tie,
        supported=not issues and not multimodal, notes=notes,
        params=total, params_layer=layer_total, new_params=new_params,
        weights_bytes=weights, weights_bf16_bytes=weights_bf16,
        kv_teacher={n: kv_teacher(n) for n in (8192, 131072, 1048576)},
        kv_converted={n: state + kv_global(n) for n in (8192, 131072, 1048576)}, state_bytes=state,
        train_bytes=train, train_parts=dict(weights=weights, new=new_params * 16, activations=saved, logits=logits),
        layerwise_layer_bytes=layerwise_one, mode="BitNet üçlü int8" if bitnet else ("4-bit" if q4 else "bf16"),
        settings=dict(window=window, sinks=sinks, seq_len=seq_len, batch=batch, grad_ckpt=grad_ckpt, q4=q4),
    )


def verdict(need: float, have: float) -> str:
    return "yeşil" if need <= 0.85 * have else ("sarı" if need <= have else "kırmızı")


def fit(r: dict, gpu_gb: float, ram_gb: float = 0.0) -> dict:
    have = gpu_gb * GB
    out = {}
    out["agirliklar"] = (r["weights_bytes"], verdict(r["weights_bytes"] * 1.1, have))
    out["donusum_egitimi"] = (r["train_bytes"], verdict(r["train_bytes"], have))
    # kuramsal katman-katman: aynı anda kaç katman? (gömme ve KL logitleri ayrı tutulur)
    fixed = 0.5 * GB
    k = int(max(0, have * 0.85 - fixed) // max(r["layerwise_layer_bytes"], 1))
    out["katman_katman_kuramsal"] = (min(k, r["layers"]), "yeşil" if k >= 1 else "kırmızı")
    if ram_gb:
        out["ram_bf16"] = (r["weights_bf16_bytes"], verdict(r["weights_bf16_bytes"] * 1.1, ram_gb * GB))
    return out


def _fmt(b: float) -> str:
    return f"{b / GB:,.1f} GB" if b >= GB / 2 else f"{b / 2**20:,.1f} MB"


def render(name: str, r: dict, f: dict, gpu_gb: float) -> str:
    s = r["settings"]
    L = [f"Model: {name}  ({r['model_type']}, {r['layers']} katman, d={r['d_model']}, {r['heads']}q/{r['kv_heads']}kv başı, "
         f"vocab {r['vocab']:,})",
         f"Parametre: {r['params'] / 1e9:.2f} B | yeni (öğrenilecek) parametre: {r['new_params'] / 1e6:.1f} M "
         f"({100 * r['new_params'] / r['params']:.1f} %)"]
    for n in r["notes"]:
        L.append(f"  ! {n}")
    L += ["", f"Donuk öğretmen ağırlıkları ({r['mode']}): {_fmt(r['weights_bytes'])}   (bf16 olsaydı {_fmt(r['weights_bf16_bytes'])})",
          "", "KV cache (bir dizi için)         orijinal      dönüşmüş (sabit)"]
    for n, name_ in ((8192, "8K"), (131072, "128K"), (1048576, "1M")):
        L.append(f"  {name_:5s}                      {_fmt(r['kv_teacher'][n]):>12s}   {_fmt(r['kv_converted'][n]):>12s}")
    mark = {"yeşil": "[ OK  ]", "sarı": "[SINIRDA]", "kırmızı": "[SIĞMAZ]"}
    p = r["train_parts"]
    L += ["", f"Eğitim belleği ({gpu_gb:g} GB kart; seq {s['seq_len']}, batch {s['batch']}, "
          f"{'grad-ckpt' if s['grad_ckpt'] else 'grad-ckpt YOK'}):",
          f"  ağırlık {_fmt(p['weights'])} + yeni param/Adam {_fmt(p['new'])} + aktivasyon~ {_fmt(p['activations'])} "
          f"+ KL logit~ {_fmt(p['logits'])}",
          f"  toplam ~{_fmt(r['train_bytes'])}  {mark[f['donusum_egitimi'][1]]}   (aktivasyon ve logit kısmı TAHMİN)"]
    k = f["katman_katman_kuramsal"][0]
    L.append(f"  katman-katman (kuramsal, kodda henüz yok): aynı anda ~{k}/{r['layers']} katman")
    if "ram_bf16" in f:
        L.append(f"  RAM: bf16 ağırlıklar {_fmt(f['ram_bf16'][0])} {mark[f['ram_bf16'][1]]}")
    return "\n".join(L)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF adı ya da config.json içeren klasör")
    p.add_argument("--gpu", default="16", help=f"VRAM GB ya da ad ({', '.join(GPUS)})")
    p.add_argument("--ram", type=float, default=0.0, help="sistem RAM GB (isteğe bağlı)")
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--sinks", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--q4", action="store_true")
    p.add_argument("--hassas-bit", type=int, default=8, choices=[4, 8])
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    gpu = float(GPUS.get(a.gpu.lower(), a.gpu))
    hf = load_config(a.model)
    r = analyze(hf, window=a.window, sinks=a.sinks, seq_len=a.seq_len, batch=a.batch, q4=a.q4,
                sensitive_bits=a.hassas_bit, grad_ckpt=a.grad_ckpt)
    f = fit(r, gpu, a.ram)
    print(json.dumps(dict(plan=r, fit=f), indent=1, default=str) if a.json else render(a.model, r, f, gpu))


if __name__ == "__main__":
    main()
