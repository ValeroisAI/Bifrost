# Heimdall

Valerois AI dil modeli: mimari, eğitim sistemi ve OpenAI uyumlu servis. Aynı kod NVIDIA (CUDA) ve AMD (ROCm) GPU'larda, ayrıca CPU'da çalışır.

> Heimdall, Bifröst köprüsünün bekçisidir: çok uzağı görür ve hiçbir şeyi unutmaz.

**Durum:** Kod CPU'da test edildi (`tests/`, 5/5). Eğitim, paketleme, yükleme ve API zinciri uçtan uca çalıştı. GPU'da henüz koşulmadı. "Transformer'ı geçiyor" iddiası literatüre dayanıyor; bizim verimizde **ölçülmedi**. Ölçüm tek komutla yapılır, bkz. [Transformer ile karşılaştırma](#transformer-ile-karşılaştırma-kanıt-prosedürü).

---

## 1. Mimari

```
girdi → Embedding ─┬─ [ D ] Gated DeltaNet ─┐
                   ├─ [ D ] Gated DeltaNet  │  × (n_layers / 4)
                   ├─ [ D ] Gated DeltaNet  │  her katmandan sonra SwiGLU FFN
                   └─ [ A ] Global dikkat ──┘  (pre-RMSNorm, artık bağlantı)
                 → RMSNorm → LM başı (embedding ile ortak) → softcap(30)
```

Katmanların dörtte üçü **Gated DeltaNet**, dörtte biri **global dikkattir** (`layout="DDDA"`).

### D: Gated DeltaNet (doğrusal zamanlı hafıza)
```
[q;k;v] = SiLU(ShortConv₄(W x))          q̂ = q/‖q‖·d^-½,  k̂ = k/‖k‖
α = exp(−e^A · softplus(W_a x + dt))      β = σ(W_b x)
S_t = α S_{t-1} + β k̂ (v − α S_{t-1}ᵀ k̂)ᵀ      o_t = S_tᵀ q̂
y = W_o( RMSNorm(o) ⊙ SiLU(W_g x) )
```
- Hafıza `S` sabit boyutlu bir matristir (kafa başına 128×128). Bağlam uzadıkça büyümez.
- **Delta kuralı:** aynı anahtara yeni değer gelince eskisi silinir ("proje id 1 → 2 → 3" sorusunda cevap 3). Salt toplamsal doğrusal dikkatin (Mamba, RetNet) yapamadığı "üzerine yazma" budur.
- **Arşiv kafaları (α ≡ 1):** kafaların bir kısmı hiç unutmaz. Önceki deneylerimizde öğrenilen unutma katsayısı 0.99997 çıktı; bu değerle 65K tokende bilginin %88'i siliniyordu. α=1 kafaları bunu düzeltti.
- Eğitimde chunk-paralel çalışır (UT dönüşümü, O(T·C·d)), üretimde token başına sabit iş yapar.

### A: Global dikkat
- **GQA** (12 sorgu, 4 KV kafası): KV cache 3 kat küçük.
- **QK-norm:** büyük öğrenme hızında kararlılık.
- **Çıkış kapısı:** `y = W_o(SDPA(q,k,v) ⊙ σ(W_gate x))`. Qwen3-Next'te dikkat batağını (attention sink) ve aktivasyon patlamalarını azaltıyor.
- **NoPE:** hibritte dikkat katmanları konum kodlaması kullanmaz; sıra bilgisini DeltaNet ve conv taşır (Kimi Linear ile aynı tercih). Eğitim uzunluğunun ötesine genelleme için RoPE'nin frekans sınırı ortadan kalkar.
- Çekirdek: PyTorch SDPA. CUDA'da FlashAttention, ROCm'da AOTriton kullanılır.

### Neden Transformer'ı geçmesi beklenir?
| Kanıt | Sonuç |
|---|---|
| Gated DeltaNet (Yang ve ark., ICLR 2025) | 1.3B / 100B token: hibrit GDN + dikkat, Transformer++, Mamba2 ve saf GDN'den daha iyi LM ve hatırlama skoru |
| Qwen3-Next (2025) | 3 Gated DeltaNet : 1 kapılı dikkat (MoE ile birlikte). Dense Qwen3-32B'nin %10'undan az eğitim maliyetiyle benzer/üstün kalite, 32K+ bağlamda ~10× verim |
| Kimi Linear (2025) | 3 KDA (delta ailesi) : 1 NoPE global dikkat. Kısa bağlam, uzun bağlam ve RL'de tam dikkati geçiyor; KV cache %75 küçük |
| Bizim ölçümümüz (CPU, `08_BIFROST_H/DURUM.md`) | Global dikkatsiz Kuzgun, eşit sürede Transformer'ın gerisinde kaldı. Bu yüzden 1/4 global dikkat geri eklendi |

Kısaca: saf doğrusal modeller kopyalama ve tam hatırlamada Transformer'a yeniliyor. Az sayıda global dikkat katmanı bu açığı kapatıyor. DeltaNet katmanları ise hem daha güçlü bir kısa-orta menzil karıştırıcısı hem de ucuz uzun hafıza sağlıyor.

### Bellek ve hız (temel ön ayar, ~100M, bf16, dizi başına)
| Bağlam | Transformer (MHA, 12 dikkat katmanı) | **Heimdall hibrit** | Heimdall `sabit` |
|---|---|---|---|
| KV/durum, 4K token | 151 MB | **16 MB** | 7 MB |
| KV/durum, 32K token | 1.21 GB | **104 MB** | 7 MB |
| KV/durum, 128K token | 4.83 GB | **406 MB** | 7 MB |
| Eğitim hesabı, T=8K | karesel × 12 katman | karesel × 3 katman + doğrusal × 9 | doğrusal |

Bellek sınırı KV cache olduğunda, aynı 16 GB'lık kartta ~12 kat daha fazla eşzamanlı kullanıcı ya da ~12 kat daha uzun bağlam demektir. Prefill parçalı yapılır (`--prefill-chunk`), bu yüzden uzun istemde karesel bellek patlaması olmaz.

### Mimari türleri
| `--arch` | Düzen | Kullanım |
|---|---|---|
| `hibrit` (varsayılan) | DDDA, global dikkat, NoPE | En iyi kalite. KV cache Transformer'ın ~1/12'si |
| `transformer` | A (her katman), RoPE, MHA | Karşılaştırma tabanı (aynı kod, aynı eğitim) |
| `sabit` | DDDA, 1024'lük pencere dikkati, RoPE | Global dikkat yok: çıkarım belleği bağlamdan tamamen bağımsız (O(1)) |

Düzen serbesttir: `--layout DDA`, `--layout DDDDDDDA` vb.

---

## 2. Kurulum

```bash
cd Heimdall
python -m venv .venv && source .venv/bin/activate

# AMD ROCm (RX 9070 XT: ROCm 6.4+)
pip install torch --index-url https://download.pytorch.org/whl/rocm6.4
# NVIDIA CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
pip install flash-linear-attention      # isteğe bağlı, önerilir: DeltaNet için Triton çekirdekleri
python -m pytest tests -q               # 5 test, CPU'da ~1 dk
```

**flash-linear-attention** CUDA'da ve ROCm'da (Triton) çalışır. Heimdall açılışta bu çekirdeği kendi PyTorch referansına karşı sınar. Sonuç uyuşmazsa ya da çekirdek hata verirse uyarı basar ve PyTorch yoluna döner, yani yanlış sonuç üretme riski yoktur. Log satırında hangisinin kullanıldığı yazar: `delta çekirdeği: ...`.

---

## 3. Eğitim

```bash
# duman testi (~9M, dakikalar)
python -m heimdall.train --preset mini --data ../stream_coder_100k.bin --tokens 20e6 --out runs/mini

# asıl model (~100M)
python -m heimdall.train --preset temel --data ../stream_coder_100k.bin veri/fineweb_edu_8k.bin:0.5 \
    --tokens 2e9 --out runs/temel

# devam (Ctrl+C güvenli: çıkarken kaydeder)
python -m heimdall.train --resume runs/temel/last.pt
```

| Ön ayar | Hibrit param | Transformer param | d × L | T | Adım başına token | Grad ckpt |
|---|---|---|---|---|---|---|
| `mini` | 9.3M | 9.4M | 256 × 8 | 512 | 32K | – |
| `kucuk` | 44.8M | 45.9M | 512 × 12 | 1024 | 128K | – |
| `temel` | 96.1M | 98.3M | 768 × 12 | 2048 | 256K | – |
| `buyuk` | 333M | 342M | 1024 × 24 | 2048 | 256K | ✓ |
| `moe` | 518M toplam / 115M aktif | – | 768 × 16, 32 uzman top-4 | 2048 | 256K | – |

**MoE (`--preset moe`):** ince taneli uzmanlar (32 uzman, token başına 4 + paylaşılan uzman), sigmoid yönlendirici, yardımcı kayıpsız yük dengeleme (DeepSeek-V3 tarzı). Token başına hesap `temel` ile aynı, kapasite ~5 kat. Aynı FLOP bütçesinde yoğun modelden belirgin düşük loss beklenir; DeepSeek ve OLMoE sonuçları bu yönde. Uzman ağırlıkları da Muon ile güncellenir.

Eğitim yığını:
- bf16 ve `torch.compile`.
- **Muon** (gizli matrisler; birleşik qkv/w12 parça parça ortogonalleştirilir) + **AdamW** (embedding, norm, kapılar).
- WSD öğrenme hızı çizelgesi.
- Gradyan biriktirme ve kırpma, arka plan veri yükleyici.
- ROCm bellek ayarları otomatik yapılır.

OOM olursa `--micro-batch` düşür ya da `--grad-ckpt` aç. Log `runs/<ad>/log.jsonl` dosyasına da yazılır.

Veri biçimi: uint16 token dosyası (`.bin`). Tokenizer: `04_TOKENIZERLAR/valerois_tokenizer_8k.json` (8192 kelime).

### Transformer ile karşılaştırma (kanıt prosedürü)
Aynı veri, aynı token, aynı optimizer. Transformer tabanının parametresi biraz daha fazla.
```bash
python -m heimdall.train --preset temel --arch hibrit      --data ../stream_coder_100k.bin --tokens 5e8 --out runs/hibrit
python -m heimdall.train --preset temel --arch transformer --data ../stream_coder_100k.bin --tokens 5e8 --out runs/transformer
python -m heimdall.compare runs/hibrit runs/transformer
```
Çıktıda son val loss, eşit tokende val loss ve medyan hız yer alır. Karar kuralı: hibritin val loss'u daha düşükse ve hızı en az ~%85 ise hibrit kazanır. Uzun bağlam avantajı ayrıca `--seq-len 8192` ile ölçülmeli; fark bu uzunlukta açılır.

---

## 4. Servis (SaaS)

### Paketle ve çalıştır
```bash
python -m heimdall.export --ckpt runs/temel/last.pt --out models/heimdall        # bf16 safetensors + config + tokenizer
HEIMDALL_API_KEYS="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')" \
  python -m heimdall.server --model models/heimdall --port 8000
```

### Docker
```bash
docker compose --profile rocm up -d --build     # AMD
docker compose --profile cuda up -d --build     # NVIDIA
```
- `models/heimdall` klasörü salt okunur bağlanır.
- Kullanım kaydı `logs/usage.jsonl` dosyasına yazılır.
- Anahtarlar `HEIMDALL_API_KEYS` ortam değişkeninden okunur.
- Konteyner root olmayan kullanıcıyla çalışır ve sağlık denetimi tanımlıdır.
- ROCm'da host'un render grubu gerekirse `RENDER_GID=$(getent group render | cut -d: -f3)` ile verilir.

### API (OpenAI uyumlu)
```bash
curl http://localhost:8000/v1/completions -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"prompt": "def quicksort(arr):", "max_tokens": 128, "temperature": 0.2, "stop": ["\n\n\n"]}'
```
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key=KEY)
for chunk in client.chat.completions.create(model="heimdall", stream=True,
                                            messages=[{"role": "user", "content": "Merhaba"}]):
    print(chunk.choices[0].delta.content or "", end="")
```

| Uç nokta | Açıklama |
|---|---|
| `POST /v1/completions` | `prompt`, `max_tokens`, `temperature`, `top_p`, `top_k`, `stop`, `stream` |
| `POST /v1/chat/completions` | `messages` (system/user/assistant) + yukarıdakiler; SSE akışı |
| `GET /v1/models` | model listesi |
| `GET /health` | motor iş parçacığı canlı mı, aktif/kuyruk sayısı (yük dengeleyici için) |
| `GET /metrics` | Prometheus: istek, token, gecikme, TTFT, aktif dizi, kuyruk, GPU belleği |

### İşletme özellikleri
| Özellik | Nasıl |
|---|---|
| **Sürekli batch'leme** | Yeni istek, çalışan batch'e bir sonraki adımda katılır; biten çıkar. Farklı uzunluktaki diziler aynı adımda ilerler. Test: tek tek üretimle birebir aynı çıktı |
| **Kimlik doğrulama** | `Authorization: Bearer`. Bellekte ve dosyada yalnız SHA-256 özeti tutulur: `--keys-file` satırları `sha256:<hex>` olabilir |
| **Sınırlar** | Anahtar başına dakikalık istek (`--rpm`), eşzamanlı istek (`--max-concurrent`), `--max-tokens-cap`, `--max-context` |
| **Faturalama** | `--usage-log`: her istek için anahtar kimliği, istem/üretim token sayısı, gecikme, TTFT (JSONL) |
| **İptal** | İstemci bağlantıyı koparırsa üretim bir sonraki adımda durur, GPU yuvası boşalır |
| **Hata yalıtımı** | Motor hatası aktif istekleri 500 ile bitirir, servis ayakta kalır (`/metrics` → `engine_errors_total`) |
| **Durdurma dizileri** | Akışta yakalanır. Durdurma dizisinin başı olabilecek son ek, eşleşme netleşene kadar gönderilmez |

Ölçekleme: GPU başına bir süreç çalıştır, önüne bir yük dengeleyici koy (nginx, Caddy, bulut LB). TLS yük dengeleyicide sonlanır. İstekler durumsuz olduğundan yapışkan oturum gerekmez.

---

## 5. Dosyalar
| Dosya | İçerik |
|---|---|
| `heimdall/config.py` | `HeimdallConfig`, mimari türleri, ön ayarlar |
| `heimdall/kernels.py` | RMSNorm, kısa conv, RoPE, pencere dikkati, delta kuralı (PyTorch + doğrulamalı Triton) |
| `heimdall/model.py` | Gated DeltaNet, dikkat, blok, `HeimdallLM`, konum etiketli cache (seç/birleştir) |
| `heimdall/optim.py` | Muon, AdamW grupları, WSD |
| `heimdall/data.py` | .bin karışımı, validation ayrımı, arka plan yükleyici |
| `heimdall/train.py` | Eğitim CLI |
| `heimdall/export.py`, `io.py` | Servis paketi (safetensors) |
| `heimdall/engine.py` | Sürekli batch'leme motoru |
| `heimdall/server.py` | FastAPI: OpenAI uyumlu API, kimlik, sınır, metrik, kullanım kaydı |
| `heimdall/compare.py` | Koşu karşılaştırma |
| `docker/`, `docker-compose.yml` | CUDA ve ROCm imajları |
| `tests/test_heimdall.py` | Delta kuralı, nedensellik, cache eşdeğerliği, sürekli batch'leme, API |

## 6. Sınırlar ve yol haritası
- **GPU'da ölçülmedi.** İlk iş: `mini` ile duman testi, sonra `temel` hibrit ve transformer karşılaştırması.
- **Tokenizer ve veri:** 8K'lık kod tokenizer'ı ve ~60M benzersiz token, genel amaçlı bir sohbet ürünü için yetmez. SaaS'ta sohbet için daha büyük bir tokenizer (32K), genel metin ve talimat verisi gerekir. Sohbet şablonu `server.py` → `Settings.chat_template` içindedir.
- **Tek GPU.** Çoklu GPU (DDP/FSDP) eğitimi yok.
- **Sayfalı KV cache yok.** Global dikkat cache'i dizi başına büyür; hibritte bu Transformer'ın ~1/12'si olduğu için pratikte sınır `--max-batch × --max-context`.
- **Nicemlenmiş servis yok** (int8/fp8).
- `flash-linear-attention` çağrı biçimi sürümler arasında değişebilir. Açılış testi uyumsuzluğu yakalar ve PyTorch yoluna döner; o durumda eğitim daha yavaş olur ama doğru kalır.

---

## 7. Bifrost dönüşümü: hazır Transformer → sabit bellekli model

Açık bir modelin (Llama, Qwen2/3, SmolLM) her dikkat katmanı şu hale getirilir. Öğretmenin q/k/v/o ağırlıkları aynen kalır; yalnız ~%2'lik yeni parametre eğitilir.
- **Birebir pencere:** son W token + ilk 4 çapa token için öğretmenin kendi dikkati. Kısa görevler tamamen pencereye sığar ve öğretmenle birebir aynı sonucu verir.
- **Gecikmeli yazılan delta hafıza:** her token pencereden çıktığı anda hafızaya yazılır; pencere ve hafıza çakışmaz.
- **Arşiv kafaları (α ≡ 1)** ve pencere içi log-normalizöre bakan **kütle kapısı**.
- Katman başına bellek sabittir: W + 4 anahtar ve bir d×d matris.

```bash
# 1) Dönüştür (model ve wikitext otomatik iner). Katman taklidi → uçtan uca damıtma.
python -m heimdall.donustur --model HuggingFaceTB/SmolLM2-135M --wikitext veri/wikitext \
    --out kosular/smol135 --window 64 --seq-len 2048 --batch 2 --steps-a 500 --steps-b 1000 --eval-len 4096 --eval-n 32
# 2) Benchmark (pip install lm-eval). --conv olmadan orijinal model ölçülür.
python -m heimdall.bifrost_eval lmeval --model HuggingFaceTB/SmolLM2-135M --conv kosular/smol135/donusum_param.pt \
    --tasks arc_easy,hellaswag,piqa
# 3) Uzun bağlam: şifre bulma testi (1.7B+ modellerde anlamlı)
python -m heimdall.bifrost_eval sifre --model ... --conv ... --lengths 4000 16000 64000
```

Doğrulananlar (CPU, `tests/test_bifrost.py`):
- Öğretmen uygulaması `transformers` çıktısıyla birebir aynı.
- Sabit bellekli token token üretim, tam paralel hesapla birebir aynı.
- Dönüştürülen katman belleği bağlam uzunluğundan bağımsız.

İlk ölçümler (SmolLM2-135M, pencere 64):

| | ppl (wikitext, 2K bağlam) | ARC-Easy (40 soru) |
|---|---|---|
| Öğretmen | 13.09 | %52.5 |
| Yalnız pencere | 23.66 | – |
| Dönüşüm, eğitimsiz | 24.52 | %52.5 |
| Dönüşüm, katman taklidi sonrası (300K token, CPU) | 21.25 | – |

**Büyük modeller (`--q4`):** Donuk öğretmen matrisleri 4-bit tutulur (32'lik gruplar, asimetrik, kırpma aramalı). En hassas matrisler (v_proj, down_proj) llama.cpp'nin Q4_K_M ayarı gibi 8-bit kalır; `--hassas-bit 4` hepsini 4-bit yapar. Ağırlıklar tensör tensör yüklenir, RAM'de tam kopya oluşmaz. Qwen3-0.6B'de ölçüm (wikitext): bf16 ppl 13.88 / 840 MB → karışık 4/8-bit 14.90 / 318 MB → tamamı 4-bit 16.26 / 262 MB. Büyük modellerde nicemleme kaybı küçülür. Tahmini VRAM: Qwen3-8B karışık ~7 GB, Qwen3-14B tamamı 4-bit ~10 GB.

```bash
python -m heimdall.donustur --model Qwen/Qwen3-8B --q4 --wikitext veri/wikitext --out kosular/qwen3_8b --window 1024 --seq-len 2048 --batch 1
```

Uzun bağlam kalitesi henüz öğretmene ulaşmadı; uçtan uca damıtma GPU'da koşulacak. Kısa benchmarklar için pencereyi 1024 yapmak skorları öğretmenle birebir aynı tutar.
