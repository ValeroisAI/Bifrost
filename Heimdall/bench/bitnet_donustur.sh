#!/bin/bash
# Microsoft BitNet b1.58-2B-4T'yi (üçlü ağırlık, 4T token) Bifrost ile sabit bellekli modele dönüştürür.
# FFN ve projeksiyonlar 1.58-bit kalır; yalnız dikkat → pencere + delta hafıza (yeni param ~%2). GPU boşken çalıştır.
#   bash bench/bitnet_donustur.sh            (BATCH=1 ile daha az bellek)
cd "$(dirname "$0")/.."
[ -f .venv101/bin/activate ] && source .venv101/bin/activate
pgrep -f "heimdall.[t]rain" >/dev/null && { echo "Eğitim çalışıyor; GPU'yu paylaşmasın diye önce: bash bench/dur.sh"; exit 1; }
mkdir -p runs
python -m heimdall.donustur --model microsoft/bitnet-b1.58-2B-4T-bf16 --wikitext veri/wikitext_bitnet --sep 128001 \
    --out runs/bitnet_donusum --window 64 --sinks 4 --seq-len 1024 --batch ${BATCH:-2} --steps-a 200 --steps-b 200 \
    2>&1 | grep --line-buffered -v amdgpu.ids | tee runs/bitnet_donusum.log
