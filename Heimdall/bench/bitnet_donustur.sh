#!/bin/bash
# Microsoft BitNet b1.58-2B-4T'yi (üçlü ağırlık, 4T token) Bifrost ile sabit bellekli modele dönüştürür.
# FFN ve projeksiyonlar 1.58-bit kalır; yalnız dikkat → pencere + delta hafıza (yeni param ~%1). GPU boşken çalıştır.
#   bash bench/bitnet_donustur.sh                 varsayılan: batch 1, katman başına yeniden hesap (grad-ckpt), 200+200 adım
#   BATCH=2 CKPT= bash bench/bitnet_donustur.sh   (daha hızlı ama ~26 GB ister: 16 GB karta SIĞMAZ, planlayıcı öyle diyor)
#   STEPS=20 SEQ=512 bash bench/bitnet_donustur.sh   (kısa duman testi)
cd "$(dirname "$0")/.."
[ -f .venv101/bin/activate ] && source .venv101/bin/activate
pgrep -f "heimdall.[t]rain" >/dev/null && { echo "Eğitim çalışıyor; GPU'yu paylaşmasın diye önce: bash bench/dur.sh"; exit 1; }
mkdir -p runs
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # parçalanma ↓
python -m heimdall.donustur --model microsoft/bitnet-b1.58-2B-4T-bf16 --wikitext veri/wikitext_bitnet --sep 128001 \
    --out ${OUT:-runs/bitnet_donusum} --window 64 --sinks 4 --seq-len ${SEQ:-1024} --batch ${BATCH:-1} \
    --steps-a ${STEPS:-200} --steps-b ${STEPS:-200} ${CKPT---grad-ckpt} \
    2>&1 | grep --line-buffered -v amdgpu.ids | tee ${OUT:-runs/bitnet_donusum}.log
