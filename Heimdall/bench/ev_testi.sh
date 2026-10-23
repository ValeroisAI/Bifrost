#!/bin/bash
# Eve geçince TEK komut: eğitimi durdur → testler → plan → BitNet duman testi (GPU, kısa) → sonucu yaz. ~15-25 dk (ilk seferde indirme dahil).
#   bash bench/ev_testi.sh          bittiğinde: bash bench/devam.sh  (uzun eğitime dön)  ya da tam dönüşüm: bash bench/bitnet_donustur.sh
cd "$(dirname "$0")/.."
[ -f .venv101/bin/activate ] && source .venv101/bin/activate
LOG=runs/ev_testi.log; mkdir -p runs veri
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }
: > $LOG
say "1/5 uzun eğitim durduruluyor (kayıtlı adımdan devam edilebilir)"
bash bench/dur.sh 2>&1 | tee -a $LOG
say "2/5 birim testleri (CPU)"
for t in test_heimdall test_bifrost test_bit test_bitkernel test_planla test_arayuz; do
  python tests/$t.py > /tmp/$t.out 2>&1 && say "  OK   $t" || { say "  FAIL $t"; tail -5 /tmp/$t.out | tee -a $LOG; }
done
say "3/5 planlayıcı (kart: $(python - <<'P'
import torch; print(f"{torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GB {torch.cuda.get_device_name(0)}")
P
))"
GB=$(python -c "import torch; print(round(torch.cuda.get_device_properties(0).total_memory/2**30))")
python -m heimdall.planla --model microsoft/bitnet-b1.58-2B-4T-bf16 --gpu $GB --seq-len 512 --batch 1 --grad-ckpt 2>&1 | grep -v Warn | tee -a $LOG
say "4/5 BitNet-2B duman testi: 10+10 adım, dizi 512, batch 1 (hız ve TEPE GPU belleği ölçülür)"
STEPS=10 SEQ=512 OUT=runs/bitnet_duman bash bench/bitnet_donustur.sh 2>&1 | grep -E "Öğretmen|ogretmen|yalniz|donusum|aşama|Kaydedildi|Error|error|out of memory|Traceback" | tee -a $LOG
say "5/5 bitti. Sonuç: runs/ev_testi.log  — bana 'tepe GPU' ve 'tok/s' satırlarını yapıştır (planlayıcıyı kalibre ederim)."
say "Tam dönüşüm: bash bench/bitnet_donustur.sh   |   Uzun eğitime dön: bash bench/devam.sh"
