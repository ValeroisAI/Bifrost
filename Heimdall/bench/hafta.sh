#!/bin/bash
# Hafta boyu gözetimsiz eğitim: veri hazırla + boyut ölç (paralel) → modeli seç → termine yetişecek kadar eğit → çökerse sürdür.
#   nohup systemd-inhibit --what=idle:sleep --why="bit egitim" bash bench/hafta.sh > hafta.log 2>&1 &   ;   tail -f hafta.log
# Ortam: DEADLINE="2026-10-23 12:00" (eğitimin biteceği yerel saat), OUT=runs/hafta, DRY=1 (yalnız karar mantığını dene)
# Kesinti / yeniden başlatma sonrası AYNI komutu tekrar çalıştır: seçim kaydedilir (secim.env), eğitim son kayıttan sürer.
cd "$(dirname "$0")/.."
DEADLINE=${DEADLINE:-"2026-10-23 12:00"}; OUT=${OUT:-runs/hafta}; DRY=${DRY:-0}
mkdir -p $OUT veri runs
SEC=$OUT/secim.env
log() { echo "[$(date '+%a %d %H:%M')] $*"; }
BINS="veri/fineweb.bin veri/fw_b.bin veri/kod.bin veri/kod_b.bin veri/cosmo.bin"

if [ ! -f $SEC ]; then
  # --- 1) veri arka planda (CPU), 2) boyut ölçümü (GPU) aynı anda
  if [ "$DRY" = 1 ]; then VERI_PID=""; else bash bench/hafta_veri.sh > veri/hazirlik.log 2>&1 & VERI_PID=$!; fi
  log "veri hazırlığı başladı (pid ${VERI_PID:-yok}); boyut ölçümü başlıyor"
  declare -A TPS
  for v in "buyuk_mb4:--preset buyuk --no-grad-ckpt --micro-batch 4" "buyuk_mb2:--preset buyuk --no-grad-ckpt --micro-batch 2" \
           "buyuk_mb4ckpt:--preset buyuk --grad-ckpt --micro-batch 4"; do
    n=${v%%:*}; a=${v#*:}; TPS[$n]=0
    if [ "$DRY" = 1 ]; then TPS[$n]=${FAKE_TPS:-13000}; continue; fi
    rm -rf runs/boyut_$n
    python -m heimdall.train --arch bit --bits 1 --data veri/fineweb.bin --tokens 3.2e6 --eval-every 100000 --save-every 0 \
        --log-every 3 --out runs/boyut_$n $a > runs/boyut_$n.log 2>&1
    TPS[$n]=$(python - <<PY
import json
r = [json.loads(l) for l in open("runs/boyut_$n/log.jsonl")] if __import__("os").path.exists("runs/boyut_$n/log.jsonl") else []
r = [x["tok_s"] for x in r if "tok_s" in x]
print(r[-1] if r else 0)
PY
)
    log "boyut ölçümü $n: ${TPS[$n]} tok/s"
  done
  [ -n "$VERI_PID" ] && { log "veri hazırlığı bekleniyor"; wait $VERI_PID; }
  # --- model seçimi: en hızlı çalışan büyük model; ≥9K tok/s değilse temel (98M)
  BEST=""; BT=0
  for n in "${!TPS[@]}"; do if python -c "import sys; sys.exit(0 if float('${TPS[$n]}') > float('$BT') else 1)"; then BEST=$n; BT=${TPS[$n]}; fi; done
  if python -c "import sys; sys.exit(0 if float('$BT') >= 9000 else 1)"; then
    case $BEST in
      buyuk_mb4)    MARGS="--preset buyuk --no-grad-ckpt --micro-batch 4";;
      buyuk_mb2)    MARGS="--preset buyuk --no-grad-ckpt --micro-batch 2";;
      buyuk_mb4ckpt) MARGS="--preset buyuk --grad-ckpt --micro-batch 4";;
    esac; TP=$BT; MODEL=buyuk
  else MARGS="--preset temel --micro-batch 8"; TP=47000; MODEL=temel; fi
  # --- veri listesi (dosya boyutuyla orantılı pay) ve token bütçesi
  DATA=""; for b in $BINS; do [ -f $b ] && DATA="$DATA $b"; done
  TOK=$(DEADLINE="$DEADLINE" TP="$TP" DATA="$DATA" python - <<'PY'
import os, time, datetime
dl = datetime.datetime.strptime(os.environ["DEADLINE"], "%Y-%m-%d %H:%M").timestamp()
rem = dl - time.time()
tp = float(os.environ["TP"])
tot = sum(os.path.getsize(f) // 2 for f in os.environ["DATA"].split())
tok = min(tp * rem * 0.92, 2.5 * tot) if os.environ.get("DRY") != "x" else 0
print(int(max(tok, 1e8)), end="")
PY
)
  { echo "MODEL=$MODEL"; echo "MARGS=\"$MARGS\""; echo "DATA=\"$DATA\""; echo "TOK=$TOK"; } > $SEC
  log "SEÇİM: $MODEL ($MARGS) | hız ~${TP} tok/s | veri:$DATA | ${TOK} token (termin $DEADLINE)"
fi
source $SEC
log "eğitim: $MODEL | $TOK token | veri: $DATA"
[ "$DRY" = 1 ] && { cat $SEC; exit 0; }

ARGS="--arch bit --bits 1 --data $DATA --tokens $TOK $MARGS --eval-every 500 --save-every 200 --log-every 50 --out $OUT/model"
for i in $(seq 1 60); do
  if [ -f $OUT/model/last.pt ]; then CMD="--resume $OUT/model/last.pt"; else CMD="$ARGS"; fi
  python -m heimdall.train $CMD 2>&1 | grep --line-buffered -v amdgpu.ids
  rc=${PIPESTATUS[0]}
  if [ "$rc" = 0 ]; then log "EĞİTİM BİTTİ"; break; fi
  log "kesildi (rc=$rc); 30 sn sonra son kayıttan sürdürülüyor ($i. kez)"; sleep 30
done
python bench/uret.py --ckpt $OUT/model/last.pt > $OUT/ornekler.txt 2>&1 && log "örnekler: $OUT/ornekler.txt"
