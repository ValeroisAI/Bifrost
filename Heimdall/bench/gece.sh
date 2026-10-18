#!/bin/bash
# Gece koşusu: veri hazırla (yoksa) → eğit → çökerse son kayıttan otomatik sürdür.
# Kullanım (uyku modunu engelleyerek, terminal kapansa da sürer):
#   nohup systemd-inhibit --what=idle:sleep --why="bit egitim" bash bench/gece.sh > gece.log 2>&1 &   ;   tail -f gece.log
# Ayarlar (ortam değişkeni): TOK=800e6  BITS=1  MB=8  PRESET=temel  OUT=runs/gece
cd "$(dirname "$0")/.."
D=veri; mkdir -p $D
TOK=${TOK:-800e6}; BITS=${BITS:-1}; MB=${MB:-8}; PRESET=${PRESET:-temel}; OUT=${OUT:-runs/gece}

[ -f $D/fineweb.bin ] || python -m heimdall.hazirla --repo HuggingFaceFW/fineweb-edu --dosya sample/10BT/000_00000.parquet \
    --kolon text --out $D/fineweb.bin --max-token 500e6 || exit 1
[ -f $D/kod.bin ] || python -m heimdall.hazirla --repo codeparrot/github-code-clean \
    --dosya data/train-00000-of-00880.parquet data/train-00001-of-00880.parquet \
    --kolon code --out $D/kod.bin --max-token 240e6 || exit 1

ARGS="--preset $PRESET --arch bit --bits $BITS --data $D/fineweb.bin:0.60 $D/kod.bin:0.32 ../stream_coder_100k.bin:0.08 \
      --tokens $TOK --micro-batch $MB --eval-every 250 --save-every 250 --log-every 20 --out $OUT"
echo "=== başlangıç: $(date) | $ARGS"
python -m heimdall.train $ARGS 2>&1 | grep --line-buffered -v amdgpu.ids
for i in $(seq 1 30); do                      # çökme / NaN olursa son iyi kayıttan sürdür
    [ "${PIPESTATUS[0]}" = "0" ] && { echo "=== bitti: $(date)"; exit 0; }
    [ -f $OUT/last.pt ] || { echo "kayıt yok, sürdürülemez"; exit 1; }
    echo "=== kesildi ($(date)); $i. kez sürdürülüyor"; sleep 20
    python -m heimdall.train --resume $OUT/last.pt 2>&1 | grep --line-buffered -v amdgpu.ids
done
