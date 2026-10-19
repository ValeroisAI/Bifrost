#!/bin/bash
# Hangi model boyutu / mikro-batch kartta hangi hızda ve bellekle koşuyor?  (~10-12 dk, derleme dahil)
#   bash bench/boyut.sh          sonuç satırları: tok/s (kararlı hız) ve bellek (GB). OOM yazan seçenek sığmıyor demektir.
cd "$(dirname "$0")/.."
D=${DATA:-veri/fineweb.bin}
t() { n=$1; shift; echo ">>> $n"
  python -m heimdall.train --arch bit --bits 1 --data $D --tokens ${TOK:-3.2e6} --eval-every 100000 --save-every 0 \
      --log-every 3 --out runs/boyut_$n "$@" 2>&1 | grep --line-buffered -v amdgpu.ids \
      | grep -E "param|tok/s|out of memory|OutOfMemory|Error" | awk 'NR==1{print} {last=$0} END{print last}'; }
t temel_mb8       --preset temel --micro-batch 8
t buyuk_mb4       --preset buyuk --no-grad-ckpt --micro-batch 4
t buyuk_mb2       --preset buyuk --no-grad-ckpt --micro-batch 2
t buyuk_mb4_ckpt  --preset buyuk --grad-ckpt --micro-batch 4
