#!/bin/bash
# İkinci tur (GPU). Kullanım:
#   bash bench/bit_matrix2.sh gurultu   # ~10 dk: tohum gürültüsü + ikili/rank varyantları (küçük model)
#   bash bench/bit_matrix2.sh olcek     # ~35 dk: 45M parametrede yoğun hibrit ile --arch bit  (fark ölçekle nasıl değişiyor?)
cd "$(dirname "$0")/.."
NC=${NC:-}   # torch.compile varsayılan açık; sorun olursa:  NC=--no-compile bash ...
run() { n=$1; shift; echo ">>> $n"; python -m heimdall.train "$@" --out runs/bit2_$n 2>&1 | grep -v amdgpu.ids | grep -E "param|val\]|Error|error" | tail -3; }
case "$1" in
gurultu)
  C="--preset mini --data ../stream_coder_100k.bin --tokens 10e6 --micro-batch ${MB:-16} $NC --eval-every 100000 --log-every 50 --save-every 0 --flip-orani 5e-3 --d-model 192"
  run A_yogun_s1       $C --seed 1
  run G_ffn_r16_s1     $C --seed 1 --bits 2 --bit-scope ffn --bit-rank 16
  run H_ikili_ffn_r16  $C --bits 1 --bit-scope ffn --bit-rank 16
  run I_uclu_ffn_r32   $C --bits 2 --bit-scope ffn --bit-rank 32
  python -m heimdall.compare runs/bit_A_yogun runs/bit2_A_yogun_s1 runs/bit_G_ffn_r16 runs/bit2_G_ffn_r16_s1 runs/bit2_H_ikili_ffn_r16 runs/bit2_I_uclu_ffn_r32 ;;
olcek)
  C="--preset kucuk --data ../stream_coder_100k.bin --tokens ${TOK:-20e6} --micro-batch ${MB:-8} $NC --eval-every 100000 --log-every 50 --save-every 0"
  run K_hibrit $C --arch hibrit
  run K_bit    $C --arch bit
  python -m heimdall.compare runs/bit2_K_hibrit runs/bit2_K_bit ;;
*) echo "kullanım: bash bench/bit_matrix2.sh gurultu|olcek" ;;
esac
