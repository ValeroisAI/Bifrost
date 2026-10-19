#!/bin/bash
# Hedef: N parametreli bit model >= N/2 parametreli yoğun model.  Ayrıca hız: torch.compile ve 1-bit çekirdekler.
#   bash bench/hedef.sh hiz      # ~5 dk : çekirdek benchmark + torch.compile denemesi
#   bash bench/hedef.sh kucuk    # ~12 dk: 6M bit (üçlü ve ikili FFN) vs 3M yoğun, iki tohum
#   bash bench/hedef.sh orta     # 46M bit vs 23M yoğun (torch.compile ile ~10 dk)
#   bash bench/hedef.sh temel    # 98M bit vs 52M yoğun, 40M token (compile ile ~30 dk)
cd "$(dirname "$0")/.."
NC=${NC:-}   # torch.compile varsayılan açık; sorun olursa:  NC=--no-compile bash ...
run() { n=$1; shift; echo ">>> $n"; python -m heimdall.train "$@" --out runs/hedef_$n 2>&1 | grep -v amdgpu.ids | grep -E "param|tok/s|val\]|Error|error" | tail -3; }
case "$1" in
hiz)
  python bench/bitkernel.py 2>&1 | grep -v amdgpu.ids
  C="--preset mini --data ../stream_coder_100k.bin --tokens 3e6 --micro-batch 16 --eval-every 100000 --log-every 40 --save-every 0 --d-model 192"
  run derlemesiz_yogun $C --no-compile
  run derlemeli_yogun  $C
  run derlemesiz_bit   $C --no-compile --arch bit
  run derlemeli_bit    $C --arch bit ;;
kucuk)
  C="--preset mini --data ../stream_coder_100k.bin --tokens 10e6 --micro-batch 16 $NC --eval-every 100000 --log-every 50 --save-every 0"
  for s in 0 1; do
    run yogun_yari_d112_s$s $C --seed $s --d-model 112
    run bit_uclu_d192_s$s   $C --seed $s --d-model 192 --arch bit
    run bit_ikili_d192_s$s  $C --seed $s --d-model 192 --arch bit --bits 1
  done
  python -m heimdall.compare runs/hedef_yogun_yari_d112_s0 runs/hedef_yogun_yari_d112_s1 runs/hedef_bit_uclu_d192_s0 \
      runs/hedef_bit_uclu_d192_s1 runs/hedef_bit_ikili_d192_s0 runs/hedef_bit_ikili_d192_s1 ;;
orta)
  C="--preset kucuk --data ../stream_coder_100k.bin --tokens ${TOK:-20e6} --micro-batch ${MB:-8} $NC --eval-every 100000 --log-every 50 --save-every 0"
  run yogun_yari_d336 $C --d-model 336
  run bit_uclu_d512   $C --d-model 512 --arch bit
  python -m heimdall.compare runs/hedef_yogun_yari_d336 runs/hedef_bit_uclu_d512 ;;
temel)
  C="--preset temel --data ../stream_coder_100k.bin --tokens ${TOK:-40e6} --micro-batch ${MB:-8} $NC --eval-every 100000 --log-every 50 --save-every 0"
  run T_yogun_yari_d512 $C --d-model 512
  run T_bit_d768        $C --arch bit
  python -m heimdall.compare runs/hedef_T_yogun_yari_d512 runs/hedef_T_bit_d768 ;;
*) echo "kullanım: bash bench/hedef.sh hiz|kucuk|orta|temel" ;;
esac
