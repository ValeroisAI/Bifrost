#!/bin/bash
# Bit tasarım matrisi (GPU, ~30-40 dk): aynı veri, aynı token, yedi varyant.   Kullanım:  bash bench/bit_matrix.sh [token]
# Sonunda son val loss tablosu basılır. Bellek hatası olursa:  MB=8 bash bench/bit_matrix.sh
cd "$(dirname "$0")/.."
NC=${NC:-}   # torch.compile varsayılan açık; sorun olursa:  NC=--no-compile bash ...
TOK=${1:-10e6}; MB=${MB:-16}
C="--preset mini --data ../stream_coder_100k.bin --tokens $TOK --micro-batch $MB $NC --eval-every 100000 --log-every 50 --save-every 0 --flip-orani 5e-3 --d-model 192"
run() { n=$1; shift; echo ">>> $n"; python -m heimdall.train $C "$@" --out runs/bit_$n 2>&1 | grep -v amdgpu.ids | grep -E "param|val\]|Error|error" | tail -3; }
run A_yogun
run B_uclu       --bits 2
run C_ikili      --bits 1
run D_uclu_d240  --bits 2 --d-model 240
run E_uclu_r16   --bits 2 --bit-rank 16
run F_uclu_ffn   --bits 2 --bit-scope ffn
run G_ffn_r16    --bits 2 --bit-scope ffn --bit-rank 16
python -m heimdall.compare runs/bit_A_yogun runs/bit_B_uclu runs/bit_C_ikili runs/bit_D_uclu_d240 runs/bit_E_uclu_r16 runs/bit_F_uclu_ffn runs/bit_G_ffn_r16
