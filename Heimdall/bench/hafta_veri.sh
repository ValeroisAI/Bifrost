#!/bin/bash
# Hafta boyu eğitim için veri (≈4.5-5B token), üç kaynak paralel hazırlanır (~1 saat). Var olan dosyalar atlanır.
#   FineWeb-Edu: fineweb.bin (gece koşusundan, 500M) + fw_b.bin (shard 1-4, 2.0B)
#   Kod:         kod.bin (240M) + kod_b.bin (GitHub kod shard 2-9, 0.9B)
#   Ders kitabı: cosmo.bin (Cosmopedia web_samples_v2, ilk 12 shard ≈ 0.95B) — sentetik, Apache-2.0
cd "$(dirname "$0")/.."
mkdir -p veri
H=python
FW=""; for i in 001 002 003 004; do FW="$FW sample/10BT/${i}_00000.parquet"; done
KD=""; for i in 2 3 4 5 6 7 8 9; do KD="$KD data/train-0000$i-of-00880.parquet"; done
CS=""; for i in 00 01 02 03 04 05 06 07 08 09 10 11; do CS="$CS data/web_samples_v2/train-000$i-of-00118.parquet"; done
[ -f veri/fw_b.bin ]   || $H -m heimdall.hazirla --repo HuggingFaceFW/fineweb-edu --dosya $FW --kolon text --out veri/fw_b.bin --max-token 2.0e9 > veri/fw_b.log 2>&1 &
[ -f veri/kod_b.bin ]  || $H -m heimdall.hazirla --repo codeparrot/github-code-clean --dosya $KD --kolon code --out veri/kod_b.bin --max-token 0.9e9 > veri/kod_b.log 2>&1 &
[ -f veri/cosmo.bin ]  || $H -m heimdall.hazirla --repo HuggingFaceTB/cosmopedia --dosya $CS --kolon text --out veri/cosmo.bin --max-token 0.95e9 > veri/cosmo.log 2>&1 &
wait
ls -la veri/*.bin; for f in veri/fw_b.log veri/kod_b.log veri/cosmo.log; do tail -2 $f | head -1; done
