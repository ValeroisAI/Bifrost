#!/bin/bash
# Durdurulan eğitimi son kayıttan sürdür (seçim ve veri aynı kalır).   bash bench/devam.sh
cd "$(dirname "$0")/.."
[ -f .venv101/bin/activate ] && source .venv101/bin/activate
pgrep -f "heimdall.[t]rain" >/dev/null && { echo "Eğitim zaten çalışıyor."; exit 1; }
nohup systemd-inhibit --what=idle:sleep --why="bit egitim" bash bench/hafta.sh >> hafta.log 2>&1 &
sleep 2; echo "Sürdürüldü. İzle: tail -f hafta.log"
