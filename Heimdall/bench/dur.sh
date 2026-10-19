#!/bin/bash
# Eğitimi temiz durdur: o anki adımda kaydeder (kayıp yok), GPU belleği boşalır.   bash bench/dur.sh
pkill -f "bash bench/[h]afta.sh"              # önce döngüyü öldür (yoksa 'bitti' sanıp örnek üretmeye geçer)
pkill -INT -f "heimdall.[t]rain"              # train.py KeyboardInterrupt → son adımı kaydeder
for i in $(seq 1 60); do pgrep -f "heimdall.[t]rain" >/dev/null || break; sleep 2; done
tail -n 3 hafta.log 2>/dev/null; pgrep -f "heimdall.[t]rain" >/dev/null && echo "HÂLÂ ÇALIŞIYOR, kayıt bitmemiş olabilir; biraz bekle" || echo "DURDU. GPU boş; oyun açabilirsin."
