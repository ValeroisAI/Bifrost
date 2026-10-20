"""
arayuz.py — Yerel web arayüzü: modeli seç → "bu kartta olur mu" planı → dönüşümü başlat / durdur → log izle.

    python -m heimdall.arayuz            # http://127.0.0.1:8765   (yalnız bu bilgisayardan erişilir; kimlik doğrulama YOK)

Başlat düğmesi `python -m heimdall.donustur ...` çalıştırır (Bifrost dönüşümü). Uzun eğitim (heimdall.train) çalışırken
başlatmayı reddeder: ikisi aynı GPU'yu paylaşmasın. Planlayıcı tahminleri için bkz. planla.py (aktivasyon belleği tahmindir).
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import planla

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "arayuz"
NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+(/[A-Za-z0-9_.\-]+)?$")
app = FastAPI(title="Heimdall / Bifrost")
state = {"proc": None, "out": None}


class Req(BaseModel):
    model: str
    gpu: str = "16"
    ram: float = 0.0
    window: int = 64
    sinks: int = 4
    seq_len: int = 1024
    batch: int = 1
    steps_a: int = 200
    steps_b: int = 200
    q4: bool = False
    grad_ckpt: bool = True


def _check_name(name: str) -> str:
    if not NAME_RE.match(name) or ".." in name:
        raise HTTPException(400, "model adı: 'sahip/ad' (harf, rakam, . _ -) olmalı")
    return name


def _running(pattern: str) -> bool:
    return subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0


def _alive() -> bool:
    p = state["proc"]
    return p is not None and p.poll() is None


@app.post("/api/plan")
def plan(r: Req) -> dict:
    _check_name(r.model)
    gpu = float(planla.GPUS.get(r.gpu.lower(), r.gpu))
    try:
        hf = planla.load_config(r.model)
    except Exception as e:  # ağ ya da yanlış ad
        raise HTTPException(404, f"config.json alınamadı: {e}")
    res = planla.analyze(hf, window=r.window, sinks=r.sinks, seq_len=r.seq_len, batch=r.batch, q4=r.q4,
                         grad_ckpt=r.grad_ckpt)
    f = planla.fit(res, gpu, r.ram)
    return {"plan": res, "fit": f, "text": planla.render(r.model, res, f, gpu), "gpu_gb": gpu}


@app.post("/api/start")
def start(r: Req) -> dict:
    _check_name(r.model)
    if _alive():
        raise HTTPException(409, "bir dönüşüm zaten çalışıyor")
    if _running(r"heimdall.[t]rain"):
        raise HTTPException(409, "uzun eğitim çalışıyor; GPU'yu paylaşmasın diye önce: bash bench/dur.sh")
    hf = planla.load_config(r.model)
    info = planla.analyze(hf, window=r.window, sinks=r.sinks, seq_len=r.seq_len, batch=r.batch, q4=r.q4,
                          grad_ckpt=r.grad_ckpt)
    if not info["supported"]:
        raise HTTPException(422, "desteklenmiyor: " + "; ".join(info["notes"]))
    eos = (hf.get("text_config") or hf).get("eos_token_id", 0)
    sep = eos[0] if isinstance(eos, list) and eos else (eos if isinstance(eos, int) else 0)
    slug = r.model.replace("/", "__")
    out = RUNS / slug
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "heimdall.donustur", "--model", r.model, "--wikitext", str(ROOT / "veri" / f"wikitext_{slug}"),
           "--sep", str(sep), "--out", str(out), "--window", str(r.window), "--sinks", str(r.sinks),
           "--seq-len", str(r.seq_len), "--batch", str(r.batch), "--steps-a", str(r.steps_a), "--steps-b", str(r.steps_b)]
    if r.q4:
        cmd.append("--q4")
    if r.grad_ckpt:
        cmd.append("--grad-ckpt")
    env = dict(os.environ, PYTORCH_HIP_ALLOC_CONF="expandable_segments:True", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    log = open(out / "log.txt", "w")
    state["proc"] = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    state["out"] = out
    return {"started": True, "cmd": " ".join(cmd), "out": str(out)}


@app.post("/api/stop")
def stop() -> dict:
    if not _alive():
        return {"stopped": False}
    os.killpg(state["proc"].pid, signal.SIGTERM)
    return {"stopped": True}


@app.get("/api/status")
def status() -> dict:
    out = state["out"]
    tail, result = "", None
    if out and (out / "log.txt").exists():
        tail = "".join((out / "log.txt").read_text(errors="replace").splitlines(keepends=True)[-40:])
        if (out / "sonuc.json").exists():
            result = json.loads((out / "sonuc.json").read_text())
    p = state["proc"]
    return {"running": _alive(), "returncode": None if p is None or p.poll() is None else p.poll(),
            "out": str(out) if out else None, "log": tail, "result": result}


PAGE = """<!doctype html><html lang="tr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Heimdall / Bifrost</title>
<style>
:root{--bg:#fff;--fg:#14181f;--mut:#5b6472;--line:#d8dde5;--card:#f5f7fa;--ok:#1a7f37;--warn:#9a6700;--bad:#cf222e;--acc:#1f6feb}
@media(prefers-color-scheme:dark){:root{--bg:#0f141b;--fg:#e6eaf0;--mut:#99a3b3;--line:#2a3340;--card:#161d27;--ok:#3fb950;--warn:#d29922;--bad:#f85149;--acc:#58a6ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
main{max-width:880px;margin:0 auto;padding:20px 16px 60px}h1{font-size:20px;margin:0 0 4px}.sub{color:var(--mut);margin:0 0 18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:12px 0}
label{display:block;font-size:13px;color:var(--mut);margin:8px 0 2px}input[type=text],input[type=number],select{width:100%;padding:7px 9px;border:1px solid var(--line);border-radius:7px;background:var(--bg);color:var(--fg);font:inherit}
.row{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}.chk{display:flex;gap:8px;align-items:center;margin-top:10px}.chk input{width:auto}
button{padding:8px 14px;border-radius:7px;border:1px solid var(--line);background:var(--acc);color:#fff;font:inherit;cursor:pointer;margin:12px 8px 0 0}button.alt{background:var(--card);color:var(--fg)}
pre{white-space:pre-wrap;word-break:break-word;background:var(--bg);border:1px solid var(--line);border-radius:7px;padding:10px;font:12.5px/1.45 ui-monospace,monospace;margin:8px 0 0}
.pill{display:inline-block;padding:2px 9px;border-radius:99px;font-size:13px;border:1px solid currentColor}.yeşil{color:var(--ok)}.sarı{color:var(--warn)}.kırmızı{color:var(--bad)}
.note{color:var(--mut);font-size:13px}
</style><main>
<h1>Heimdall / Bifrost</h1><p class="sub">Bir modeli sabit bellekli yapıya dönüştürmeden önce: bu kartta olur mu?</p>
<div class="card">
<label>Model (Hugging Face adı)</label><input id="model" type="text" value="microsoft/bitnet-b1.58-2B-4T-bf16">
<div class="row">
<div><label>Kart (GB ya da ad: 9070xt, mi300x…)</label><input id="gpu" type="text" value="16"></div>
<div><label>Sistem RAM (GB, isteğe bağlı)</label><input id="ram" type="number" value="0"></div>
<div><label>Dizi uzunluğu</label><input id="seq_len" type="number" value="1024"></div>
<div><label>Batch</label><input id="batch" type="number" value="1"></div>
<div><label>Aşama A adım</label><input id="steps_a" type="number" value="200"></div>
<div><label>Aşama B adım</label><input id="steps_b" type="number" value="200"></div>
</div>
<div class="chk"><input id="q4" type="checkbox"><span>Donuk öğretmen 4-bit (büyük modeller için)</span></div>
<div class="chk"><input id="grad_ckpt" type="checkbox" checked><span>Gradient checkpointing (bellek az, hız ~%30 düşer)</span></div>
<button onclick="plan()">Planla</button><button class="alt" onclick="start()">Dönüşümü başlat</button><button class="alt" onclick="stopRun()">Durdur</button>
<p class="note">Parametre ve KV sayıları kesin; eğitim aktivasyon belleği TAHMİNDİR (±%40). Uzun eğitim çalışırken başlatma reddedilir.</p>
</div>
<div class="card" id="plancard" hidden><div id="verdict"></div><pre id="plantext"></pre></div>
<div class="card" id="runcard" hidden><div id="runstate"></div><pre id="log"></pre></div>
<script>
const $=id=>document.getElementById(id);
const body=()=>({model:$('model').value.trim(),gpu:$('gpu').value.trim(),ram:+$('ram').value,seq_len:+$('seq_len').value,batch:+$('batch').value,
 steps_a:+$('steps_a').value,steps_b:+$('steps_b').value,q4:$('q4').checked,grad_ckpt:$('grad_ckpt').checked});
async function call(path,method,b){const r=await fetch(path,{method,headers:{'Content-Type':'application/json'},body:b?JSON.stringify(b):undefined});
 const j=await r.json();if(!r.ok)throw new Error(j.detail||r.statusText);return j}
async function plan(){try{const j=await call('/api/plan','POST',body());$('plancard').hidden=false;
 const f=j.fit.donusum_egitimi[1],lab={yeşil:'Sığar',sarı:'Sınırda',kırmızı:'Sığmaz'}[f];
 $('verdict').innerHTML='<span class="pill '+f+'">'+lab+'</span> '+(j.plan.supported?'':'<span class="pill kırmızı">bu kodla desteklenmiyor</span>');
 $('plantext').textContent=j.text}catch(e){$('plancard').hidden=false;$('verdict').textContent='';$('plantext').textContent='Hata: '+e.message}}
async function start(){try{await call('/api/start','POST',body());poll()}catch(e){$('runcard').hidden=false;$('runstate').textContent='';$('log').textContent='Hata: '+e.message}}
async function stopRun(){try{await call('/api/stop','POST')}catch(e){}}
async function poll(){try{const j=await call('/api/status','GET');$('runcard').hidden=!j.out;
 $('runstate').innerHTML=j.running?'<span class="pill sarı">çalışıyor</span>':(j.returncode===0?'<span class="pill yeşil">bitti</span>':(j.returncode===null?'':'<span class="pill kırmızı">durdu (kod '+j.returncode+')</span>'));
 $('log').textContent=j.log;if(j.running)setTimeout(poll,3000)}catch(e){}}
poll();
</script></main></html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE


def main() -> None:
    import uvicorn
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8765)
    a = p.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
