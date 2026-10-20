"""Arayüz API'si: plan, giriş doğrulama, başlat reddi (eğitim çalışırken / desteklenmeyen model), komut kurulumu."""

import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heimdall import arayuz  # noqa: E402

HF = dict(model_type="qwen3", hidden_size=64, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
          head_dim=32, intermediate_size=128, vocab_size=97, tie_word_embeddings=True, eos_token_id=7)


class FakeProc:
    def __init__(self, cmd, **kw):
        FakeProc.cmd, FakeProc.kw, self.pid = cmd, kw, 4242

    def poll(self):
        return None


def test_api():
    arayuz.RUNS = Path(tempfile.mkdtemp())                      # gerçek runs/ klasörünü kirletme
    arayuz.planla.load_config = lambda name: dict(HF) if name == "a/ok" else (
        dict(HF, model_type="gemma") if name == "a/gemma" else (_ for _ in ()).throw(OSError("yok")))
    c = TestClient(arayuz.app)
    assert "Heimdall" in c.get("/").text
    j = c.post("/api/plan", json={"model": "a/ok", "gpu": "16"}).json()
    assert j["plan"]["supported"] and j["fit"]["donusum_egitimi"][1] == "yeşil" and "KV cache" in j["text"]
    assert c.post("/api/plan", json={"model": "a/yok"}).status_code == 404
    assert c.post("/api/plan", json={"model": "../etc/passwd"}).status_code == 400
    assert c.post("/api/plan", json={"model": "a b; rm -rf /"}).status_code == 400

    arayuz._running = lambda pattern: True                       # uzun eğitim çalışıyor
    r = c.post("/api/start", json={"model": "a/ok"})
    assert r.status_code == 409 and "dur.sh" in r.json()["detail"]
    arayuz._running = lambda pattern: False
    assert c.post("/api/start", json={"model": "a/gemma"}).status_code == 422   # desteklenmeyen model

    arayuz.subprocess.Popen = FakeProc
    r = c.post("/api/start", json={"model": "a/ok", "q4": True, "batch": 1, "steps_a": 5, "steps_b": 6})
    assert r.status_code == 200, r.text
    cmd = FakeProc.cmd
    assert cmd[1:3] == ["-m", "heimdall.donustur"] and "--q4" in cmd and "--grad-ckpt" in cmd
    assert cmd[cmd.index("--sep") + 1] == "7" and cmd[cmd.index("--steps-b") + 1] == "6"
    assert FakeProc.kw.get("shell") is None and FakeProc.kw["start_new_session"] is True   # kabuk yok, ayrı süreç grubu
    assert c.post("/api/start", json={"model": "a/ok"}).status_code == 409                  # zaten çalışıyor
    assert c.get("/api/status").json()["running"] is True


if __name__ == "__main__":
    test_api()
    print("OK arayüz API (plan, doğrulama, reddetme, komut kurulumu)")
