import io
import json
import uuid
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from voxshield.app import create_app
from voxshield.storage import InMemoryAuditStore


def speechish(seconds=4.0, sr=16000):
    t = np.arange(int(seconds * sr)) / sr
    f0 = 120 + 25 * np.sin(2 * np.pi * 1.7 * t)
    sig = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12))
    env = (0.5 + 0.5 * np.sin(2 * np.pi * 3.1 * t)) ** 2
    sig = sig * env
    sig += 0.01 * np.sin(2 * np.pi * 4000 * t)
    return (sig / np.max(np.abs(sig)) * 0.5).astype(np.float32)


buf = io.BytesIO()
sf.write(buf, speechish(), 16000, format="WAV", subtype="PCM_16")
payload = buf.getvalue()
print("wav bytes:", len(payload))

store = InMemoryAuditStore()
client = TestClient(create_app(audit_store=store), raise_server_exceptions=False)

r = client.get("/health")
print("GET /health ->", r.status_code, r.json())
r = client.get("/v1/meta/formats")
print("GET formats ->", r.status_code, json.dumps(r.json())[:160])

sid = str(uuid.uuid4())
r = client.post(
    "/v1/analyze/file",
    files={"audio": ("clip.wav", payload, "audio/wav")},
    data={"session_id": sid},
)
print("POST analyze ->", r.status_code)
body = r.json()
print(json.dumps(body, indent=2, default=str))
assert body["detector"]["score"] is None, "expected no score with no model"
assert body["recommended_action"]["action"] == "none", "expected no action"
assert body["audio_retained"] is False
assert len(store) == 1, f"expected 1 audit record, got {len(store)}"
rec = next(iter(store.iter_all()))
print("audit record keys:", sorted(rec.as_dict()))
print("SMOKE OK")
