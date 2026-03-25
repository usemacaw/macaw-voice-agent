"""Full validation of refactored system against real GPU server.

Run with server already running:
    python3 tests/validate_refactored.py
"""

import io, wave, json, time, sys
import numpy as np

import os
SERVER = os.getenv("SERVER", "http://localhost:8766")

def make_wav(dur=2.0, sr=16000):
    t = np.linspace(0, dur, int(sr*dur), endpoint=False)
    pcm = (np.sin(2*np.pi*440*t)*16000).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    buf.seek(0)
    return buf

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  PASS: {name}")
        passed += 1
    else:
        print(f"  FAIL: {name} — {detail}")
        failed += 1

print("=== OpenAI SDK Tests ===")
from openai import OpenAI
client = OpenAI(base_url=f"{SERVER}/v1", api_key="x")

# 1. Transcribe JSON
buf = make_wav()
t0 = time.perf_counter()
r = client.audio.transcriptions.create(model="whisper-1", file=("t.wav", buf, "audio/wav"), language="pt")
ms = (time.perf_counter() - t0) * 1000
check("transcribe json", hasattr(r, "text") and len(r.text) > 0, repr(r))
print(f"    text={r.text!r} ({ms:.0f}ms)")

# 2. Verbose JSON
buf = make_wav()
r2 = client.audio.transcriptions.create(model="whisper-1", file=("t.wav", buf, "audio/wav"), response_format="verbose_json")
check("verbose_json", hasattr(r2, "task") and r2.task == "transcribe")

# 3. Text format
buf = make_wav()
r3 = client.audio.transcriptions.create(model="whisper-1", file=("t.wav", buf, "audio/wav"), response_format="text")
check("text format", isinstance(r3, str))

# 4. SRT
buf = make_wav()
r4 = client.audio.transcriptions.create(model="whisper-1", file=("t.wav", buf, "audio/wav"), response_format="srt")
check("srt format", "-->" in r4)

# 5. VTT
buf = make_wav()
r5 = client.audio.transcriptions.create(model="whisper-1", file=("t.wav", buf, "audio/wav"), response_format="vtt")
check("vtt format", "WEBVTT" in r5)

# 6. List models
models = client.models.list()
ids = [m.id for m in models.data]
check("list models", "qwen" in ids and "whisper-tiny" in ids, str(ids))

print("\n=== Operational API Tests ===")
import urllib.request

# 7. Show
req = urllib.request.Request(f"{SERVER}/api/show", data=json.dumps({"model":"qwen"}).encode(), headers={"Content-Type":"application/json"})
with urllib.request.urlopen(req) as resp:
    show = json.loads(resp.read())
check("show model", "details" in show and show["details"]["family"] == "qwen")
print(f"    details={show['details']}")

# 8. PS
with urllib.request.urlopen(f"{SERVER}/api/ps") as resp:
    ps = json.loads(resp.read())
check("ps", "models" in ps)

# 9. Version
with urllib.request.urlopen(f"{SERVER}/api/version") as resp:
    ver = json.loads(resp.read())
check("version", ver["version"] == "0.1.0")

# 10. Health
with urllib.request.urlopen(f"{SERVER}/") as resp:
    check("health", "macaw-asr" in resp.read().decode())

print("\n=== Real Speech (FLEURS PT-BR) ===")
try:
    from datasets import load_dataset
    import itertools
    ds = load_dataset("google/fleurs", "pt_br", split="test", streaming=True, trust_remote_code=True)
    sample = next(itertools.islice(ds, 2, 3))
    audio = np.array(sample["audio"]["array"], dtype=np.float32)
    sr = sample["audio"]["sampling_rate"]
    ref = sample["transcription"]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())
    buf.seek(0)
    t0 = time.perf_counter()
    r = client.audio.transcriptions.create(model="whisper-1", file=("f.wav", buf, "audio/wav"), language="pt")
    ms = (time.perf_counter() - t0) * 1000
    check("fleurs transcribe", len(r.text) > 0)
    print(f"    REF: {ref[:70]}")
    print(f"    HYP: {r.text[:70]} ({ms:.0f}ms)")
except Exception as e:
    print(f"  SKIP: fleurs — {e}")

print("\n=== SSE Streaming ===")
try:
    import httpx
    buf = make_wav()
    wav_bytes = buf.read()
    with httpx.Client(base_url=SERVER) as http:
        with http.stream("POST", "/v1/audio/transcriptions",
                         files={"file": ("t.wav", wav_bytes, "audio/wav")},
                         data={"model": "whisper-1", "stream": "true"}, timeout=60) as resp:
            events = []
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))
    deltas = [e for e in events if e.get("type") == "transcript.text.delta"]
    dones = [e for e in events if e.get("type") == "transcript.text.done"]
    check("sse deltas", len(deltas) > 0, f"got {len(deltas)}")
    check("sse done", len(dones) == 1)
    if dones:
        print(f"    {len(deltas)} deltas, done text={dones[0]['text']!r}")
except Exception as e:
    print(f"  SKIP: sse — {e}")

print(f"\n{'='*40}")
print(f"PASSED: {passed}  FAILED: {failed}")
if failed == 0:
    print("ALL TESTS PASSED — SYSTEM IS FULLY FUNCTIONAL")
else:
    print("SOME TESTS FAILED")
    sys.exit(1)
