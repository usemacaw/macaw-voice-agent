"""Validate Whisper via OpenAI SDK against server on port 8773."""

from openai import OpenAI
import io, wave, numpy as np, time, json, sys

import os
SERVER = os.getenv("SERVER", "http://localhost:8766")
client = OpenAI(base_url=f"{SERVER}/v1", api_key="x")

passed = failed = 0
def check(name, ok, detail=""):
    global passed, failed
    if ok: print(f"  PASS: {name}"); passed += 1
    else: print(f"  FAIL: {name} — {detail}"); failed += 1

def wav(dur=2.0):
    t = np.linspace(0, dur, int(16000*dur), endpoint=False)
    pcm = (np.sin(2*np.pi*440*t)*16000).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
        wf.writeframes(pcm.tobytes())
    buf.seek(0)
    return buf

print("=== Whisper via OpenAI SDK ===")

# 1. Transcribe JSON
buf = wav()
t0 = time.perf_counter()
r = client.audio.transcriptions.create(model="whisper-1", file=("t.wav", buf, "audio/wav"), language="pt")
ms = (time.perf_counter()-t0)*1000
check("transcribe", hasattr(r, "text") and len(r.text) > 0)
print(f"    text={r.text!r} ({ms:.0f}ms)")

# 2. verbose_json
buf = wav()
r2 = client.audio.transcriptions.create(model="whisper-1", file=("t.wav", buf, "audio/wav"), response_format="verbose_json")
check("verbose_json", hasattr(r2, "task") and r2.task == "transcribe")

# 3. text
buf = wav()
r3 = client.audio.transcriptions.create(model="whisper-1", file=("t.wav", buf, "audio/wav"), response_format="text")
check("text format", isinstance(r3, str))

# 4. srt
buf = wav()
r4 = client.audio.transcriptions.create(model="whisper-1", file=("t.wav", buf, "audio/wav"), response_format="srt")
check("srt", "-->" in r4)

# 5. models
models = client.models.list()
ids = [m.id for m in models.data]
check("models", "whisper-tiny" in ids and "qwen" in ids, str(ids))

# 6. show
import urllib.request
req = urllib.request.Request(f"{SERVER}/api/show", data=json.dumps({"model":"whisper-tiny"}).encode(), headers={"Content-Type":"application/json"})
with urllib.request.urlopen(req) as resp:
    show = json.loads(resp.read())
check("show", "details" in show)
print(f"    details={show['details']}")

# 7. FLEURS real speech
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
        wf.writeframes((audio*32767).astype(np.int16).tobytes())
    buf.seek(0)
    t0 = time.perf_counter()
    r7 = client.audio.transcriptions.create(model="whisper-1", file=("f.wav", buf, "audio/wav"), language="pt")
    ms = (time.perf_counter()-t0)*1000
    check("fleurs", len(r7.text) > 0)
    print(f"    REF: {ref[:60]}")
    print(f"    HYP: {r7.text[:60]} ({ms:.0f}ms)")
except Exception as e:
    print(f"  SKIP: fleurs — {e}")

# 8. ps
with urllib.request.urlopen(f"{SERVER}/api/ps") as resp:
    ps = json.loads(resp.read())
check("ps", len(ps["models"]) > 0)

print(f"\n{'='*40}")
print(f"PASSED: {passed}  FAILED: {failed}")
if failed == 0:
    print("WHISPER SERVER — ALL TESTS PASSED")
else:
    sys.exit(1)
