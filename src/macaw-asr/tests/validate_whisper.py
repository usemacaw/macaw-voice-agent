"""Validate Whisper model on GPU."""

from macaw_asr.models.factory import ModelFactory
from macaw_asr.config import EngineConfig, AudioConfig
import numpy as np, time, sys

config = EngineConfig(
    model_name="whisper-tiny", model_id="openai/whisper-tiny",
    device="cuda:0", dtype="float16", language="pt",
    audio=AudioConfig(input_sample_rate=16000, model_sample_rate=16000),
)

print("Creating whisper-tiny...")
model = ModelFactory.create("whisper-tiny")

print("Loading...")
t0 = time.perf_counter()
model.load(config)
load_ms = (time.perf_counter() - t0) * 1000
print(f"Loaded in {load_ms:.0f}ms, EOS={model.eos_token_id}")

print("Warmup...")
model.warmup(config)

print("\n--- Silence 2s ---")
audio = np.zeros(32000, dtype=np.float32)
inputs = model.prepare_inputs(audio)
out = model.generate(inputs)
total = out.timings.get("total_ms", 0)
print(f"Text: {out.text!r} ({total:.0f}ms, {out.n_tokens} tokens)")

print("\n--- Tone 440Hz 2s ---")
t = np.linspace(0, 2.0, 32000, endpoint=False)
tone = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)
inputs2 = model.prepare_inputs(tone)
out2 = model.generate(inputs2)
total2 = out2.timings.get("total_ms", 0)
print(f"Text: {out2.text!r} ({total2:.0f}ms)")

print("\n--- FLEURS PT-BR ---")
try:
    from datasets import load_dataset
    import itertools
    ds = load_dataset("google/fleurs", "pt_br", split="test", streaming=True, trust_remote_code=True)
    sample = next(itertools.islice(ds, 2, 3))
    audio_real = np.array(sample["audio"]["array"], dtype=np.float32)
    ref = sample["transcription"]

    t0 = time.perf_counter()
    inp = model.prepare_inputs(audio_real)
    out3 = model.generate(inp)
    ms = (time.perf_counter() - t0) * 1000

    print(f"REF: {ref}")
    print(f"HYP: {out3.text}")
    print(f"Time: {ms:.0f}ms, Tokens: {out3.n_tokens}")
except Exception as e:
    print(f"SKIP: {e}")

print("\n--- Stream test ---")
dummy = np.zeros(16000, dtype=np.float32)
inp = model.prepare_inputs(dummy)
for delta, done, output in model.generate_stream(inp):
    print(f"delta={delta!r} done={done}")
    if done and output:
        print(f"Final: {output.text!r}")

model.unload()
print("\nWHISPER-TINY FULLY VALIDATED")
