"""Test OpenAI SDK against real macaw-asr server with Qwen model.

Run on GPU instance with server already running:
    python3 -m pytest tests/test_openai_sdk_real.py -v -s
"""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest


def _make_wav(duration_sec=1.0, sr=16000):
    n = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, n, endpoint=False)
    pcm = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    buf.seek(0)
    return buf


@pytest.fixture(scope="module")
def client():
    from openai import OpenAI
    return OpenAI(base_url="http://localhost:8766/v1", api_key="unused")


class TestOpenAISDK:

    def test_transcribe_json(self, client):
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=("test.wav", _make_wav(1.0), "audio/wav"),
            language="pt",
        )
        assert hasattr(result, "text")
        assert isinstance(result.text, str)
        print(f"\n  JSON: {result.text!r}")

    def test_transcribe_text_format(self, client):
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=("test.wav", _make_wav(0.5), "audio/wav"),
            response_format="text",
        )
        # text format returns raw string
        assert isinstance(result, str)
        print(f"\n  Text: {result!r}")

    def test_transcribe_verbose_json(self, client):
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=("test.wav", _make_wav(1.0), "audio/wav"),
            language="pt",
            response_format="verbose_json",
        )
        assert hasattr(result, "text")
        assert hasattr(result, "task")
        assert result.task == "transcribe"
        assert hasattr(result, "duration")
        assert hasattr(result, "segments")
        print(f"\n  Verbose: task={result.task} dur={result.duration} text={result.text!r}")

    def test_transcribe_srt_format(self, client):
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=("test.wav", _make_wav(1.0), "audio/wav"),
            response_format="srt",
        )
        assert isinstance(result, str)
        assert "-->" in result
        print(f"\n  SRT:\n{result}")

    def test_transcribe_vtt_format(self, client):
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=("test.wav", _make_wav(1.0), "audio/wav"),
            response_format="vtt",
        )
        assert isinstance(result, str)
        assert "WEBVTT" in result
        print(f"\n  VTT:\n{result}")

    def test_model_whisper1_works(self, client):
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=("test.wav", _make_wav(1.0), "audio/wav"),
        )
        assert len(result.text) >= 0

    def test_model_gpt4o_transcribe_works(self, client):
        result = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=("test.wav", _make_wav(1.0), "audio/wav"),
        )
        assert len(result.text) >= 0

    def test_transcribe_streaming_sse(self, client):
        """OpenAI streaming: stream=true returns SSE with transcript.text.delta events."""
        import httpx

        wav_buf = _make_wav(2.0)
        wav_bytes = wav_buf.read()

        # Use httpx directly for SSE streaming (OpenAI SDK doesn't expose raw SSE)
        with httpx.Client(base_url="http://localhost:8766") as http:
            with http.stream(
                "POST", "/v1/audio/transcriptions",
                files={"file": ("test.wav", wav_bytes, "audio/wav")},
                data={"model": "whisper-1", "stream": "true", "language": "pt"},
                timeout=60,
            ) as response:
                assert response.status_code == 200
                events = []
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        import json
                        events.append(json.loads(line[6:]))

                assert len(events) >= 1
                # Check deltas
                deltas = [e for e in events if e.get("type") == "transcript.text.delta"]
                done = [e for e in events if e.get("type") == "transcript.text.done"]
                assert len(done) == 1, f"Expected 1 done event, got {len(done)}"

                full_text = done[0]["text"]
                delta_text = "".join(d["delta"] for d in deltas)

                print(f"\n  Deltas: {len(deltas)} events")
                print(f"  Delta text: {delta_text!r}")
                print(f"  Done text: {full_text!r}")
                assert len(full_text) > 0

    def test_list_models(self, client):
        models = client.models.list()
        ids = [m.id for m in models.data]
        assert "whisper-1" in ids
        assert "qwen" in ids
        print(f"\n  Models: {ids}")

    def test_real_speech_fleurs(self, client):
        """Transcribe real FLEURS PT-BR via OpenAI SDK."""
        try:
            from datasets import load_dataset
            import itertools
            ds = load_dataset("google/fleurs", "pt_br", split="test",
                              streaming=True, trust_remote_code=True)
            sample = next(itertools.islice(ds, 2, 3))
            audio = np.array(sample["audio"]["array"], dtype=np.float32)
            sr = sample["audio"]["sampling_rate"]
            ref = sample["transcription"]

            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes((audio * 32767).astype(np.int16).tobytes())
            buf.seek(0)

            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=("fleurs.wav", buf, "audio/wav"),
                language="pt",
            )
            print(f"\n  REF: {ref}")
            print(f"  HYP: {result.text}")
            assert len(result.text) > 0
        except ImportError:
            pytest.skip("datasets not installed")
