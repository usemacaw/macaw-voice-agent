#!/usr/bin/env python3
"""
End-to-end test for OpenVoiceAPI.

Tests the full flow:
  1. Connect via WebSocket
  2. Send session.update with config
  3. Send a text message (conversation.item.create)
  4. Trigger response (response.create)
  5. Collect and validate all response events
  6. Optionally send audio (input_audio_buffer.append + commit)

Usage:
  # Text-only mode (no ASR/TTS needed):
  python scripts/test_e2e.py --url ws://localhost:8765/v1/realtime --text-only

  # Audio mode (requires ASR + TTS servers):
  python scripts/test_e2e.py --url ws://localhost:8765/v1/realtime

  # With API key:
  python scripts/test_e2e.py --url ws://localhost:8765/v1/realtime --api-key mykey

  # Against Vast.ai:
  python scripts/test_e2e.py --url ws://VAST_IP:8765/v1/realtime
"""

import argparse
import asyncio
import base64
import json
import struct
import sys
import time
import math

import websockets


def generate_sine_pcm(freq: float = 440.0, duration_s: float = 1.0, sample_rate: int = 24000) -> bytes:
    """Generate a sine wave as PCM 16-bit audio."""
    num_samples = int(sample_rate * duration_s)
    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        value = int(math.sin(2 * math.pi * freq * t) * 16000)
        samples.append(struct.pack('<h', max(-32768, min(32767, value))))
    return b''.join(samples)


async def test_text_mode(url: str, api_key: str = "") -> bool:
    """Test text-only mode: create message, get text response."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    print(f"\n{'='*60}")
    print(f"  E2E Test: TEXT MODE")
    print(f"  URL: {url}")
    print(f"{'='*60}\n")

    async with websockets.connect(url, additional_headers=headers) as ws:
        # 1. session.created
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        assert msg["type"] == "session.created", f"Expected session.created, got {msg['type']}"
        session_id = msg["session"]["id"]
        print(f"  [OK] session.created (id={session_id[:16]}...)")

        # 2. conversation.created
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        assert msg["type"] == "conversation.created"
        print(f"  [OK] conversation.created")

        # 3. Switch to text-only mode
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {"modalities": ["text"]},
        }))
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        assert msg["type"] == "session.updated"
        print(f"  [OK] session.updated (text-only)")

        # 4. Create user message
        await ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Say hello in exactly 5 words."}],
            },
        }))
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        assert msg["type"] == "conversation.item.created"
        print(f"  [OK] conversation.item.created")

        # 5. Request response
        t0 = time.perf_counter()
        await ws.send(json.dumps({"type": "response.create"}))

        # 6. Collect response events
        events = []
        full_text = ""
        for _ in range(50):
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
                events.append(msg)

                if msg["type"] == "response.text.delta":
                    full_text += msg.get("delta", "")

                if msg["type"] == "response.done":
                    break
            except asyncio.TimeoutError:
                print(f"  [WARN] Timeout waiting for events after {len(events)} events")
                break

        elapsed_ms = (time.perf_counter() - t0) * 1000
        types = [e["type"] for e in events]

        print(f"\n  Response events ({len(events)}):")
        for t in types:
            print(f"    - {t}")

        # Validate event ordering
        ok = True
        for expected in ["response.created", "response.output_item.added", "response.content_part.added", "response.done"]:
            if expected in types:
                print(f"  [OK] {expected}")
            else:
                print(f"  [FAIL] Missing {expected}")
                ok = False

        if full_text:
            print(f"\n  LLM Response: \"{full_text}\"")
        else:
            print(f"\n  [WARN] No text received")

        done_event = next((e for e in events if e["type"] == "response.done"), None)
        if done_event:
            status = done_event.get("response", {}).get("status", "?")
            print(f"  Response status: {status}")
            if status != "completed":
                print(f"  [FAIL] Expected status=completed, got {status}")
                ok = False

        print(f"\n  Total time: {elapsed_ms:.0f}ms")
        return ok


async def test_audio_mode(url: str, api_key: str = "") -> bool:
    """Test audio mode: send audio, get audio response."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    print(f"\n{'='*60}")
    print(f"  E2E Test: AUDIO MODE")
    print(f"  URL: {url}")
    print(f"{'='*60}\n")

    async with websockets.connect(url, additional_headers=headers) as ws:
        # 1-2. Initial events
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        assert msg["type"] == "session.created"
        print(f"  [OK] session.created")

        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        assert msg["type"] == "conversation.created"

        # 3. Update session - disable VAD for manual commit test
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "turn_detection": None,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
            },
        }))
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        assert msg["type"] == "session.updated"
        print(f"  [OK] session.updated (audio, manual turn)")

        # 4. Instead of real audio, create a text message to test audio response pipeline
        await ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello, say hi back briefly."}],
            },
        }))
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        assert msg["type"] == "conversation.item.created"
        print(f"  [OK] conversation.item.created (text input for audio response)")

        # 5. Request response — this triggers LLM → TTS pipeline
        t0 = time.perf_counter()
        await ws.send(json.dumps({"type": "response.create"}))

        # 6. Collect events
        events = []
        audio_deltas = 0
        transcript_parts = []
        for _ in range(200):
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
                events.append(msg)

                if msg["type"] == "response.audio.delta":
                    audio_deltas += 1
                elif msg["type"] == "response.audio_transcript.delta":
                    transcript_parts.append(msg.get("delta", ""))
                elif msg["type"] == "response.done":
                    break
            except asyncio.TimeoutError:
                print(f"  [WARN] Timeout after {len(events)} events")
                break

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Summarize events (don't print every audio.delta)
        type_counts = {}
        for e in events:
            t = e["type"]
            type_counts[t] = type_counts.get(t, 0) + 1

        print(f"\n  Response events ({len(events)} total):")
        for t, count in type_counts.items():
            print(f"    - {t}: {count}")

        # Validate
        ok = True
        for expected in ["response.created", "response.output_item.added", "response.content_part.added"]:
            if expected in type_counts:
                print(f"  [OK] {expected}")
            else:
                print(f"  [FAIL] Missing {expected}")
                ok = False

        if audio_deltas > 0:
            print(f"  [OK] {audio_deltas} audio.delta events (audio received)")
        else:
            print(f"  [FAIL] No audio.delta events — TTS pipeline may have failed")
            ok = False

        transcript = "".join(transcript_parts)
        if transcript:
            print(f"  [OK] Transcript: \"{transcript}\"")
        else:
            print(f"  [WARN] No transcript received")

        done_event = next((e for e in events if e["type"] == "response.done"), None)
        if done_event:
            status = done_event.get("response", {}).get("status", "?")
            print(f"  Response status: {status}")
            if status != "completed":
                ok = False

        # Test audio commit flow (send actual audio bytes)
        print(f"\n  --- Testing audio buffer commit ---")
        audio_pcm = generate_sine_pcm(440, 0.5, 24000)
        audio_b64 = base64.b64encode(audio_pcm).decode("ascii")

        # Send in chunks (simulating real-time streaming)
        chunk_size = 4800  # ~100ms at 24kHz
        for i in range(0, len(audio_b64), chunk_size):
            chunk = audio_b64[i:i + chunk_size]
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": chunk,
            }))
        print(f"  [OK] Sent {len(audio_pcm)} bytes of audio ({len(audio_pcm)/48000*1000:.0f}ms)")

        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        # Wait for committed + item created
        commit_events = []
        for _ in range(10):
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
                commit_events.append(msg)
                types_so_far = [e["type"] for e in commit_events]
                if "input_audio_buffer.committed" in types_so_far:
                    break
            except asyncio.TimeoutError:
                break

        commit_types = [e["type"] for e in commit_events]
        if "input_audio_buffer.committed" in commit_types:
            print(f"  [OK] input_audio_buffer.committed")
        else:
            print(f"  [FAIL] No committed event. Got: {commit_types}")
            ok = False

        print(f"\n  Total time: {elapsed_ms:.0f}ms")
        return ok


async def test_audio_buffer_operations(url: str, api_key: str = "") -> bool:
    """Test audio buffer operations: append, clear, commit."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    print(f"\n{'='*60}")
    print(f"  E2E Test: BUFFER OPERATIONS")
    print(f"{'='*60}\n")

    async with websockets.connect(url, additional_headers=headers) as ws:
        await ws.recv()  # session.created
        await ws.recv()  # conversation.created

        # Disable VAD
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {"turn_detection": None},
        }))
        await ws.recv()  # session.updated

        # Test clear
        await ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        assert msg["type"] == "input_audio_buffer.cleared"
        print(f"  [OK] input_audio_buffer.cleared")

        # Test item CRUD
        await ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {"id": "test_item_1", "type": "message", "role": "user", "content": [{"type": "input_text", "text": "test"}]},
        }))
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        assert msg["type"] == "conversation.item.created"
        print(f"  [OK] conversation.item.created")

        await ws.send(json.dumps({
            "type": "conversation.item.delete",
            "item_id": "test_item_1",
        }))
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        assert msg["type"] == "conversation.item.deleted"
        print(f"  [OK] conversation.item.deleted")

        return True


async def main():
    parser = argparse.ArgumentParser(description="OpenVoiceAPI E2E Test")
    parser.add_argument("--url", default="ws://localhost:8765/v1/realtime")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--text-only", action="store_true", help="Skip audio tests")
    args = parser.parse_args()

    results = []

    # Test 1: Buffer operations
    try:
        ok = await test_audio_buffer_operations(args.url, args.api_key)
        results.append(("Buffer Operations", ok))
    except Exception as e:
        print(f"  [ERROR] {e}")
        results.append(("Buffer Operations", False))

    # Test 2: Text mode
    try:
        ok = await test_text_mode(args.url, args.api_key)
        results.append(("Text Mode", ok))
    except Exception as e:
        print(f"  [ERROR] {e}")
        results.append(("Text Mode", False))

    # Test 3: Audio mode
    if not args.text_only:
        try:
            ok = await test_audio_mode(args.url, args.api_key)
            results.append(("Audio Mode", ok))
        except Exception as e:
            print(f"  [ERROR] {e}")
            results.append(("Audio Mode", False))

    # Summary
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    all_ok = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("  All tests passed!")
    else:
        print("  Some tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
