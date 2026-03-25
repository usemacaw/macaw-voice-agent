"""WER (Word Error Rate) benchmark with real speech dataset.

Extracted from NeMo: speech_to_text_eval.py pattern.
- Load real audio + reference transcriptions (Google FLEURS PT-BR)
- Transcribe each utterance with the configured model
- Compute WER = (S + D + I) / N
- Report per-utterance and aggregate
- Gate on WER threshold

Dataset: google/fleurs pt_br test split (no auth needed, streaming).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from macaw_asr.audio.preprocessing import resample
from macaw_asr.decode.strategies import GreedyWithEarlyStopping


# ==================== WER Computation (NeMo pattern) ====================


def _edit_distance(ref_words: list[str], hyp_words: list[str]) -> tuple[int, int, int]:
    """Levenshtein at word level. Returns (substitutions, deletions, insertions)."""
    n, m = len(ref_words), len(hyp_words)
    # dp[i][j] = edit distance between ref[:i] and hyp[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bt = [[(0, 0, 0)] * (m + 1) for _ in range(n + 1)]  # (S, D, I) counts

    for i in range(n + 1):
        dp[i][0] = i
        bt[i][0] = (0, i, 0)
    for j in range(m + 1):
        dp[0][j] = j
        bt[0][j] = (0, 0, j)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1].lower() == hyp_words[j - 1].lower():
                dp[i][j] = dp[i - 1][j - 1]
                bt[i][j] = bt[i - 1][j - 1]
            else:
                sub = dp[i - 1][j - 1] + 1
                dele = dp[i - 1][j] + 1
                ins = dp[i][j - 1] + 1
                best = min(sub, dele, ins)
                dp[i][j] = best
                if best == sub:
                    s, d, ii = bt[i - 1][j - 1]
                    bt[i][j] = (s + 1, d, ii)
                elif best == dele:
                    s, d, ii = bt[i - 1][j]
                    bt[i][j] = (s, d + 1, ii)
                else:
                    s, d, ii = bt[i][j - 1]
                    bt[i][j] = (s, d, ii + 1)

    return bt[n][m]


def compute_wer(reference: str, hypothesis: str) -> dict:
    """Compute Word Error Rate between reference and hypothesis."""
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()

    if not ref_words:
        return {"wer": 0.0 if not hyp_words else 1.0, "S": 0, "D": 0, "I": len(hyp_words), "N": 0}

    S, D, I = _edit_distance(ref_words, hyp_words)
    N = len(ref_words)
    wer = (S + D + I) / N

    return {"wer": wer, "S": S, "D": D, "I": I, "N": N}


# ==================== Dataset Loading ====================


def _load_fleurs_samples(n_samples: int = 10) -> list[dict]:
    """Load Google FLEURS PT-BR test samples via streaming.

    FLEURS: open dataset, no auth, has PT-BR with verified transcriptions.
    Requires datasets<=2.21 for trust_remote_code support.
    """
    from datasets import load_dataset
    import itertools

    ds = load_dataset(
        "google/fleurs", "pt_br", split="test",
        streaming=True, trust_remote_code=True,
    )
    samples = []
    for sample in itertools.islice(ds, n_samples * 3):  # over-fetch to filter
        audio_array = np.array(sample["audio"]["array"], dtype=np.float32)
        sr = sample["audio"]["sampling_rate"]
        text = sample["transcription"].strip()

        # Skip very short utterances (< 3 words)
        if len(text.split()) < 3:
            continue

        # Resample to 16kHz if needed
        if sr != 16000:
            audio_array = resample(audio_array, sr, 16000)

        duration = len(audio_array) / 16000

        # Skip very long (> 15s) — max_new_tokens=32 can't cover long audio
        if duration > 15.0:
            continue

        samples.append({
            "audio": audio_array,
            "text": text,
            "duration": duration,
        })

        if len(samples) >= n_samples:
            break

    return samples


@pytest.fixture(scope="module")
def fleurs_dataset():
    """Load 10 FLEURS PT-BR test samples."""
    samples = _load_fleurs_samples(10)
    assert len(samples) >= 5, f"Need at least 5 samples, got {len(samples)}"
    return samples


# ==================== Tests ====================


class TestWERComputation:
    """Sanity checks for WER computation before using on real data."""

    def test_identical(self):
        assert compute_wer("olá mundo", "olá mundo")["wer"] == 0.0

    def test_one_substitution(self):
        r = compute_wer("olá mundo", "olá terra")
        assert r["wer"] == 0.5
        assert r["S"] == 1

    def test_one_deletion(self):
        r = compute_wer("olá mundo cruel", "olá mundo")
        assert r["wer"] == pytest.approx(1 / 3)
        assert r["D"] == 1

    def test_one_insertion(self):
        r = compute_wer("olá mundo", "olá belo mundo")
        assert r["wer"] == 0.5
        assert r["I"] == 1

    def test_completely_wrong(self):
        r = compute_wer("olá mundo", "foo bar baz")
        assert r["wer"] >= 1.0

    def test_empty_hypothesis(self):
        r = compute_wer("olá mundo", "")
        assert r["wer"] == 1.0
        assert r["D"] == 2


class TestWEROnRealSpeech:
    """WER evaluation on FLEURS PT-BR with real model inference."""

    def test_aggregate_wer(self, model, fleurs_dataset):
        """From NeMo speech_to_text_eval.py: aggregate WER across utterances.

        Small models won't be SOTA — gate is WER < 100% (produces related text).
        """
        strategy_fn = lambda: GreedyWithEarlyStopping(
            eos_token_id=model.eos_token_id, repetition_window=3
        )

        total_S, total_D, total_I, total_N = 0, 0, 0, 0
        total_audio_sec = 0.0
        total_proc_sec = 0.0

        print(f"\n  === WER Evaluation: {len(fleurs_dataset)} utterances ===\n")

        for i, sample in enumerate(fleurs_dataset):
            t0 = time.perf_counter()
            inputs = model.prepare_inputs(sample["audio"])
            output = model.generate(inputs, strategy_fn())
            proc_sec = time.perf_counter() - t0
            total_proc_sec += proc_sec
            total_audio_sec += sample["duration"]

            hyp = output.text.strip()
            ref = sample["text"].strip()
            wer_result = compute_wer(ref, hyp)

            total_S += wer_result["S"]
            total_D += wer_result["D"]
            total_I += wer_result["I"]
            total_N += wer_result["N"]

            print(f"  [{i}] WER={wer_result['wer']:.0%} ({wer_result['S']}S {wer_result['D']}D {wer_result['I']}I / {wer_result['N']}W)")
            print(f"       REF: {ref[:80]}{'...' if len(ref) > 80 else ''}")
            print(f"       HYP: {hyp[:80]}{'...' if len(hyp) > 80 else ''}")

        agg_wer = (total_S + total_D + total_I) / total_N if total_N > 0 else 1.0
        rtf = total_proc_sec / total_audio_sec if total_audio_sec > 0 else 0

        print(f"\n  === AGGREGATE ===")
        print(f"  WER:  {agg_wer:.1%} ({total_S}S + {total_D}D + {total_I}I = {total_S+total_D+total_I} errors / {total_N} words)")
        print(f"  RTF:  {rtf:.3f} ({rtf**-1:.1f}x real-time)")
        print(f"  Audio: {total_audio_sec:.1f}s total")

        # Gate: model must produce SOMETHING related (WER < 100%)
        assert agg_wer < 1.0, f"WER={agg_wer:.1%} — model produces nothing useful"

    def test_no_empty_hypotheses(self, model, fleurs_dataset):
        """Model must produce non-empty output for all real speech utterances."""
        strategy_fn = lambda: GreedyWithEarlyStopping(
            eos_token_id=model.eos_token_id, repetition_window=3
        )

        empty_count = 0
        for i, sample in enumerate(fleurs_dataset):
            inputs = model.prepare_inputs(sample["audio"])
            output = model.generate(inputs, strategy_fn())
            if not output.text.strip():
                empty_count += 1
                print(f"\n  [{i}] EMPTY for: {sample['text'][:60]}...")

        print(f"\n  Empty hypotheses: {empty_count}/{len(fleurs_dataset)}")
        assert empty_count == 0, f"{empty_count} utterances got empty transcription"

    def test_per_utterance_rtf(self, model, fleurs_dataset):
        """From NeMo: every utterance must transcribe faster than real-time."""
        strategy_fn = lambda: GreedyWithEarlyStopping(
            eos_token_id=model.eos_token_id, repetition_window=3
        )

        for i, sample in enumerate(fleurs_dataset):
            t0 = time.perf_counter()
            inputs = model.prepare_inputs(sample["audio"])
            model.generate(inputs, strategy_fn())
            proc_sec = time.perf_counter() - t0

            rtf = proc_sec / sample["duration"]
            assert rtf < 1.0, (
                f"Utterance {i}: RTF={rtf:.3f} >= 1.0 "
                f"({proc_sec*1000:.0f}ms / {sample['duration']:.1f}s)"
            )
