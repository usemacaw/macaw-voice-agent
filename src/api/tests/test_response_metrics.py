"""Tests for ResponseMetrics: defaults, to_dict() filtering rules, and merge_prior()."""

import pytest

from protocol.metrics import ResponseMetrics


class TestDefaultValues:
    def test_all_float_fields_default_to_zero(self):
        m = ResponseMetrics()

        float_fields = [
            "session_duration_s", "asr_ms", "speech_ms", "speech_rms",
            "vad_silence_wait_ms", "smart_turn_inference_ms",
            "llm_ttft_ms", "llm_total_ms", "pipeline_first_audio_ms",
            "llm_first_sentence_ms", "tts_synth_ms", "tts_wait_ms",
            "tts_first_chunk_ms", "e2e_ms", "total_ms", "slo_target_ms",
        ]
        for name in float_fields:
            assert getattr(m, name) == 0.0, f"{name} should default to 0.0"

    def test_all_int_fields_default_to_zero(self):
        m = ResponseMetrics()

        int_fields = [
            "turn", "barge_in_count", "input_chars", "asr_partial_count",
            "smart_turn_waits", "output_chars", "tool_rounds",
            "backpressure_level", "events_dropped", "early_trigger_words",
        ]
        for name in int_fields:
            assert getattr(m, name) == 0, f"{name} should default to 0"

    def test_all_string_fields_default_to_empty(self):
        m = ResponseMetrics()

        str_fields = [
            "response_id", "asr_mode", "asr_last_partial", "early_trigger_partial",
        ]
        for name in str_fields:
            assert getattr(m, name) == "", f"{name} should default to empty string"

    def test_slo_met_defaults_to_none(self):
        m = ResponseMetrics()
        assert m.slo_met is None

    def test_list_fields_default_to_empty_lists(self):
        m = ResponseMetrics()
        assert m.tools_used == []
        assert m.tool_timings == []

    def test_list_fields_are_independent_across_instances(self):
        # Each instance must get its own list — no shared mutable default
        m1 = ResponseMetrics()
        m2 = ResponseMetrics()

        m1.tools_used.append("web_search")
        assert m2.tools_used == []

        m1.tool_timings.append({"tool": "web_search", "ms": 50})
        assert m2.tool_timings == []


class TestToDictOmitsDefaults:
    def test_empty_metrics_omits_all_optional_fields(self):
        m = ResponseMetrics()
        result = m.to_dict()

        # Only the zero-value fields that are explicitly kept must appear
        optional_float_fields = [
            "session_duration_s", "asr_ms", "speech_ms", "speech_rms",
            "vad_silence_wait_ms", "smart_turn_inference_ms",
            "llm_ttft_ms", "llm_total_ms", "pipeline_first_audio_ms",
            "llm_first_sentence_ms", "tts_synth_ms", "tts_wait_ms",
            "tts_first_chunk_ms", "e2e_ms", "total_ms", "slo_target_ms",
            "input_chars", "output_chars", "asr_partial_count",
            "smart_turn_waits", "early_trigger_words",
        ]
        for name in optional_float_fields:
            assert name not in result, f"{name}=0 should be omitted from to_dict()"

    def test_empty_string_fields_omitted(self):
        m = ResponseMetrics()
        result = m.to_dict()

        for name in ("response_id", "asr_mode", "asr_last_partial", "early_trigger_partial"):
            assert name not in result, f"empty string field {name!r} should be omitted"

    def test_empty_list_fields_omitted(self):
        m = ResponseMetrics()
        result = m.to_dict()

        assert "tools_used" not in result
        assert "tool_timings" not in result

    def test_slo_met_none_omitted(self):
        m = ResponseMetrics()
        result = m.to_dict()
        assert "slo_met" not in result


class TestToDictKeepsZeroCounters:
    """turn, barge_in_count, tool_rounds, backpressure_level, events_dropped
    must appear in to_dict() even when their value is 0."""

    def test_turn_zero_is_kept(self):
        m = ResponseMetrics()
        assert "turn" in m.to_dict()
        assert m.to_dict()["turn"] == 0

    def test_barge_in_count_zero_is_kept(self):
        m = ResponseMetrics()
        assert "barge_in_count" in m.to_dict()
        assert m.to_dict()["barge_in_count"] == 0

    def test_tool_rounds_zero_is_kept(self):
        m = ResponseMetrics()
        assert "tool_rounds" in m.to_dict()
        assert m.to_dict()["tool_rounds"] == 0

    def test_backpressure_level_zero_is_kept(self):
        m = ResponseMetrics()
        assert "backpressure_level" in m.to_dict()
        assert m.to_dict()["backpressure_level"] == 0

    def test_events_dropped_zero_is_kept(self):
        m = ResponseMetrics()
        assert "events_dropped" in m.to_dict()
        assert m.to_dict()["events_dropped"] == 0


class TestToDictIncludesNonZeroValues:
    def test_asr_ms_zero_omitted_nonzero_included(self):
        m_zero = ResponseMetrics()
        assert "asr_ms" not in m_zero.to_dict()

        m_set = ResponseMetrics()
        m_set.asr_ms = 100.5
        result = m_set.to_dict()
        assert "asr_ms" in result
        assert result["asr_ms"] == 100.5

    def test_nonempty_string_field_included(self):
        m = ResponseMetrics()
        m.response_id = "resp_abc123"
        result = m.to_dict()
        assert result["response_id"] == "resp_abc123"

    def test_nonempty_asr_mode_included(self):
        m = ResponseMetrics()
        m.asr_mode = "streaming"
        assert m.to_dict()["asr_mode"] == "streaming"

    def test_nonempty_tools_used_included(self):
        m = ResponseMetrics()
        m.tools_used = ["web_search", "recall_memory"]
        result = m.to_dict()
        assert result["tools_used"] == ["web_search", "recall_memory"]

    def test_nonempty_tool_timings_included(self):
        m = ResponseMetrics()
        m.tool_timings = [{"tool": "web_search", "ms": 320}]
        result = m.to_dict()
        assert result["tool_timings"] == [{"tool": "web_search", "ms": 320}]

    def test_nonzero_float_fields_all_included(self):
        m = ResponseMetrics()
        m.llm_ttft_ms = 42.0
        m.llm_total_ms = 800.0
        m.tts_synth_ms = 120.5
        m.e2e_ms = 950.0

        result = m.to_dict()
        assert result["llm_ttft_ms"] == 42.0
        assert result["llm_total_ms"] == 800.0
        assert result["tts_synth_ms"] == 120.5
        assert result["e2e_ms"] == 950.0


class TestToDictSloMet:
    def test_slo_met_true_included(self):
        m = ResponseMetrics()
        m.slo_met = True
        result = m.to_dict()
        assert "slo_met" in result
        assert result["slo_met"] is True

    def test_slo_met_false_included(self):
        m = ResponseMetrics()
        m.slo_met = False
        result = m.to_dict()
        assert "slo_met" in result
        assert result["slo_met"] is False

    def test_slo_met_none_omitted(self):
        m = ResponseMetrics()
        m.slo_met = None
        assert "slo_met" not in m.to_dict()


class TestMergePrior:
    def test_merge_full_dict_sets_all_mergeable_fields(self):
        m = ResponseMetrics()
        prior = {
            "asr_ms": 250.0,
            "speech_ms": 1200.0,
            "asr_mode": "batch",
            "input_chars": 42,
            "speech_rms": 0.15,
            "vad_silence_wait_ms": 300.0,
            "smart_turn_inference_ms": 18.5,
            "smart_turn_waits": 2,
            "asr_partial_count": 5,
            "asr_last_partial": "Olá tudo",
            "early_trigger_words": 3,
            "early_trigger_partial": "Olá",
        }

        m.merge_prior(prior)

        assert m.asr_ms == 250.0
        assert m.speech_ms == 1200.0
        assert m.asr_mode == "batch"
        assert m.input_chars == 42
        assert m.speech_rms == 0.15
        assert m.vad_silence_wait_ms == 300.0
        assert m.smart_turn_inference_ms == 18.5
        assert m.smart_turn_waits == 2
        assert m.asr_partial_count == 5
        assert m.asr_last_partial == "Olá tudo"
        assert m.early_trigger_words == 3
        assert m.early_trigger_partial == "Olá"

    def test_merge_partial_dict_only_sets_present_keys(self):
        m = ResponseMetrics()
        prior = {"asr_ms": 150.0, "asr_mode": "streaming"}

        m.merge_prior(prior)

        assert m.asr_ms == 150.0
        assert m.asr_mode == "streaming"
        # Fields not in prior remain at default
        assert m.speech_ms == 0.0
        assert m.input_chars == 0
        assert m.speech_rms == 0.0

    def test_merge_extra_keys_are_ignored(self):
        m = ResponseMetrics()
        prior = {
            "asr_ms": 75.0,
            "nonexistent_field": "should_not_crash",
            "another_unknown_key": 999,
        }

        # Must not raise AttributeError for unknown keys
        m.merge_prior(prior)

        assert m.asr_ms == 75.0

    def test_merge_empty_dict_leaves_all_fields_at_default(self):
        m = ResponseMetrics()
        m.merge_prior({})

        assert m.asr_ms == 0.0
        assert m.speech_ms == 0.0
        assert m.asr_mode == ""
        assert m.input_chars == 0
        assert m.early_trigger_words == 0

    def test_merge_does_not_overwrite_non_mergeable_fields(self):
        # Fields outside _MERGEABLE must never be touched by merge_prior
        m = ResponseMetrics()
        m.llm_ttft_ms = 42.0
        m.turn = 3

        m.merge_prior({"llm_ttft_ms": 999.0, "turn": 99})

        # These are not in _MERGEABLE — must remain unchanged
        assert m.llm_ttft_ms == 42.0
        assert m.turn == 3


class TestFieldReadWrite:
    def test_setting_and_reading_identity_fields(self):
        m = ResponseMetrics()
        m.response_id = "resp_xyz"
        m.turn = 7
        m.session_duration_s = 45.2
        m.barge_in_count = 2

        assert m.response_id == "resp_xyz"
        assert m.turn == 7
        assert m.session_duration_s == 45.2
        assert m.barge_in_count == 2

    def test_setting_and_reading_llm_timing_fields(self):
        m = ResponseMetrics()
        m.llm_ttft_ms = 123.4
        m.llm_total_ms = 987.6

        assert m.llm_ttft_ms == 123.4
        assert m.llm_total_ms == 987.6

    def test_setting_and_reading_tts_fields(self):
        m = ResponseMetrics()
        m.tts_synth_ms = 200.0
        m.tts_wait_ms = 50.0
        m.tts_first_chunk_ms = 80.0

        assert m.tts_synth_ms == 200.0
        assert m.tts_wait_ms == 50.0
        assert m.tts_first_chunk_ms == 80.0

    def test_setting_and_reading_slo_fields(self):
        m = ResponseMetrics()
        m.slo_target_ms = 500.0
        m.slo_met = True

        assert m.slo_target_ms == 500.0
        assert m.slo_met is True

    def test_setting_and_reading_backpressure_fields(self):
        m = ResponseMetrics()
        m.backpressure_level = 2
        m.events_dropped = 15

        assert m.backpressure_level == 2
        assert m.events_dropped == 15

    def test_setting_and_reading_early_trigger_fields(self):
        m = ResponseMetrics()
        m.early_trigger_words = 4
        m.early_trigger_partial = "Qual o saldo"

        assert m.early_trigger_words == 4
        assert m.early_trigger_partial == "Qual o saldo"


class TestListMutations:
    def test_tools_used_append_accumulates(self):
        m = ResponseMetrics()
        m.tools_used.append("web_search")
        m.tools_used.append("recall_memory")

        assert m.tools_used == ["web_search", "recall_memory"]

    def test_tool_timings_append_accumulates(self):
        m = ResponseMetrics()
        m.tool_timings.append({"tool": "web_search", "ms": 320})
        m.tool_timings.append({"tool": "recall_memory", "ms": 12})

        assert len(m.tool_timings) == 2
        assert m.tool_timings[0]["tool"] == "web_search"
        assert m.tool_timings[1]["ms"] == 12

    def test_tools_used_appears_in_to_dict_after_append(self):
        m = ResponseMetrics()
        m.tools_used.append("web_search")

        result = m.to_dict()
        assert result["tools_used"] == ["web_search"]

    def test_tool_timings_appears_in_to_dict_after_append(self):
        m = ResponseMetrics()
        m.tool_timings.append({"tool": "web_search", "ms": 450})

        result = m.to_dict()
        assert result["tool_timings"] == [{"tool": "web_search", "ms": 450}]
