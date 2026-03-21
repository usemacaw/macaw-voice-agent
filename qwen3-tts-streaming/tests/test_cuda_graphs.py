"""Tests for TalkerCUDAGraph and PredictorCUDAGraph — CPU-safe logic only.

Tests initialization, buffer allocation, and state management.
capture() and run() require GPU and are NOT tested here.
"""

import torch
import pytest

from macaw_tts.talker_graph import TalkerCUDAGraph
from macaw_tts.predictor_graph import PredictorCUDAGraph


class _TalkerConfig:
    """Real config object matching Qwen3-TTS talker config shape."""

    def __init__(self):
        self.hidden_size = 128
        self.num_hidden_layers = 4
        self.num_attention_heads = 4
        self.num_key_value_heads = 4
        self.head_dim = 32  # hidden_size / num_attention_heads


class _PredictorConfig:
    """Real config object matching Qwen3-TTS predictor config shape."""

    def __init__(self):
        self.hidden_size = 64
        self.num_hidden_layers = 2
        self.num_attention_heads = 4
        self.num_key_value_heads = 4
        self.head_dim = 16
        self.num_code_groups = 16  # CB0 + CB1-CB15
        self.layer_types = ["full_attention", "full_attention"]


class _DummyModel(torch.nn.Module):
    """Minimal model for testing graph initialization without GPU."""

    def __init__(self, config):
        super().__init__()
        self.config = config


class TestTalkerCUDAGraphInit:
    """Tests for TalkerCUDAGraph initialization and buffer allocation."""

    def test_buffers_allocated_correctly(self):
        config = _TalkerConfig()
        model = _DummyModel(config)
        graph = TalkerCUDAGraph(model, config, device="cpu", max_seq_len=512)

        assert graph.input_buf.shape == (1, 1, 128)
        assert graph.output_buf.shape == (1, 1, 128)
        assert graph.cache_position.shape == (1,)
        assert graph.max_seq_len == 512
        assert graph.hidden_size == 128
        assert graph.num_layers == 4

    def test_not_captured_initially(self):
        config = _TalkerConfig()
        model = _DummyModel(config)
        graph = TalkerCUDAGraph(model, config, device="cpu")

        assert graph.captured is False
        assert graph.graph is None
        assert graph.static_cache is None

    def test_dtype_preserved(self):
        config = _TalkerConfig()
        model = _DummyModel(config)
        graph = TalkerCUDAGraph(model, config, device="cpu", dtype=torch.float32)

        assert graph.input_buf.dtype == torch.float32
        assert graph.output_buf.dtype == torch.float32

    def test_reset_without_cache_is_safe(self):
        config = _TalkerConfig()
        model = _DummyModel(config)
        graph = TalkerCUDAGraph(model, config, device="cpu")
        graph.reset()  # should not raise


class TestPredictorCUDAGraphInit:
    """Tests for PredictorCUDAGraph initialization and buffer allocation."""

    def test_buffers_allocated_correctly(self):
        config = _PredictorConfig()

        class _Predictor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.small_to_mtp_projection = torch.nn.Linear(128, 64)
                self.model = _DummyModel(config)
                self.lm_head = torch.nn.ModuleList([torch.nn.Linear(64, 100) for _ in range(15)])
                self.model.codec_embedding = torch.nn.ModuleList(
                    [torch.nn.Embedding(100, 64) for _ in range(15)]
                )

        predictor = _Predictor()
        graph = PredictorCUDAGraph(
            predictor, config, talker_hidden_size=128,
            device="cpu", dtype=torch.float32,
        )

        assert graph.input_buf.shape == (1, 2, 128)
        assert graph.output_tokens.shape == (15,)
        assert graph.num_codebooks == 15
        assert graph.max_seq == 17  # 2 + 15

    def test_not_captured_initially(self):
        config = _PredictorConfig()

        class _Predictor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.small_to_mtp_projection = torch.nn.Linear(128, 64)
                self.model = _DummyModel(config)
                self.lm_head = torch.nn.ModuleList([torch.nn.Linear(64, 100) for _ in range(15)])
                self.model.codec_embedding = torch.nn.ModuleList(
                    [torch.nn.Embedding(100, 64) for _ in range(15)]
                )

        predictor = _Predictor()
        graph = PredictorCUDAGraph(
            predictor, config, talker_hidden_size=128,
            device="cpu", dtype=torch.float32,
        )

        assert graph.captured is False
        assert graph.graph is None

    def test_cache_position_tensors_prebuilt(self):
        config = _PredictorConfig()

        class _Predictor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.small_to_mtp_projection = torch.nn.Linear(128, 64)
                self.model = _DummyModel(config)
                self.lm_head = torch.nn.ModuleList([torch.nn.Linear(64, 100) for _ in range(15)])
                self.model.codec_embedding = torch.nn.ModuleList(
                    [torch.nn.Embedding(100, 64) for _ in range(15)]
                )

        predictor = _Predictor()
        graph = PredictorCUDAGraph(
            predictor, config, talker_hidden_size=128,
            device="cpu", dtype=torch.float32,
        )

        # Prefill position should be [0, 1]
        assert graph.prefill_cache_pos.tolist() == [0, 1]
        # Decode positions: [2, 3, 4, ..., 16]
        assert len(graph.decode_cache_positions) == 14  # num_codebooks - 1
        assert graph.decode_cache_positions[0].item() == 2
        assert graph.decode_cache_positions[-1].item() == 15

    def test_sampling_parameters_stored(self):
        config = _PredictorConfig()

        class _Predictor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.small_to_mtp_projection = torch.nn.Linear(128, 64)
                self.model = _DummyModel(config)
                self.lm_head = torch.nn.ModuleList([torch.nn.Linear(64, 100) for _ in range(15)])
                self.model.codec_embedding = torch.nn.ModuleList(
                    [torch.nn.Embedding(100, 64) for _ in range(15)]
                )

        predictor = _Predictor()
        graph = PredictorCUDAGraph(
            predictor, config, talker_hidden_size=128,
            device="cpu", dtype=torch.float32,
            temperature=0.7, top_k=30, top_p=0.9, do_sample=False,
        )

        assert graph.temperature == 0.7
        assert graph.top_k == 30
        assert graph.top_p == 0.9
        assert graph.do_sample is False
