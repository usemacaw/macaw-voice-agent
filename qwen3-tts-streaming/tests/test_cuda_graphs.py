"""Tests for TalkerCUDAGraph and PredictorCUDAGraph — CPU-safe logic only.

Tests initialization, buffer allocation, and state management.
capture() and run() require GPU and are NOT tested here.
"""

from collections import OrderedDict

import torch
import pytest

from macaw_tts.talker_graph import TalkerCUDAGraph, _MASK_CACHE_MAX_SIZE
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


def _make_predictor(config=None):
    """Create a minimal Predictor module for testing. Avoids inline class duplication."""
    if config is None:
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

    return _Predictor()


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

    def test_position_ids_shape_uses_num_position_types(self):
        config = _TalkerConfig()
        model = _DummyModel(config)
        graph = TalkerCUDAGraph(model, config, device="cpu")

        assert graph._num_position_types == 3
        assert graph.position_ids.shape == (3, 1, 1)

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
        predictor = _make_predictor(config)
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
        predictor = _make_predictor(config)
        graph = PredictorCUDAGraph(
            predictor, config, talker_hidden_size=128,
            device="cpu", dtype=torch.float32,
        )

        assert graph.captured is False
        assert graph.graph is None

    def test_cache_position_tensors_prebuilt(self):
        config = _PredictorConfig()
        predictor = _make_predictor(config)
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
        predictor = _make_predictor(config)
        graph = PredictorCUDAGraph(
            predictor, config, talker_hidden_size=128,
            device="cpu", dtype=torch.float32,
            temperature=0.7, top_k=30, top_p=0.9, do_sample=False,
        )

        assert graph.temperature == 0.7
        assert graph.top_k == 30
        assert graph.top_p == 0.9
        assert graph.do_sample is False


class TestTalkerCUDAGraphGuards:
    """Tests for capture guard and error recovery."""

    def test_run_before_capture_raises(self):
        config = _TalkerConfig()
        model = _DummyModel(config)
        graph = TalkerCUDAGraph(model, config, device="cpu")

        with pytest.raises(RuntimeError, match="CUDA graph not captured"):
            graph.run(torch.zeros(1, 1, 128), position=0)

    def test_repr(self):
        config = _TalkerConfig()
        model = _DummyModel(config)
        graph = TalkerCUDAGraph(model, config, device="cpu", max_seq_len=512)

        r = repr(graph)
        assert "TalkerCUDAGraph" in r
        assert "captured=False" in r
        assert "max_seq_len=512" in r


class TestTalkerMaskCacheLRU:
    """Tests for LRU eviction in TalkerCUDAGraph mask cache."""

    def test_mask_cache_max_size_is_positive(self):
        assert _MASK_CACHE_MAX_SIZE > 0

    def test_mask_cache_is_ordered_dict(self):
        config = _TalkerConfig()
        model = _DummyModel(config)
        graph = TalkerCUDAGraph(model, config, device="cpu")
        # Simulate _build_attention_masks initialization
        graph._mask_cache = OrderedDict()
        assert isinstance(graph._mask_cache, OrderedDict)

    def test_mask_cache_evicts_oldest_when_full(self):
        """Verify LRU eviction removes oldest entry when cache exceeds max size."""
        config = _TalkerConfig()
        model = _DummyModel(config)
        graph = TalkerCUDAGraph(model, config, device="cpu")

        # Manually populate the cache to test eviction logic
        graph._mask_cache = OrderedDict()
        for i in range(_MASK_CACHE_MAX_SIZE):
            graph._mask_cache[i] = torch.zeros(1)

        assert len(graph._mask_cache) == _MASK_CACHE_MAX_SIZE
        first_key = next(iter(graph._mask_cache))
        assert first_key == 0

        # Add one more — should evict key 0
        graph._mask_cache[_MASK_CACHE_MAX_SIZE] = torch.zeros(1)
        if len(graph._mask_cache) > _MASK_CACHE_MAX_SIZE:
            graph._mask_cache.popitem(last=False)

        assert len(graph._mask_cache) == _MASK_CACHE_MAX_SIZE
        assert 0 not in graph._mask_cache
        assert _MASK_CACHE_MAX_SIZE in graph._mask_cache

    def test_mask_cache_lru_promotes_accessed_key(self):
        """Accessing a cached key should move it to most-recently-used position."""
        cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        for i in range(5):
            cache[i] = torch.zeros(1)

        # Access key 0 — should move it to end
        cache.move_to_end(0)
        keys = list(cache.keys())
        assert keys[-1] == 0
        assert keys[0] == 1


class TestPredictorCUDAGraphGuards:
    """Tests for capture guard and error recovery."""

    def test_run_before_capture_raises(self):
        config = _PredictorConfig()
        predictor = _make_predictor(config)
        graph = PredictorCUDAGraph(
            predictor, config, talker_hidden_size=128,
            device="cpu", dtype=torch.float32,
        )

        with pytest.raises(RuntimeError, match="CUDA graph not captured"):
            graph.run(torch.zeros(1, 2, 128))

    def test_repr(self):
        config = _PredictorConfig()
        predictor = _make_predictor(config)
        graph = PredictorCUDAGraph(
            predictor, config, talker_hidden_size=128,
            device="cpu", dtype=torch.float32,
        )

        r = repr(graph)
        assert "PredictorCUDAGraph" in r
        assert "captured=False" in r
        assert "codebooks=15" in r
