"""RED/GREEN tests: Model manifest with real model files."""

from __future__ import annotations

from pathlib import Path

import pytest

from macaw_asr.manifest.paths import ModelPaths
from macaw_asr.manifest.registry import ModelRegistry


class TestModelPaths:

    def test_model_dir_slash_to_double_dash(self, tmp_path):
        paths = ModelPaths(home=tmp_path)
        d = paths.model_dir("Qwen/Qwen3-ASR-0.6B")
        assert d.name == "Qwen--Qwen3-ASR-0.6B"

    def test_exists_with_files(self, tmp_path):
        paths = ModelPaths(home=tmp_path)
        d = paths.model_dir("org/model")
        d.mkdir(parents=True)
        (d / "config.json").write_text("{}")
        assert paths.model_exists("org/model")

    def test_not_exists_empty_dir(self, tmp_path):
        paths = ModelPaths(home=tmp_path)
        d = paths.model_dir("org/empty")
        d.mkdir(parents=True)
        assert not paths.model_exists("org/empty")

    def test_list_and_remove(self, tmp_path):
        paths = ModelPaths(home=tmp_path)
        registry = ModelRegistry(paths=paths)

        d = paths.model_dir("org/test")
        d.mkdir(parents=True)
        (d / "w.bin").write_bytes(b"x" * 100)

        models = registry.list()
        assert len(models) == 1
        assert models[0].model_id == "org/test"
        assert models[0].size_bytes == 100

        assert registry.remove("org/test")
        assert not d.exists()

    def test_resolve_not_found_shows_fix(self, tmp_path):
        registry = ModelRegistry(paths=ModelPaths(home=tmp_path))
        with pytest.raises(FileNotFoundError, match="macaw-asr pull"):
            registry.resolve("nonexistent/model")
