"""Model registry — download, cache, and resolve models.

Equivalent to Ollama's model pull/management system.
Uses HuggingFace Hub for downloading and local cache for storage.

Usage:
    registry = ModelRegistry()
    path = await registry.pull("Qwen/Qwen3-ASR-0.6B")  # Downloads if needed
    path = registry.resolve("Qwen/Qwen3-ASR-0.6B")     # Returns local path or raises
    models = registry.list()                              # Lists local models
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Callable

from macaw_asr.api.types import ModelInfo, PullResponse
from macaw_asr.manifest.paths import ModelPaths

logger = logging.getLogger("macaw-asr.manifest.registry")

ProgressCallback = Callable[[PullResponse], None]


from macaw_asr.manifest.contracts import IModelRegistry


class ModelRegistry(IModelRegistry):
    """Downloads, caches, and resolves ASR models.

    Models are stored locally at ~/.macaw-asr/models/.
    Downloads from HuggingFace Hub on first use.
    """

    def __init__(self, paths: ModelPaths | None = None) -> None:
        self._paths = paths or ModelPaths()
        self._paths.ensure_dirs()

    @property
    def paths(self) -> ModelPaths:
        return self._paths

    def resolve(self, model_id: str) -> str:
        """Resolve a model ID to a local path.

        Returns the model_id as-is if it's a local path that exists.
        Otherwise looks in the local cache.

        Raises:
            FileNotFoundError: If model is not downloaded locally.
        """
        # Direct local path
        if Path(model_id).exists():
            return model_id

        # Check local cache
        if self._paths.model_exists(model_id):
            return str(self._paths.model_dir(model_id))

        # Check HuggingFace cache (model may have been loaded via transformers)
        try:
            hf_path = self._resolve_hf_cache(model_id)
            if hf_path:
                return hf_path
        except Exception:
            pass

        raise FileNotFoundError(
            f"Model '{model_id}' not found locally. "
            f"Run: macaw-asr pull {model_id}"
        )

    def pull(
        self,
        model_id: str,
        progress_fn: ProgressCallback | None = None,
    ) -> str:
        """Download a model from HuggingFace Hub.

        Args:
            model_id: HuggingFace model ID (e.g. 'Qwen/Qwen3-ASR-0.6B').
            progress_fn: Optional callback for download progress.

        Returns:
            Local path to the downloaded model.
        """
        if self._paths.model_exists(model_id):
            if progress_fn:
                progress_fn(PullResponse(status="complete", completed=1, total=1))
            logger.info("Model already downloaded: %s", model_id)
            return str(self._paths.model_dir(model_id))

        if progress_fn:
            progress_fn(PullResponse(status="downloading"))

        try:
            from huggingface_hub import snapshot_download

            local_dir = self._paths.model_dir(model_id)
            snapshot_download(
                repo_id=model_id,
                local_dir=str(local_dir),
            )

            if progress_fn:
                progress_fn(PullResponse(status="complete"))

            logger.info("Model downloaded: %s → %s", model_id, local_dir)
            return str(local_dir)

        except ImportError:
            raise RuntimeError(
                "huggingface_hub is required for downloading models. "
                "Install: pip install huggingface_hub"
            )
        except Exception as e:
            if progress_fn:
                progress_fn(PullResponse(status="error"))
            raise RuntimeError(f"Failed to download model '{model_id}': {e}") from e

    def remove(self, model_id: str) -> bool:
        """Remove a locally downloaded model.

        Returns True if removed, False if not found.
        """
        model_dir = self._paths.model_dir(model_id)
        if model_dir.exists():
            shutil.rmtree(model_dir)
            logger.info("Model removed: %s", model_id)
            return True
        return False

    def list(self) -> list[ModelInfo]:
        """List all locally downloaded models."""
        models = []
        for model_id in self._paths.list_models():
            model_dir = self._paths.model_dir(model_id)
            size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
            models.append(
                ModelInfo(
                    name=model_id.split("/")[-1] if "/" in model_id else model_id,
                    model_id=model_id,
                    size_bytes=size,
                )
            )
        return models

    def _resolve_hf_cache(self, model_id: str) -> str | None:
        """Check if model exists in HuggingFace's default cache."""
        try:
            from huggingface_hub import try_to_load_from_cache
            from huggingface_hub.utils import EntryNotFoundError

            result = try_to_load_from_cache(model_id, "config.json")
            if result is not None and isinstance(result, str):
                # Return parent directory of config.json
                return str(Path(result).parent)
        except Exception:
            pass
        return None
