"""Model storage paths.

Manages the local filesystem layout for downloaded models.
Equivalent to Ollama's manifest/paths.go.

Layout:
    ~/.macaw-asr/
    ├── models/
    │   ├── Qwen--Qwen3-ASR-0.6B/    # HuggingFace cache (symlink or dir)
    │   └── ...
    └── config.json                    # Optional global config
"""

from __future__ import annotations

import os
from pathlib import Path


class ModelPaths:
    """Resolves filesystem paths for model storage.

    Uses MACAW_ASR_HOME env var or defaults to ~/.macaw-asr/.
    """

    def __init__(self, home: str | Path | None = None) -> None:
        if home is not None:
            self._home = Path(home)
        else:
            self._home = Path(
                os.getenv("MACAW_ASR_HOME", Path.home() / ".macaw-asr")
            )

    @property
    def home(self) -> Path:
        """Root directory: ~/.macaw-asr/"""
        return self._home

    @property
    def models_dir(self) -> Path:
        """Models directory: ~/.macaw-asr/models/"""
        return self._home / "models"

    def model_dir(self, model_id: str) -> Path:
        """Directory for a specific model.

        Converts 'Qwen/Qwen3-ASR-0.6B' → ~/.macaw-asr/models/Qwen--Qwen3-ASR-0.6B/
        """
        safe_name = model_id.replace("/", "--")
        return self.models_dir / safe_name

    def model_exists(self, model_id: str) -> bool:
        """Check if a model is downloaded locally."""
        model_dir = self.model_dir(model_id)
        if not model_dir.exists():
            return False
        # Check for actual model files (not just empty dir)
        return any(model_dir.iterdir())

    def list_models(self) -> list[str]:
        """List all locally downloaded model IDs."""
        if not self.models_dir.exists():
            return []
        models = []
        for entry in sorted(self.models_dir.iterdir()):
            if entry.is_dir() and any(entry.iterdir()):
                model_id = entry.name.replace("--", "/")
                models.append(model_id)
        return models

    def ensure_dirs(self) -> None:
        """Create directory structure if it doesn't exist."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
