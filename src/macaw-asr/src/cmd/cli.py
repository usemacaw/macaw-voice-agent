"""CLI for macaw-asr.

Equivalent to Ollama's cmd/cmd.go. Provides subcommands:
    macaw-asr pull <model>       — Download a model
    macaw-asr serve              — Start HTTP server
    macaw-asr transcribe <file>  — Transcribe audio file
    macaw-asr list               — List local models
    macaw-asr remove <model>     — Remove a model

Uses argparse (stdlib) — no external CLI framework needed.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import wave
from pathlib import Path

logger = logging.getLogger("macaw-asr.cmd")


def main(argv: list[str] | None = None) -> None:
    """Entry point for macaw-asr CLI."""
    parser = argparse.ArgumentParser(
        prog="macaw-asr",
        description="Self-contained ASR engine with pluggable models",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    subparsers = parser.add_subparsers(dest="command")

    # pull
    pull_parser = subparsers.add_parser("pull", help="Download a model")
    pull_parser.add_argument("model", help="Model ID (e.g. Qwen/Qwen3-ASR-0.6B)")

    # serve
    serve_parser = subparsers.add_parser("serve", help="Start HTTP server")
    serve_parser.add_argument(
        "--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)"
    )
    serve_parser.add_argument(
        "--port", type=int, default=8766, help="Port (default: 8766)"
    )

    # transcribe
    transcribe_parser = subparsers.add_parser("transcribe", help="Transcribe audio file")
    transcribe_parser.add_argument("file", help="Audio file (WAV PCM16)")
    transcribe_parser.add_argument("--model", default="", help="Model ID")
    transcribe_parser.add_argument("--language", default="pt", help="Language code")
    transcribe_parser.add_argument("--device", default="cuda:0", help="Device")

    # list
    subparsers.add_parser("list", help="List local models")

    # remove
    remove_parser = subparsers.add_parser("remove", help="Remove a model")
    remove_parser.add_argument("model", help="Model ID to remove")

    args = parser.parse_args(argv)

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "pull":
        _cmd_pull(args)
    elif args.command == "serve":
        _cmd_serve(args)
    elif args.command == "transcribe":
        asyncio.run(_cmd_transcribe(args))
    elif args.command == "list":
        _cmd_list(args)
    elif args.command == "remove":
        _cmd_remove(args)


def _cmd_pull(args: argparse.Namespace) -> None:
    """Download a model."""
    from macaw_asr.manifest.registry import ModelRegistry

    registry = ModelRegistry()

    def _progress(resp):
        if resp.status == "downloading":
            print(f"Downloading {args.model}...")
        elif resp.status == "complete":
            print(f"Done: {args.model}")
        elif resp.status == "error":
            print(f"Error downloading {args.model}", file=sys.stderr)

    try:
        path = registry.pull(args.model, progress_fn=_progress)
        print(f"Model saved to: {path}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _cmd_serve(args: argparse.Namespace) -> None:
    """Start the HTTP server."""
    print(f"Starting macaw-asr server on {args.host}:{args.port}...")
    print("(HTTP server not yet implemented — use the Python API directly)")
    print()
    print("Python API usage:")
    print("  import macaw_asr")
    print('  engine = await macaw_asr.create_engine(model="qwen")')
    print("  text = await engine.transcribe(audio_bytes)")


async def _cmd_transcribe(args: argparse.Namespace) -> None:
    """Transcribe an audio file directly (no server needed)."""
    filepath = Path(args.file)
    if not filepath.exists():
        print(f"File not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    # Read audio file
    audio_bytes = _read_audio_file(filepath)
    if not audio_bytes:
        print(f"Failed to read audio from: {filepath}", file=sys.stderr)
        sys.exit(1)

    from macaw_asr.config import AudioConfig, EngineConfig

    model_name = "qwen"
    model_id = args.model or "Qwen/Qwen3-ASR-0.6B"
    for key in ("qwen", "whisper", "parakeet"):
        if key in model_id.lower():
            model_name = key
            break

    config = EngineConfig(
        model_name=model_name,
        model_id=model_id,
        device=args.device,
        language=args.language,
        audio=AudioConfig(
            input_sample_rate=_detect_sample_rate(filepath),
        ),
    )

    from macaw_asr.runner.engine import ASREngine

    engine = ASREngine(config)
    await engine.start()
    try:
        text = await engine.transcribe(audio_bytes)
        print(text)
    finally:
        await engine.stop()


def _cmd_list(args: argparse.Namespace) -> None:
    """List locally downloaded models."""
    from macaw_asr.manifest.registry import ModelRegistry

    registry = ModelRegistry()
    models = registry.list()

    if not models:
        print("No models downloaded.")
        print("Run: macaw-asr pull Qwen/Qwen3-ASR-0.6B")
        return

    print(f"{'NAME':<30} {'MODEL ID':<40} {'SIZE':<12}")
    print("-" * 82)
    for m in models:
        size_str = _format_size(m.size_bytes)
        print(f"{m.name:<30} {m.model_id:<40} {size_str:<12}")


def _cmd_remove(args: argparse.Namespace) -> None:
    """Remove a local model."""
    from macaw_asr.manifest.registry import ModelRegistry

    registry = ModelRegistry()
    if registry.remove(args.model):
        print(f"Removed: {args.model}")
    else:
        print(f"Model not found: {args.model}", file=sys.stderr)
        sys.exit(1)


# ==================== Helpers ====================


def _read_audio_file(path: Path) -> bytes:
    """Read PCM data from a WAV file."""
    try:
        with wave.open(str(path), "rb") as wf:
            return wf.readframes(wf.getnframes())
    except wave.Error:
        # Try reading as raw PCM
        return path.read_bytes()


def _detect_sample_rate(path: Path) -> int:
    """Detect sample rate from WAV header, default 16000."""
    try:
        with wave.open(str(path), "rb") as wf:
            return wf.getframerate()
    except wave.Error:
        return 16000


def _format_size(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


if __name__ == "__main__":
    main()
