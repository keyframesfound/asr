#!/usr/bin/env python3
"""Download local ASR weights into ``models/`` (idempotent).

Whisper and SenseVoice are stored as Hugging Face snapshots that
``from_pretrained`` / FunASR ``AutoModel`` already load from disk.
Parakeet is stored in the Hugging Face hub cache layout that
``parakeet_mlx.from_pretrained(..., cache_dir=models/parakeet-mlx)`` uses.

iFlytek is cloud-only. Hex / CoreML trees are not downloaded.

Re-running skips any engine whose weight files are already present.
"""
from __future__ import annotations

import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Same folder names as engines.whisper / engines.sensevoice / engines.parakeet.
WHISPER_DIRNAME = "whisper-large-v3-turbo"
SENSEVOICE_DIRNAME = "sensevoice-small"
PARAKEET_DIRNAME = "parakeet-mlx"
WHISPER_REPO_ID = "openai/whisper-large-v3-turbo"
SENSEVOICE_REPO_ID = "FunAudioLLM/SenseVoiceSmall"
PARAKEET_REPO_ID = "mlx-community/parakeet-tdt-0.6b-v3"

# FunASR AutoModel reads these when ``model`` is a local directory
# (configuration.json file_path_metas → config.yaml, model.pt, bpe, cmvn).
SENSEVOICE_FILES = (
    "configuration.json",
    "config.yaml",
    "model.pt",
    "am.mvn",
    "chn_jpn_yue_eng_ko_spectok.bpe.model",
)

# Real weights are hundreds of MB. A git-lfs pointer is ~100 bytes.
MIN_WEIGHT_BYTES = 10 * 1024 * 1024

_OFFLINE_ENV = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")

Downloader = Callable[..., str]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    title: str
    repo_id: str
    approx: str


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="whisper",
        title="Whisper Large V3 Turbo",
        repo_id=WHISPER_REPO_ID,
        approx="~1.6 GB",
    ),
    ModelSpec(
        key="sensevoice",
        title="SenseVoice Small",
        repo_id=SENSEVOICE_REPO_ID,
        approx="~0.9 GB",
    ),
    ModelSpec(
        key="parakeet",
        title="Parakeet TDT 0.6B (parakeet-mlx)",
        repo_id=PARAKEET_REPO_ID,
        approx="~2.5 GB",
    ),
)


def model_dest(root: Path, key: str) -> Path:
    models = root / "models"
    if key == "whisper":
        return models / WHISPER_DIRNAME
    if key == "sensevoice":
        return models / SENSEVOICE_DIRNAME
    if key == "parakeet":
        return models / PARAKEET_DIRNAME
    raise KeyError(key)


def _nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _large(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size >= MIN_WEIGHT_BYTES
    except OSError:
        return False


def _download_incomplete(path: Path) -> bool:
    """True when Hugging Face left a partial ``*.incomplete`` file behind."""
    try:
        return any(path.rglob("*.incomplete"))
    except OSError:
        return False


def whisper_weights_ready(model_dir: Path) -> bool:
    """True when ``AutoModelForSpeechSeq2Seq.from_pretrained(model_dir)`` can load."""
    if not model_dir.is_dir() or _download_incomplete(model_dir):
        return False
    weights = _large(model_dir / "model.safetensors") or _large(model_dir / "pytorch_model.bin")
    return bool(
        weights
        and _nonempty(model_dir / "config.json")
        and _nonempty(model_dir / "preprocessor_config.json")
        and _nonempty(model_dir / "tokenizer.json")
    )


def sensevoice_weights_ready(model_dir: Path) -> bool:
    """True when FunASR ``AutoModel(model=model_dir)`` has its local files."""
    if not model_dir.is_dir() or _download_incomplete(model_dir):
        return False
    small = [model_dir / name for name in SENSEVOICE_FILES if name != "model.pt"]
    return all(_nonempty(path) for path in small) and _large(model_dir / "model.pt")


def parakeet_weights_ready(
    cache_dir: Path,
    repo_id: str = PARAKEET_REPO_ID,
) -> bool:
    """True when this cache holds Parakeet MLX weights.

    Matches ``engines.parakeet.weights_cached`` for the project cache:
    ``models--<repo>/snapshots/*/config.json`` plus a safetensors file.
    Hub-cache symlinks count. A git-lfs pointer or ``*.incomplete`` blob does not.
    """
    if not cache_dir.is_dir() or _download_incomplete(cache_dir):
        return False
    repo = cache_dir / ("models--" + repo_id.replace("/", "--"))
    if not repo.is_dir():
        return False
    for cfg in repo.glob("snapshots/*/config.json"):
        parent = cfg.parent
        if not _nonempty(cfg):
            continue
        if _large(parent / "model.safetensors") or any(
            _large(path) for path in parent.glob("*.safetensors")
        ):
            return True
    return False


def weights_ready(key: str, dest: Path) -> bool:
    if key == "whisper":
        return whisper_weights_ready(dest)
    if key == "sensevoice":
        return sensevoice_weights_ready(dest)
    if key == "parakeet":
        return parakeet_weights_ready(dest)
    raise KeyError(key)


def _enable_network() -> None:
    """Drop offline flags so an explicit download can reach Hugging Face.

    ``./run`` sets ``HF_HUB_OFFLINE`` once Parakeet is cached. This script is
    the install path and must still be able to fetch Whisper or SenseVoice.
    """
    cleared = [key for key in _OFFLINE_ENV if os.environ.pop(key, None) is not None]
    if cleared:
        print(
            "[note] cleared "
            + ", ".join(cleared)
            + " so this download can use the network",
            flush=True,
        )


def _explain(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".strip()
    low = text.lower()
    if any(tok in low for tok in ("401", "403", "unauthorized", "forbidden", "gated")):
        return (
            f"{text}\n"
            "Hugging Face denied access. These repos are public; "
            "if HF_TOKEN is set, confirm it is valid, then re-run "
            "python scripts/download_models.py."
        )
    if any(
        tok in low
        for tok in (
            "offline",
            "connection",
            "timed out",
            "timeout",
            "name resolution",
            "temporary failure",
            "network is unreachable",
        )
    ):
        return (
            f"{text}\n"
            "Could not reach Hugging Face. Check the network and re-run "
            "python scripts/download_models.py."
        )
    return text


def _fetch(downloader: Downloader, spec: ModelSpec, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if spec.key == "parakeet":
        # Same cache_dir contract as parakeet_mlx.from_pretrained.
        downloader(repo_id=spec.repo_id, cache_dir=str(dest))
        return
    kwargs: dict[str, object] = {"repo_id": spec.repo_id, "local_dir": str(dest)}
    if spec.key == "sensevoice":
        kwargs["allow_patterns"] = list(SENSEVOICE_FILES)
    downloader(**kwargs)


def _hub_downloader() -> Downloader | None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        print(
            "huggingface_hub is required to download weights. "
            "Install project dependencies first: pip install -r requirements.txt "
            f"({exc})",
            file=sys.stderr,
        )
        return None
    return snapshot_download


def download_local_models(
    root: Path,
    downloader: Downloader | None = None,
) -> int:
    """Download any missing local engine. Return 0 when all three are ready."""
    _enable_network()
    fetch = downloader
    if fetch is None:
        fetch = _hub_downloader()
        if fetch is None:
            return 1
    print(
        "Local ASR weights → models/  (about 5 GB; iFlytek is cloud-only and is not downloaded)",
        flush=True,
    )
    for spec in MODELS:
        dest = model_dest(root, spec.key)
        if weights_ready(spec.key, dest):
            print(f"[skip] {spec.title}: already present at {dest}", flush=True)
            continue
        print(
            f"[download] {spec.title} ({spec.approx})",
            flush=True,
        )
        print(f"  repo: {spec.repo_id}", flush=True)
        print(f"  dest: {dest}", flush=True)
        try:
            _fetch(fetch, spec, dest)
        except AssertionError:
            # Test doubles raise this; it must not look like a hub failure.
            raise
        except Exception as exc:
            print(f"[fail] {spec.title}: {_explain(exc)}", file=sys.stderr)
            print(
                "Stopped. Weights already on disk are left in place; "
                "re-run python scripts/download_models.py to continue.",
                file=sys.stderr,
            )
            return 1
        if not weights_ready(spec.key, dest):
            print(
                f"[fail] {spec.title}: download finished but required files "
                f"are missing under {dest}",
                file=sys.stderr,
            )
            return 1
        print(f"[done] {spec.title}", flush=True)
    print("Local ASR weights are ready (Whisper, SenseVoice, Parakeet).", flush=True)
    return 0


def main() -> int:
    return download_local_models(ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
