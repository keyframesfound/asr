#!/usr/bin/env python3
"""Download local ASR weights into ``models/`` (idempotent).

Whisper and SenseVoice prefer the GitHub ``models-v1`` release tarballs
(``whisper-large-v3-turbo.tar``, ``sensevoice-small.tar``). Each archive is
rooted at ``<dirname>/`` and unpacks to ``models/<dirname>/``, the same
layout ``from_pretrained`` / FunASR ``AutoModel`` already load from disk.
Hugging Face LFS (``cdn-lfs.huggingface.co``) can stall after a few MB on
networks where that CDN is unreachable; if the release fetch fails, the
script falls back to ``snapshot_download``.

Parakeet is larger than GitHub's 2 GiB release-asset limit, so it stays on
the Hugging Face hub cache layout that
``parakeet_mlx.from_pretrained(..., cache_dir=models/parakeet-mlx)`` uses.

iFlytek is cloud-only. Hex / CoreML trees are not downloaded.

Re-running skips any engine whose weight files are already present.
"""
from __future__ import annotations

import hashlib
import os
import sys
import tarfile
import urllib.request
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

# GitHub release assets. Parakeet is omitted (over the 2 GiB asset limit).
RELEASE_TAG = "models-v1"
RELEASE_BASE_URL = (
    f"https://github.com/keyframesfound/asr/releases/download/{RELEASE_TAG}"
)
# Pinned to models-v1 SHA256SUMS.txt so a truncated CDN body is not extracted.
RELEASE_SHA256 = {
    f"{WHISPER_DIRNAME}.tar": (
        "3133baf9fd6dd260ec17914974323e9c35697c98fca8f40ea33e8a486e460a59"
    ),
    f"{SENSEVOICE_DIRNAME}.tar": (
        "9b01ea831411f5aade6e9d5dcff69ffaeeee8b6b87be751a09bf7f086f0478d9"
    ),
}

_OFFLINE_ENV = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
_DOWNLOAD_CHUNK = 1024 * 1024
_PROGRESS_EVERY = 64 * 1024 * 1024
_DOWNLOAD_TIMEOUT_SEC = 120.0

Downloader = Callable[..., str]
ReleaseFetcher = Callable[[str, Path], None]


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


def release_asset_url(dirname: str) -> str:
    """URL of a ``models-v1`` tarball named ``<dirname>.tar``."""
    return f"{RELEASE_BASE_URL}/{dirname}.tar"


def _release_dirname(key: str) -> str | None:
    """Directory engines shipped as GitHub release assets. Parakeet has none."""
    if key == "whisper":
        return WHISPER_DIRNAME
    if key == "sensevoice":
        return SENSEVOICE_DIRNAME
    return None


def _remove_incomplete(path: Path) -> None:
    """Drop Hugging Face ``*.incomplete`` markers left by an earlier stall."""
    try:
        markers = list(path.rglob("*.incomplete"))
    except OSError:
        return
    for marker in markers:
        try:
            marker.unlink()
        except OSError:
            pass


def _short_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".strip().splitlines()[0]
    if len(text) > 240:
        return text[:237] + "..."
    return text


def extract_release_archive(archive: Path, dest: Path) -> None:
    """Unpack a ``models-v1`` tarball into ``dest``.

    Members live under ``<dirname>/`` (``dest.name``). That prefix is
    stripped so files land directly in ``dest``. macOS AppleDouble ``._*``
    forks are skipped. A member outside that directory is refused.
    """
    root = dest.name
    prefix = root + "/"
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tar:
        selected: list[tarfile.TarInfo] = []
        for member in tar.getmembers():
            name = member.name[2:] if member.name.startswith("./") else member.name
            if not name or name == root:
                continue
            base = name.rsplit("/", 1)[-1]
            if (
                base.startswith("._")
                or name.startswith("PaxHeader/")
                or "/PaxHeader/" in name
            ):
                continue
            if not name.startswith(prefix):
                raise ValueError(
                    f"release archive member {member.name!r} is outside {root}/"
                )
            rel = name[len(prefix):]
            if not rel or rel.startswith("/") or ".." in Path(rel).parts:
                raise ValueError(f"unsafe archive member {member.name!r}")
            member.name = rel
            selected.append(member)
        if not selected:
            raise ValueError(f"release archive contains no files for {root}/")
        tar.extractall(dest, members=selected, filter="data")


def _stream_download(url: str, dest: Path, timeout: float = _DOWNLOAD_TIMEOUT_SEC) -> str:
    """Stream ``url`` to ``dest``. Return the SHA-256 hex digest."""
    req = urllib.request.Request(url, headers={"User-Agent": "asr-download-models"})
    digest = hashlib.sha256()
    got = 0
    next_report = _PROGRESS_EVERY
    with urllib.request.urlopen(req, timeout=timeout) as resp, dest.open("wb") as out:
        while True:
            chunk = resp.read(_DOWNLOAD_CHUNK)
            if not chunk:
                break
            out.write(chunk)
            digest.update(chunk)
            got += len(chunk)
            if got >= next_report:
                print(f"  received {got // (1024 * 1024)} MB", flush=True)
                next_report += _PROGRESS_EVERY
    return digest.hexdigest()


def fetch_release_tarball(url: str, dest: Path) -> None:
    """Download one release asset, verify its pinned checksum, and extract it.

    The partial file sits inside ``dest`` (gitignored with the rest of the
    model tree) and is removed when the fetch finishes or fails.
    """
    filename = url.rstrip("/").rsplit("/", 1)[-1]
    expected = RELEASE_SHA256.get(filename)
    if not expected:
        raise ValueError(f"no pinned checksum for release asset {filename}")
    dest.mkdir(parents=True, exist_ok=True)
    partial = dest / f".{filename}.partial"
    try:
        print(f"  downloading {filename}", flush=True)
        digest = _stream_download(url, partial)
        if digest != expected:
            raise RuntimeError(
                f"checksum mismatch for {filename}: got {digest}, expected {expected}"
            )
        print(f"  extracting {filename}", flush=True)
        extract_release_archive(partial, dest)
    finally:
        try:
            partial.unlink()
        except OSError:
            pass


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


def _fetch_hub(downloader: Downloader, spec: ModelSpec, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if spec.key == "parakeet":
        # Same cache_dir contract as parakeet_mlx.from_pretrained.
        downloader(repo_id=spec.repo_id, cache_dir=str(dest))
        return
    kwargs: dict[str, object] = {"repo_id": spec.repo_id, "local_dir": str(dest)}
    if spec.key == "sensevoice":
        kwargs["allow_patterns"] = list(SENSEVOICE_FILES)
    downloader(**kwargs)


def _try_release(spec: ModelSpec, dest: Path, release_fetcher: ReleaseFetcher) -> bool:
    """Fetch the GitHub tarball. Return True when the tree is ready to load.

    Any fetch or layout failure is reported and returns False so the caller
    can fall back to Hugging Face. ``AssertionError`` still propagates so a
    test double fails the test instead of looking like a network error.
    """
    dirname = _release_dirname(spec.key)
    if dirname is None:
        return False
    url = release_asset_url(dirname)
    print(f"  release: {url}", flush=True)
    try:
        release_fetcher(url, dest)
    except AssertionError:
        raise
    except Exception as exc:
        print(
            f"[note] GitHub release fetch failed ({_short_error(exc)}); "
            "falling back to Hugging Face",
            flush=True,
        )
        return False
    _remove_incomplete(dest)
    if weights_ready(spec.key, dest):
        return True
    print(
        "[note] GitHub release did not contain the required weight files; "
        "falling back to Hugging Face",
        flush=True,
    )
    return False


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
    *,
    release_fetcher: ReleaseFetcher | None = None,
) -> int:
    """Download any missing local engine. Return 0 when all three are ready.

    ``downloader`` replaces Hugging Face ``snapshot_download`` (tests).
    ``release_fetcher`` replaces the GitHub release download (tests) and is
    called as ``release_fetcher(url, dest)`` for Whisper and SenseVoice.
    """
    _enable_network()
    fetch = downloader
    if fetch is None:
        fetch = _hub_downloader()
        if fetch is None:
            return 1
    fetch_release = release_fetcher or fetch_release_tarball
    print(
        "Local ASR weights → models/  (about 5 GB; iFlytek is cloud-only and is not downloaded)",
        flush=True,
    )
    print(
        "Whisper and SenseVoice prefer the GitHub models-v1 release "
        "(Hugging Face if that fetch fails). Parakeet uses Hugging Face.",
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
        print(f"  dest: {dest}", flush=True)
        dest.mkdir(parents=True, exist_ok=True)
        ready = _try_release(spec, dest, fetch_release)
        if not ready:
            print(f"  repo: {spec.repo_id}", flush=True)
            try:
                _fetch_hub(fetch, spec, dest)
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
