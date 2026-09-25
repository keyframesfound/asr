"""Local model weight management — status, download, uninstall.

Backs the model picker: the three local engines keep weights under models/
(gitignored); iFlytek is cloud-only and has no weights. Download layout
mirrors how each engine loads:

- sensevoice / whisper: plain snapshot into models/<name> (loaded with
  ``AutoModel(dir)`` / ``from_pretrained(dir)``)
- parakeet: HF hub cache layout under models/parakeet-mlx (loaded with
  ``parakeet_mlx.from_pretrained(..., cache_dir=...)``)

Downloads run on a daemon thread; the UI polls :func:`download_state` for
byte progress. ``snapshot_download`` itself cannot be cancelled mid-flight —
a finished-but-unwanted download can simply be uninstalled.
"""
from __future__ import annotations

import os
import shutil
import tarfile
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path

# Reliable plain-HTTP transfer when this module is imported before
# huggingface_hub (main.py sets the same thing even earlier).
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
# tqdm writes its bar to stderr; the download child's stderr is a pipe nobody
# reads until exit, and a full pipe blocks the child mid-download.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# Self-hosted weights on GitHub Releases (see models-v1): tried first for the
# plain-directory engines, no login on either side. Override or disable with
# ASR_WEIGHTS_BASE_URL (set it to "hf" or "off" to skip GitHub and use HF).
DEFAULT_RELEASE_BASE = "https://github.com/keyframesfound/asr/releases/download/models-v1"

# No byte growth for this long while "downloading" → flag a stall and let the
# user retry (the stalled socket itself cannot be cancelled cleanly).
STALL_TIMEOUT_SEC = 45


def _release_base() -> str:
    return os.environ.get("ASR_WEIGHTS_BASE_URL", DEFAULT_RELEASE_BASE).strip()

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

# code → (HF repo, on-disk location, layout kind, file that must exist)
# The key file is the actual weights blob: config-only leftovers (e.g. a
# half-finished download) must not count as installed.
HUB_REPOS: dict[str, tuple[str, Path, str, str]] = {
    "sensevoice": ("FunAudioLLM/SenseVoiceSmall", MODELS_DIR / "sensevoice-small", "dir", "model.pt"),
    "whisper": ("openai/whisper-large-v3-turbo", MODELS_DIR / "whisper-large-v3-turbo", "dir", "model.safetensors"),
    "parakeet": ("mlx-community/parakeet-tdt-0.6b-v3", MODELS_DIR / "parakeet-mlx", "hub-cache", ""),
}


def is_local_engine(code: str) -> bool:
    return code in HUB_REPOS


def weights_present(code: str) -> bool:
    """True when the engine's weights are on disk (cloud engines: always True)."""
    entry = HUB_REPOS.get(code)
    if entry is None:
        return True
    _, path, kind, key_file = entry
    if kind == "hub-cache":
        try:
            from engines.parakeet import weights_cached

            return bool(weights_cached())
        except Exception:
            return _hub_cache_ok(path, entry[0])
    return path.is_dir() and (path / key_file).is_file() and (path / key_file).stat().st_size > 0


def human_size(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num) < 1024 or unit == "GB":
            return f"{num:.1f} {unit}" if unit != "B" else f"{int(num)} B"
        num /= 1024
    return f"{num:.1f} GB"


def human_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"


SPEED_WINDOW_SEC = 6.0


class _Meter:
    """Byte-rate over a sliding window of (time, bytes) progress samples."""

    def __init__(self) -> None:
        self._samples: deque[tuple[float, int]] = deque()

    def feed(self, n: int) -> float:
        now = time.monotonic()
        self._samples.append((now, n))
        while len(self._samples) > 2 and now - self._samples[0][0] > SPEED_WINDOW_SEC:
            self._samples.popleft()
        t0, n0 = self._samples[0]
        dt = now - t0
        return max(0.0, (n - n0) / dt) if dt > 0 else 0.0


def dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total


def weights_size(code: str) -> int:
    """Bytes on disk for the engine's weights (0 for cloud engines)."""
    entry = HUB_REPOS.get(code)
    if entry is None:
        return 0
    path = entry[1]
    return dir_size(path) if path.exists() else 0


@dataclass
class DownloadState:
    running: bool = False
    downloaded: int = 0
    error: str = ""
    finished_at: float = 0.0
    total: int = 0  # expected full size in bytes; 0 = unknown
    speed: float = 0.0  # bytes/sec over the recent window; 0 = not yet known


_lock = threading.Lock()
_states: dict[str, DownloadState] = {}
_meters: dict[str, "_Meter"] = {}


def download_state(code: str) -> DownloadState:
    with _lock:
        state = _states.get(code)
        return DownloadState(**vars(state)) if state else DownloadState()


def _note_progress(code: str, n: int) -> None:
    """Record byte progress and refresh the speed estimate. Single writer."""
    with _lock:
        state = _states[code]
        state.downloaded = n
        state.speed = _meters.setdefault(code, _Meter()).feed(n)


def _set_total(code: str, n: int) -> None:
    if n > 0:
        with _lock:
            _states[code].total = n


def download_weights_async(code: str) -> bool:
    """Start downloading in the background. False when already running."""
    if code not in HUB_REPOS:
        raise ValueError(f"{code} has no local weights to download")
    with _lock:
        state = _states.setdefault(code, DownloadState())
        if state.running:
            return False
        state.running = True
        state.error = ""
        state.downloaded = 0
        state.total = 0
        state.speed = 0.0
        _meters.pop(code, None)
    threading.Thread(target=_download_worker, args=(code,), daemon=True, name=f"dl-{code}").start()
    return True


def _download_worker(code: str) -> None:
    """Manager thread: run the downloader child, surface progress and errors.

    The child self-terminates with exit 42 when its stream stalls, so HF file
    locks die with the process and pressing i again starts a clean retry.
    Tests set ASR_WEIGHTS_INPROC=1 to run the same logic in this thread.
    """
    _, path, _kind, _key = HUB_REPOS[code]

    def finish(error: str = "") -> None:
        with _lock:
            state = _states[code]
            state.running = False
            state.error = error
            state.finished_at = time.time()
            state.downloaded = dir_size(path)
            state.speed = 0.0

    if os.environ.get("ASR_WEIGHTS_INPROC") == "1":
        from engines import weights_worker as worker

        stop = threading.Event()

        def poll() -> None:
            """Test-mode watchdog (the child can't self-exit in-process)."""
            last = -1
            last_change = time.monotonic()
            while not stop.wait(0.5):
                n = dir_size(path)
                with _lock:
                    state = _states[code]
                    if not state.running:
                        return
                _note_progress(code, n)
                if n != last:
                    last, last_change = n, time.monotonic()
                elif time.monotonic() - last_change > STALL_TIMEOUT_SEC:
                    with _lock:
                        state = _states[code]
                        if state.running:
                            state.running = False
                            state.error = (
                                f"stalled — no progress for {STALL_TIMEOUT_SEC}s; press i to retry"
                            )
                    stop.set()

        poller = threading.Thread(target=poll, daemon=True)
        poller.start()
        try:
            worker.download_sync(
                code, watch=False, on_total=lambda n: _set_total(code, n)
            )
            if download_ok(code):
                finish("")
            else:
                finish("download incomplete: weights failed verification")
        except Exception as exc:
            with _lock:
                stalled = bool(_states[code].error)
            finish("" if stalled else str(exc))
        finally:
            stop.set()
        return

    try:
        import subprocess
        import sys

        # This route to HF drops connections every ~100–200 MB; the child
        # resumes from partial files, so just respawn on stall (exit 42).
        rc = 1
        err = ""
        for _attempt in range(3):
            proc = subprocess.Popen(
                [sys.executable, "-m", "engines.weights_worker", code],
                cwd=str(MODELS_DIR.parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            # The child's stderr is never parsed here, but a pipe nobody reads
            # fills up (tracebacks, library chatter) and blocks the child
            # mid-download. Drain it on a side thread instead.
            errbuf: list[str] = []
            errdrain = threading.Thread(
                target=lambda: errbuf.append(proc.stderr.read() or ""), daemon=True
            )
            errdrain.start()
            for line in proc.stdout or []:
                parts = line.split()
                if len(parts) == 2 and parts[0] == "progress":
                    _note_progress(code, int(parts[1]))
                elif len(parts) == 2 and parts[0] == "total":
                    _set_total(code, int(parts[1]))
            rc = proc.wait()
            errdrain.join(timeout=5)
            err = (errbuf[0] if errbuf else "").strip()
            if rc == 0 and download_ok(code):
                finish("")
                return
            if rc != 42:
                break
            time.sleep(2)  # fresh connection; resume picks up where it died
        if rc == 42:
            finish(err or f"stalled — no progress for {STALL_TIMEOUT_SEC}s; press i to retry")
        else:
            finish(err.splitlines()[-1] if err else f"downloader exited {rc}")
    except Exception as exc:
        finish(str(exc))


def _key_ok(path: Path, key: str) -> bool:
    f = path / key
    return f.is_file() and f.stat().st_size > 0


# File extensions that count as real weights when verifying a hub cache.
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")


def _hub_cache_ok(path: Path, repo: str) -> bool:
    """A hub cache is complete only with config + a non-empty weights blob."""
    repo_dir = path / ("models--" + repo.replace("/", "--"))
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir() or not any(snapshots.glob("*/config.json")):
        return False
    for f in snapshots.glob("*/*"):
        if f.suffix in _WEIGHT_SUFFIXES:
            try:
                if f.stat().st_size > 0:  # stat follows the symlink into blobs
                    return True
            except OSError:
                continue
    return False


def download_ok(code: str) -> bool:
    """Post-download verification: the engine's actual weights are on disk."""
    entry = HUB_REPOS.get(code)
    if entry is None:
        return False
    _, path, kind, key = entry
    if kind == "hub-cache":
        return _hub_cache_ok(path, entry[0])
    return _key_ok(path, key)


def _download_and_extract_tarball(
    url: str, dest_parent: Path, part: Path, on_total=None
) -> None:
    """Stream a release .tar (containing <name>/) and unpack it atomically-ish.

    Any failure raises; the caller falls back to the HF hub. A leftover
    .part file from a killed run is simply re-downloaded over.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "asr-weights"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        if on_total is not None:
            try:
                on_total(int(resp.headers.get("Content-Length") or 0))
            except (TypeError, ValueError):
                pass
        with open(part, "wb") as fh:
            shutil.copyfileobj(resp, fh, length=1 << 20)
    with tarfile.open(part) as tf:
        tf.extractall(dest_parent, filter="data")
    part.unlink(missing_ok=True)


def _hf_total_bytes(repo: str) -> int:
    """Best-effort full size of a repo's files — 0 when it can't be known."""
    try:
        from huggingface_hub import HfApi

        info = HfApi(timeout=10).model_info(repo_id=repo, files_metadata=True)
        return sum(s.size or 0 for s in info.siblings or [])
    except Exception:
        return 0


def _download_hf(repo: str, path: Path, kind: str, on_total=None) -> None:
    if on_total is not None:
        total = _hf_total_bytes(repo)
        if total:
            on_total(total)
    from huggingface_hub import snapshot_download

    path.mkdir(parents=True, exist_ok=True)
    if kind == "dir":
        snapshot_download(repo_id=repo, local_dir=path)
    else:
        snapshot_download(repo_id=repo, cache_dir=path)


def remove_weights(code: str) -> str:
    """Delete the engine's weights. Returns the freed size, human readable."""
    entry = HUB_REPOS.get(code)
    if entry is None:
        raise ValueError(f"{code} is cloud-only — no local weights to remove")
    _, path, kind, _key = entry
    if kind == "hub-cache" and not path.exists():
        try:
            from engines.parakeet import weights_cached

            if not weights_cached():
                return "0 B"
        except Exception:
            return "0 B"
    freed = dir_size(path) if path.exists() else 0
    shutil.rmtree(path, ignore_errors=True)
    return human_size(freed)
