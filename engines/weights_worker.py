"""One-shot model weight downloader, run as a child process (or in-proc).

`python -m engines.weights_worker <code>` downloads one engine's weights —
self-hosted tarballs first (R2, then the GitHub release), Hugging Face as
fallback — printing `progress <bytes>` lines to stdout so a parent can show a
counter.

Stalls are handled by self-terminating with exit code 42: a hub download that
stops moving cannot be cancelled as a thread, and a hung thread holds HF's
file lock forever (every later retry blocks on it). Killing the process is
the only clean cancel — locks die with the process and a retry starts fresh.
"""
from __future__ import annotations

import os
import sys
import threading
import time

from engines import weights as w

# Progress lines are throttled to byte steps; 1 MB keeps the picker's
# percentage ticking smoothly against its 0.5 s UI refresh.
PROGRESS_STEP_BYTES = 1 * 1024 * 1024


def download_sync(code: str, watch: bool = True, on_total=None) -> int:
    """Download one engine's weights. Returns 0 ok, 42 stalled, 1 failed.

    `watch=True` (child mode) prints progress/total lines and hard-exits 42
    when the stream stalls. `watch=False` (in-process test mode) leaves
    progress and stall handling to the caller. Either way, a known expected
    size goes to `on_total(bytes)` (child mode also prints `total <bytes>`).
    """
    repo, path, kind, key = w.HUB_REPOS[code]
    part = path.parent / f"{path.name}.tar.part"
    last = 0
    last_change = time.monotonic()
    done = threading.Event()

    def report_total(n: int) -> None:
        if not n:
            return
        if on_total is not None:
            on_total(int(n))
        if watch:
            print(f"total {int(n)}", flush=True)

    def poll() -> None:
        nonlocal last, last_change
        while not done.wait(1.0):
            n = w.dir_size(path)
            try:
                n += part.stat().st_size
            except OSError:
                pass
            if n - last >= PROGRESS_STEP_BYTES:
                last = n
                print(f"progress {n}", flush=True)
            if n > last:
                last_change = time.monotonic()
            elif time.monotonic() - last_change > w.STALL_TIMEOUT_SEC:
                os._exit(42)

    if watch:
        threading.Thread(target=poll, daemon=True).start()
    try:
        # Self-hosted tarballs first (R2, then the GitHub release; see
        # weights._tarball_bases). Hugging Face LFS (cdn-lfs) is often
        # unreachable or stalls after a few MB of config; only fall through
        # to HF when every tarball source fails or is disabled.
        used_tarball = False
        for base in w._tarball_bases():
            try:
                w._download_and_extract_tarball(
                    f"{base}/{path.name}.tar", path.parent, part, on_total=report_total
                )
                used_tarball = True
                break
            except Exception:
                used_tarball = False
        if not used_tarball:
            w._download_hf(repo, path, kind, on_total=report_total)
        if not w.download_ok(code):
            raise RuntimeError(f"download incomplete: {key or 'weights'} missing or empty")
        if watch:
            print(f"progress {w.dir_size(path)}", flush=True)
        return 0
    except Exception as exc:
        print(f"error {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        done.set()


def main() -> int:
    code = sys.argv[1]
    if code not in w.HUB_REPOS:
        print(f"error unknown engine {code}", file=sys.stderr)
        return 1
    return download_sync(code)


if __name__ == "__main__":
    raise SystemExit(main())
