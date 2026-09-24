"""Live mic level bus — the sounddevice callback pushes, the UI polls.

Kept dependency-free so the TUI can read levels without importing
sounddevice. Peak falls linearly so a short burst stays visible; ``ts``
lets the UI treat a silent stream (no callbacks) as idle.
"""
from __future__ import annotations

import threading
import time

# Peak hold: seconds for a full-scale peak to decay back to the live RMS.
_PEAK_FALL_PER_SEC = 0.45
# A touched full-scale sample lights the clip lamp for this long.
_CLIP_HOLD_SEC = 0.6

_lock = threading.Lock()
_rms = 0.0
_peak = 0.0
_ts = 0.0
_clip_until = 0.0


def update_level(rms: float, peak_sample: float, *, now: float | None = None) -> None:
    """Record one block's RMS (energy) and peak sample (0–1 float32 scale)."""
    global _rms, _peak, _ts, _clip_until
    t = time.monotonic() if now is None else now
    with _lock:
        dt = 0.0 if _ts == 0.0 else max(0.0, t - _ts)
        fell = _peak - _PEAK_FALL_PER_SEC * dt
        _peak = max(fell, float(peak_sample), float(rms))
        _rms = max(0.0, min(1.0, float(rms)))
        _ts = t
        if peak_sample >= 0.99:
            _clip_until = t + _CLIP_HOLD_SEC


def snapshot(*, now: float | None = None) -> tuple[float, float, float, bool]:
    """Return (rms, decaying peak, last-update monotonic ts, clipped)."""
    t = time.monotonic() if now is None else now
    with _lock:
        return _rms, _peak, _ts, t < _clip_until


def reset_level() -> None:
    """Silence the meter (stream closed / session stopped)."""
    global _rms, _peak, _ts, _clip_until
    with _lock:
        _rms = 0.0
        _peak = 0.0
        _ts = 0.0
        _clip_until = 0.0
