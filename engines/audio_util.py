"""Shared mic/chunk helpers for local ASR engines."""
from __future__ import annotations

import numpy as np

# Skip near-silence; boost quiet mics toward TARGET_RMS.
# Whisper overrides min_rms higher to cut ambient / self-echo ghosts.
MIN_RMS = 0.008
TARGET_RMS = 0.05


def chunk_rms(audio: np.ndarray) -> float:
    a = audio.reshape(-1).astype(np.float64)
    if a.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(a))))


def prepare_chunk(
    audio: np.ndarray,
    *,
    min_rms: float = MIN_RMS,
    target_rms: float = TARGET_RMS,
) -> np.ndarray | None:
    """Return float32 mono audio, or None if too quiet to bother."""
    a = audio.reshape(-1).astype(np.float32)
    rms = chunk_rms(a)
    if rms < min_rms:
        return None
    if rms < target_rms:
        a = a * np.float32(target_rms / max(rms, 1e-8))
        a = np.clip(a, -1.0, 1.0)
    return a
