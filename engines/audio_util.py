"""Shared mic/chunk helpers for local ASR engines.

``prepare_chunk`` is the RMS gate / gentle AGC used before local inference.
``PostFinalCooldown`` drops the audio that would otherwise be reprinted after
``on_final`` (speaker/headphone echo, and the chunk overlap buffer).
"""
from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from .config import (
    DEFAULT_MIN_RMS,
    DEFAULT_TARGET_RMS,
    load_audio_config,
)

# Public aliases. Prefer config.json; these match the built-in fallbacks.
MIN_RMS = DEFAULT_MIN_RMS
TARGET_RMS = DEFAULT_TARGET_RMS

# Cap applied when lifting quiet speech toward target_rms.
# The previous gate could multiply near-silence by ~50x and turn echo into speech.
MAX_GAIN = 4.0


def chunk_rms(audio: np.ndarray) -> float:
    a = audio.reshape(-1).astype(np.float64)
    if a.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(a))))


def prepare_chunk(
    audio: np.ndarray,
    *,
    min_rms: float | None = None,
    target_rms: float | None = None,
) -> np.ndarray | None:
    """Return float32 mono audio, or None if too quiet to bother.

    ``min_rms`` / ``target_rms`` default to config.json (safe built-ins if the
    file is missing). Gain toward ``target_rms`` never exceeds ``MAX_GAIN``.
    """
    cfg = load_audio_config()
    if min_rms is None:
        min_rms = cfg.min_rms
    if target_rms is None:
        target_rms = cfg.target_rms

    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    if a.size == 0:
        return None
    rms = chunk_rms(a)
    if rms < min_rms:
        return None
    if rms < target_rms:
        gain = target_rms / max(rms, 1e-8)
        if gain > MAX_GAIN:
            gain = MAX_GAIN
        if gain > 1.0:
            a = np.clip(a * np.float32(gain), -1.0, 1.0)
    return a


class PostFinalCooldown:
    """Skip mic audio for ``cooldown_ms`` after a committed final.

    A second ``arm`` during an open window does not extend it, so a burst of
    streaming deltas only opens one gap. ``cooldown_ms`` of 0 disables the drop.
    """

    def __init__(
        self,
        cooldown_ms: int | None = None,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if cooldown_ms is None:
            cooldown_ms = load_audio_config().cooldown_ms
        self.cooldown_ms = max(0, int(cooldown_ms))
        self._clock = clock or time.monotonic
        self._until = 0.0

    def arm(self) -> None:
        if self.cooldown_ms <= 0:
            self._until = 0.0
            return
        now = self._clock()
        if now < self._until:
            return
        self._until = now + (self.cooldown_ms / 1000.0)

    def active(self) -> bool:
        if self.cooldown_ms <= 0 or self._until <= 0.0:
            return False
        return self._clock() < self._until

    def gate(self, audio: np.ndarray, *, keep_loud: bool = False) -> np.ndarray | None:
        """``prepare_chunk`` result, or None when this audio should be skipped.

        Chunked engines leave ``keep_loud`` false and drop the whole cooldown
        window. Streaming engines that emit a final on every hypothesis update
        pass ``keep_loud=True``: blocks already at ``target_rms`` (live speech)
        still pass, quieter echo does not.
        """
        if self.active():
            if not keep_loud:
                return None
            cfg = load_audio_config()
            if chunk_rms(audio) < cfg.target_rms:
                return None
        return prepare_chunk(audio)


def accept_input_block(
    buf: np.ndarray,
    block: np.ndarray,
    cooldown: PostFinalCooldown,
) -> np.ndarray:
    """Append a mic block, or discard it and any overlap while cooling down.

    Clearing the overlap buffer is what stops the same utterance from being
    transcribed again on the next chunk.
    """
    if cooldown.active():
        return np.zeros(0, dtype=np.float32)
    flat = np.asarray(block, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return buf
    if buf.size == 0:
        return np.ascontiguousarray(flat)
    return np.concatenate([buf, flat])
