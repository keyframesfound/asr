"""Shared mic/chunk helpers for local ASR engines.

``prepare_chunk`` is the RMS gate / gentle AGC used before local inference.
``PostFinalCooldown`` drops the audio that would otherwise be reprinted after
``on_final`` (speaker/headphone echo, and the chunk overlap buffer).
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable

import numpy as np

from .config import (
    DEFAULT_MIN_RMS,
    DEFAULT_TARGET_RMS,
    load_audio_config,
    load_config,
)

logger = logging.getLogger("engines.audio_util")

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


# —— full-session polish (batch re-decode on stop) ——

def is_cjk(ch: str) -> bool:
    """True for CJK punctuation / kana / hanzi ranges (shared with the TUI)."""
    o = ord(ch)
    return (
        0x3000 <= o <= 0x303F  # CJK punctuation
        or 0x3040 <= o <= 0x30FF  # kana
        or 0x3400 <= o <= 0x4DBF  # CJK ext A
        or 0x4E00 <= o <= 0x9FFF  # CJK unified
        or 0xF900 <= o <= 0xFAFF  # CJK compatibility
        or 0xFF00 <= o <= 0xFFEF  # fullwidth forms
        or 0x20000 <= o <= 0x2FA1F  # CJK ext B–F
    )


def join_transcript_parts(parts: Iterable[str]) -> str:
    """Join piecewise decode output: no space across a CJK seam, one otherwise."""
    out = ""
    for part in parts:
        p = (part or "").strip()
        if not p:
            continue
        if not out:
            out = p
        elif is_cjk(out[-1]) or is_cjk(p[0]):
            out += p
        else:
            out += " " + p
    return out


def plan_silent_chunks(
    audio: np.ndarray,
    sr: int,
    *,
    max_sec: float,
    min_sec: float = 4.0,
) -> list[tuple[int, int]]:
    """Window ``[0, len)`` into pieces ≤ max_sec that end at the quietest stretch.

    Each boundary snaps to the lowest-RMS 0.25 s window inside the last
    ``min_sec`` seconds of the candidate chunk, so long-session re-decodes do
    not cut words mid-syllable. Pure numpy — no model, no audio leaves range.
    """
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    total = a.size
    if total == 0:
        return []
    frame = max(1, int(0.05 * sr))
    span = max(1, (int(0.25 * sr)) // frame)
    n_frames = total // frame
    if n_frames == 0:
        return [(0, total)]
    frames = a[: n_frames * frame].reshape(n_frames, frame)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    cum = np.concatenate([[0.0], np.cumsum(rms)])
    min_frames = max(1, (int(min_sec * sr)) // frame)
    max_frames = max(min_frames + 1, (int(max_sec * sr)) // frame)

    chunks: list[tuple[int, int]] = []
    pos = 0
    while pos < total:
        if total - pos <= max_frames * frame:
            chunks.append((pos, total))
            break
        hard = pos + max_frames * frame
        lo = (pos + min_frames * frame) // frame
        hi = hard // frame - span
        if hi > lo:
            starts = np.arange(lo, hi)
            vals = (cum[starts + span] - cum[starts]) / span
            boundary = int(starts[int(np.argmin(vals))]) * frame
        else:
            boundary = hard
        chunks.append((pos, int(boundary)))
        pos = int(boundary)
    return chunks


def polish_enabled() -> bool:
    """config.json ``post_stop_polish`` — re-decode the whole session on stop."""
    return bool(load_config().get("post_stop_polish", True))


def polish_session(
    engine,
    blocks: list[np.ndarray],
    *,
    sample_rate: int,
    on_partial: Callable[[str], None],
    summary,
    min_sec: float = 1.0,
) -> None:
    """Best-effort full-session re-decode after a live stop.

    ``blocks`` are the raw mic audio recorded before any gating or echo
    cooldown, so the polish pass sees exactly what a batch dictation app
    would have seen. Failures are logged and the live transcript stands.
    """
    if not polish_enabled() or summary.error or not blocks:
        return
    audio = np.concatenate(
        [np.asarray(b, dtype=np.float32).reshape(-1) for b in blocks]
    )
    recorded_sec = audio.size / float(sample_rate)
    if recorded_sec < min_sec:
        return
    summary.recorded_sec = recorded_sec
    try:
        on_partial("Polishing session audio…")
        text = engine.polish(audio, sample_rate=sample_rate)
        if text and text.strip():
            summary.polished = text.strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("session polish failed (%s); keeping live transcript", exc)
