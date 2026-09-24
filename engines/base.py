"""Shared types for live transcription engines."""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

OnPartial = Callable[[str], None]
OnFinal = Callable[[str], None]


@dataclass
class SessionSummary:
    engine: str
    partial_count: int = 0
    final_count: int = 0
    finals: list[str] = field(default_factory=list)
    error: str | None = None
    # Full-session batch re-decode (post_stop_polish). None when disabled,
    # unsupported, or the session was too short to polish.
    polished: str | None = None
    recorded_sec: float = 0.0
    # Raw session recording (16 kHz mono float32) for audio export (MP3/WAV).
    audio: np.ndarray | None = None
    audio_sample_rate: int = 16000

    def short_text(self) -> str:
        lines = [
            f"Engine: {self.engine}",
            f"Partial updates: {self.partial_count}",
            f"Final segments: {self.final_count}",
        ]
        if self.polished:
            preview = (
                self.polished if len(self.polished) <= 240 else self.polished[:237] + "..."
            )
            lines.append(f"Polished: {preview}")
        elif self.finals:
            joined = " ".join(self.finals).strip()
            preview = joined if len(joined) <= 240 else joined[:237] + "..."
            lines.append(f"Transcript preview: {preview}")
        if self.error:
            lines.append(f"Error: {self.error}")
        return "\n".join(lines)


class LiveEngine(ABC):
    name: str

    @abstractmethod
    def run(
        self,
        on_partial: OnPartial,
        on_final: OnFinal,
        *,
        sample_rate: int = 16000,
        stop_event: threading.Event | None = None,
    ) -> SessionSummary:
        """Block until stop_event / Ctrl+C; print via callbacks; return summary."""

    def polish(
        self, audio: np.ndarray, *, sample_rate: int = 16000
    ) -> str | None:
        """Re-decode a full session recording in one batch pass.

        Receives the raw recorded mic audio (16 kHz mono float32) — everything
        the mic delivered, including audio the live loop gated or dropped —
        and returns polished text, or None when the engine cannot polish.
        Default: unsupported.
        """
        return None
