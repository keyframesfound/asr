"""Shared types for live transcription engines."""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable


OnPartial = Callable[[str], None]
OnFinal = Callable[[str], None]


@dataclass
class SessionSummary:
    engine: str
    partial_count: int = 0
    final_count: int = 0
    finals: list[str] = field(default_factory=list)
    error: str | None = None

    def short_text(self) -> str:
        lines = [
            f"Engine: {self.engine}",
            f"Partial updates: {self.partial_count}",
            f"Final segments: {self.final_count}",
        ]
        if self.finals:
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
