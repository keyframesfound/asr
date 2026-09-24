"""Parakeet Unified EN (CoreML / Hex FluidAudio) — structural runner."""
from __future__ import annotations

import threading
from pathlib import Path

from .base import LiveEngine, OnFinal, OnPartial, SessionSummary

MODEL_DIR = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "parakeet-unified-en-0.6b-coreml"
)


class ParakeetEngine(LiveEngine):
    name = "Parakeet Unified EN"

    def run(
        self,
        on_partial: OnPartial,
        on_final: OnFinal,
        *,
        sample_rate: int = 16000,
        stop_event: threading.Event | None = None,
    ) -> SessionSummary:
        del on_partial, on_final, sample_rate, stop_event
        summary = SessionSummary(engine=self.name)
        if not MODEL_DIR.exists():
            summary.error = f"Model not found at {MODEL_DIR}."
            return summary

        mlcs = list(MODEL_DIR.rglob("*.mlmodelc"))
        if not mlcs:
            summary.error = f"No .mlmodelc bundles under {MODEL_DIR}."
            return summary

        # Hex / FluidAudio is the intended runtime for these CoreML packages.
        # A pure-Python live path is not shipped with the HF snapshot.
        summary.error = (
            f"Parakeet CoreML weights are present ({len(mlcs)} .mlmodelc bundles at "
            f"{MODEL_DIR}), but live decode needs Apple's Hex / FluidAudio runtime "
            "(not a plain Python pipeline). Use iFlytek, SenseVoice, or Whisper "
            "from this CLI for mic transcription today."
        )
        return summary
