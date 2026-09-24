"""SenseVoice Small — local FunASR when available; chunked mic."""
from __future__ import annotations

import queue
import threading
from pathlib import Path

import numpy as np

from .base import LiveEngine, OnFinal, OnPartial, SessionSummary
from .mic import open_input_stream

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "sensevoice-small"
CHUNK_SEC = 5.0


class SenseVoiceEngine(LiveEngine):
    name = "SenseVoice"

    def run(
        self,
        on_partial: OnPartial,
        on_final: OnFinal,
        *,
        sample_rate: int = 16000,
    ) -> SessionSummary:
        summary = SessionSummary(engine=self.name)
        if not MODEL_DIR.exists():
            summary.error = f"Model not found at {MODEL_DIR}."
            return summary

        try:
            from funasr import AutoModel
        except ImportError:
            summary.error = (
                "SenseVoice needs funasr. Install with: pip install funasr "
                f"(weights already at {MODEL_DIR})."
            )
            return summary

        on_partial("Loading SenseVoice…")
        model = AutoModel(model=str(MODEL_DIR), disable_update=True, device="cpu")

        stop = threading.Event()
        need = int(CHUNK_SEC * sample_rate)
        buf = np.zeros(0, dtype=np.float32)
        stream, q = open_input_stream(sample_rate=sample_rate)
        stream.start()
        on_partial("Listening (Ctrl+C to stop)…")
        try:
            while not stop.is_set():
                try:
                    block = q.get(timeout=0.2)
                except queue.Empty:
                    continue
                buf = np.concatenate([buf, block.reshape(-1).astype(np.float32)])
                while len(buf) >= need:
                    chunk = buf[:need]
                    buf = buf[need:]
                    out = model.generate(input=chunk, cache={}, language="auto", use_itn=True)
                    text = ""
                    if isinstance(out, list) and out:
                        text = str(out[0].get("text") or out[0]).strip()
                    elif isinstance(out, dict):
                        text = str(out.get("text") or "").strip()
                    if text:
                        summary.final_count += 1
                        summary.finals.append(text)
                        on_final(text)
        except KeyboardInterrupt:
            pass
        finally:
            stream.stop()
            stream.close()
        return summary
