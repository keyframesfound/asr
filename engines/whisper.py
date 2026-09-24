"""Whisper Large V3 Turbo — chunked live mic transcription."""
from __future__ import annotations

import queue
import threading
from pathlib import Path

import numpy as np

from .base import LiveEngine, OnFinal, OnPartial, SessionSummary
from .mic import open_input_stream

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "whisper-large-v3-turbo"
CHUNK_SEC = 4.0


class WhisperTurboEngine(LiveEngine):
    name = "Whisper Large V3 Turbo"

    def run(
        self,
        on_partial: OnPartial,
        on_final: OnFinal,
        *,
        sample_rate: int = 16000,
    ) -> SessionSummary:
        summary = SessionSummary(engine=self.name)
        if not MODEL_DIR.exists():
            summary.error = f"Model not found at {MODEL_DIR}. Download whisper-large-v3-turbo first."
            return summary

        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        except ImportError as exc:
            summary.error = f"Missing dependency: {exc}. pip install torch transformers"
            return summary

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float16 if device == "mps" else torch.float32
        on_partial(f"Loading {self.name} on {device}…")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            str(MODEL_DIR), torch_dtype=dtype, low_cpu_mem_usage=True
        )
        model.to(device)
        processor = AutoProcessor.from_pretrained(str(MODEL_DIR))
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=dtype,
            device=device,
        )

        stop = threading.Event()
        need = int(CHUNK_SEC * sample_rate)
        buf = np.zeros(0, dtype=np.float32)

        stream, q = open_input_stream(sample_rate=sample_rate, blocksize=3200)
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
                    buf = buf[need // 4 :]  # 75% overlap-ish hop
                    result = pipe(chunk, generate_kwargs={"task": "transcribe"})
                    text = (result.get("text") or "").strip()
                    if text:
                        summary.final_count += 1
                        summary.finals.append(text)
                        on_final(text)
        except KeyboardInterrupt:
            pass
        finally:
            stream.stop()
            stream.close()
            if len(buf) > sample_rate // 2:
                try:
                    result = pipe(buf, generate_kwargs={"task": "transcribe"})
                    text = (result.get("text") or "").strip()
                    if text:
                        summary.final_count += 1
                        summary.finals.append(text)
                        on_final(text)
                except Exception as exc:  # noqa: BLE001
                    summary.error = str(exc)
        return summary
