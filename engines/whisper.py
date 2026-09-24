"""Whisper Large V3 Turbo — chunked live mic transcription."""
from __future__ import annotations

import os
import queue
import threading
from pathlib import Path

# Avoid tokenizer/fork side channels that break under Textual + PortAudio on macOS.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

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
        stop_event: threading.Event | None = None,
    ) -> SessionSummary:
        summary = SessionSummary(engine=self.name)
        if not MODEL_DIR.exists():
            summary.error = (
                f"Model not found at {MODEL_DIR}. Download whisper-large-v3-turbo first."
            )
            return summary

        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        except ImportError as exc:
            summary.error = f"Missing dependency: {exc}. pip install torch transformers"
            return summary

        # Force spawn so nothing tries to fork while PortAudio holds FDs.
        try:
            import multiprocessing as mp

            if mp.get_start_method(allow_none=True) != "spawn":
                mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float16 if device == "mps" else torch.float32
        on_partial(f"Loading {self.name} on {device}…")

        # Load BEFORE opening the mic so HF/torch setup never races PortAudio FDs.
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            str(MODEL_DIR),
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model.to(device)
        model.eval()
        processor = AutoProcessor.from_pretrained(str(MODEL_DIR))

        def transcribe(audio: np.ndarray) -> str:
            # Direct generate path — no transformers pipeline (avoids worker pools).
            inputs = processor(
                audio,
                sampling_rate=sample_rate,
                return_tensors="pt",
            )
            input_features = inputs.input_features.to(device=device, dtype=dtype)
            with torch.inference_mode():
                predicted_ids = model.generate(
                    input_features,
                    task="transcribe",
                    language=None,
                )
            text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            return (text or "").strip()

        stop = stop_event or threading.Event()
        need = int(CHUNK_SEC * sample_rate)
        buf = np.zeros(0, dtype=np.float32)

        stream, q = open_input_stream(sample_rate=sample_rate, blocksize=3200)
        stream.start()
        on_partial("Listening (Ctrl+C / q to stop)…")
        try:
            while not stop.is_set():
                try:
                    block = q.get(timeout=0.2)
                except queue.Empty:
                    continue
                buf = np.concatenate([buf, block.reshape(-1).astype(np.float32)])
                while len(buf) >= need and not stop.is_set():
                    chunk = buf[:need]
                    buf = buf[need // 4 :]  # ~75% overlap hop
                    try:
                        text = transcribe(chunk)
                    except Exception as exc:  # noqa: BLE001
                        summary.error = str(exc)
                        stop.set()
                        break
                    if text:
                        summary.final_count += 1
                        summary.finals.append(text)
                        on_final(text)
        except KeyboardInterrupt:
            pass
        finally:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
            if len(buf) > sample_rate // 2 and not summary.error:
                try:
                    text = transcribe(buf)
                    if text:
                        summary.final_count += 1
                        summary.finals.append(text)
                        on_final(text)
                except Exception as exc:  # noqa: BLE001
                    summary.error = str(exc)
        return summary
