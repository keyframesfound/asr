"""Whisper Large V3 Turbo — chunked live mic transcription."""
from __future__ import annotations

import os
import queue
import threading
from pathlib import Path

# Textual holds irregular FDs; tqdm→multiprocessing.RLock→spawnv_passfds then
# raises ValueError: bad value(s) in fds_to_keep. Keep all of this in-process.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np

from .base import LiveEngine, OnFinal, OnPartial, SessionSummary
from .mic import open_input_stream

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "whisper-large-v3-turbo"
CHUNK_SEC = 4.0


def _disable_hf_progress() -> None:
    try:
        from transformers.utils.logging import disable_progress_bar

        disable_progress_bar()
    except Exception:
        pass
    try:
        from tqdm import tqdm
        from functools import partialmethod

        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)  # type: ignore[method-assign]
    except Exception:
        pass


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

        _disable_hf_progress()

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float16 if device == "mps" else torch.float32
        on_partial(f"Loading {self.name} on {device}…")

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            str(MODEL_DIR),
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model.to(device)
        model.eval()
        processor = AutoProcessor.from_pretrained(str(MODEL_DIR))

        def transcribe(audio: np.ndarray) -> str:
            inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
            input_features = inputs.input_features.to(device=device, dtype=dtype)
            with torch.inference_mode():
                predicted_ids = model.generate(input_features)
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
                    buf = buf[need // 4 :]
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
