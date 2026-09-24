"""Parakeet Unified EN — Apple Silicon via parakeet-mlx (streaming)."""
from __future__ import annotations

import queue
import threading
from pathlib import Path

import numpy as np

from .audio_util import prepare_chunk
from .base import LiveEngine, OnFinal, OnPartial, SessionSummary
from .mic import open_input_stream

# MLX weights (downloaded on first use). CoreML Hex bundles under
# models/parakeet-unified-en-0.6b-coreml remain available for Hex/FluidAudio.
MODEL_ID = "mlx-community/parakeet-tdt-0.6b-v3"
CACHE_DIR = Path(__file__).resolve().parents[1] / "models" / "parakeet-mlx"
COREML_DIR = (
    Path(__file__).resolve().parents[1] / "models" / "parakeet-unified-en-0.6b-coreml"
)

_MODEL = None


def preload() -> str:
    global _MODEL
    if _MODEL is not None:
        return f"already loaded ({MODEL_ID})"
    from parakeet_mlx import from_pretrained

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _MODEL = from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    return f"loaded {MODEL_ID}"


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
        summary = SessionSummary(engine=self.name)

        try:
            import mlx.core as mx
            from parakeet_mlx import from_pretrained  # noqa: F401
        except ImportError:
            n = len(list(COREML_DIR.rglob("*.mlmodelc"))) if COREML_DIR.exists() else 0
            summary.error = (
                "Parakeet needs parakeet-mlx (Apple Silicon). "
                "Install with: pip install parakeet-mlx"
                + (f" — CoreML Hex weights also present ({n} bundles)." if n else "")
            )
            return summary

        try:
            on_partial("Loading Parakeet (MLX)…")
            info = preload()
            on_partial(f"Ready ({info})")
        except Exception as exc:
            summary.error = f"Model load failed: {exc}"
            return summary

        assert _MODEL is not None
        model = _MODEL
        stop = stop_event or threading.Event()
        stream_mic, q = open_input_stream(sample_rate=sample_rate, blocksize=3200)
        stream_mic.start()
        on_partial("Listening…")
        last_text = ""

        try:
            with model.transcribe_stream() as asr:
                while not stop.is_set():
                    try:
                        block = q.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    audio = prepare_chunk(block, min_rms=0.0005)
                    if audio is None:
                        continue
                    # parakeet-mlx expects float32 samples at 16 kHz
                    asr.add_audio(mx.array(audio))
                    text = (asr.result.text or "").strip()
                    if text and text != last_text:
                        # Emit only the newly appended portion when possible
                        if last_text and text.startswith(last_text):
                            delta = text[len(last_text) :].strip()
                        else:
                            delta = text
                        last_text = text
                        if delta:
                            summary.final_count += 1
                            summary.finals.append(delta)
                            on_final(delta)
                        else:
                            on_partial(text)
        except KeyboardInterrupt:
            pass
        except Exception as exc:
            summary.error = str(exc)
        finally:
            try:
                stream_mic.stop()
                stream_mic.close()
            except Exception:
                pass
            if last_text and not summary.finals:
                summary.final_count += 1
                summary.finals.append(last_text)
                on_final(last_text)
        return summary
