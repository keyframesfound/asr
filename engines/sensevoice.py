"""SenseVoice Small — local FunASR, chunked mic."""
from __future__ import annotations

import queue
import re
import threading
from pathlib import Path

import numpy as np

from .audio_util import prepare_chunk
from .base import LiveEngine, OnFinal, OnPartial, SessionSummary
from .mic import open_input_stream

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "sensevoice-small"
CHUNK_SEC = 4.0

_MODEL = None


def preload() -> str:
    global _MODEL
    if _MODEL is not None:
        return "already loaded"
    from funasr import AutoModel

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Missing model dir: {MODEL_DIR}")
    _MODEL = AutoModel(model=str(MODEL_DIR), disable_update=True, device="cpu")
    return f"loaded from {MODEL_DIR.name}"


_TAG_RE = re.compile(r"<\|[^|>]+\|>")


def _extract_text(out) -> str:
    if isinstance(out, list) and out:
        item = out[0]
        if isinstance(item, dict):
            text = str(item.get("text") or "").strip()
        else:
            text = str(item).strip()
    elif isinstance(out, dict):
        text = str(out.get("text") or "").strip()
    else:
        text = str(out or "").strip()
    text = _TAG_RE.sub("", text).strip(" ,")
    return text


class SenseVoiceEngine(LiveEngine):
    name = "SenseVoice"

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
            summary.error = f"Model not found at {MODEL_DIR}."
            return summary

        try:
            from funasr import AutoModel  # noqa: F401
        except ImportError:
            summary.error = (
                "SenseVoice needs funasr. Install with: pip install funasr "
                f"(weights already at {MODEL_DIR})."
            )
            return summary

        try:
            on_partial("Loading SenseVoice…")
            info = preload()
            on_partial(f"Ready ({info})")
        except Exception as exc:
            summary.error = f"Model load failed: {exc}"
            return summary

        assert _MODEL is not None
        model = _MODEL

        def transcribe(audio: np.ndarray) -> str:
            audio = prepare_chunk(audio)
            if audio is None:
                return ""
            out = model.generate(input=audio, cache={}, language="auto", use_itn=True)
            return _extract_text(out)

        stop = stop_event or threading.Event()
        need = int(CHUNK_SEC * sample_rate)
        buf = np.zeros(0, dtype=np.float32)
        stream, q = open_input_stream(sample_rate=sample_rate)
        stream.start()
        on_partial("Listening…")
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
                    except Exception as exc:
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
                except Exception as exc:
                    summary.error = str(exc)
        return summary
