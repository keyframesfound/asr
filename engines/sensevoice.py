"""SenseVoice Small — local FunASR, chunked mic (Cantonese, 香港繁體)."""
from __future__ import annotations

import queue
import time
import re
import threading
from pathlib import Path

import numpy as np

from .audio_util import chunk_rms, prepare_chunk
from .config import load_config
from .base import LiveEngine, OnFinal, OnPartial, SessionSummary
from .mic import open_input_stream

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "sensevoice-small"
CHUNK_SEC = 4.0

# FunASR / SenseVoice language ids (README + webui language_abbr, lid_dict):
# auto, zh, en, yue, ja, ko, nospeech. Cantonese is "yue" (lid 7).
# Older Hugging Face README copies typo Mandarin as "zn"; the model key is "zh".
LANGUAGE = "yue"

_MODEL = None
_HK = None


def unload() -> None:
    """Drop the warm SenseVoice model so another engine can own RAM."""
    global _MODEL
    _MODEL = None


def preload() -> str:
    global _MODEL
    if _MODEL is not None:
        return "already loaded"
    from funasr import AutoModel

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Missing model dir: {MODEL_DIR}. Run: python scripts/download_models.py"
        )
    _MODEL = AutoModel(model=str(MODEL_DIR), disable_update=True, device="cpu")
    return f"loaded from {MODEL_DIR.name}"


_TAG_RE = re.compile(r"<\|[^|>]+\|>")


def _hk_converter():
    """OpenCC ``s2hk``: Simplified Chinese → 香港繁體."""
    global _HK
    if _HK is None:
        try:
            import opencc
        except ImportError as exc:
            raise ImportError(
                "SenseVoice needs OpenCC for 香港繁體 (s2hk). "
                "Install with: pip install opencc"
            ) from exc
        _HK = opencc.OpenCC("s2hk")
    return _HK


def _to_hk_traditional(text: str) -> str:
    if not text:
        return text
    return _hk_converter().convert(text)


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
    return _to_hk_traditional(text)


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
            summary.error = (
                f"Model not found at {MODEL_DIR}. "
                "Run: python scripts/download_models.py"
            )
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
            _hk_converter()
        except ImportError as exc:
            summary.error = str(exc)
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
        cfg = load_config()
        cooldown_sec = float(cfg.get("post_final_cooldown_sec", 1.25))
        blocksize = int(cfg.get("mic_blocksize", 4096))
        min_rms = float(cfg.get("vad_min_rms", 0.02))

        def transcribe(audio: np.ndarray) -> str:
            audio = prepare_chunk(audio, min_rms=min_rms)
            if audio is None:
                return ""
            out = model.generate(input=audio, cache={}, language=LANGUAGE, use_itn=True)
            return _extract_text(out)

        stop = stop_event or threading.Event()
        need = int(CHUNK_SEC * sample_rate)
        buf = np.zeros(0, dtype=np.float32)
        cool_until = 0.0
        stream, q = open_input_stream(sample_rate=sample_rate, blocksize=blocksize)
        stream.start()
        on_partial("Listening…")
        signaled_transcribing = False
        try:
            while not stop.is_set():
                try:
                    block = q.get(timeout=0.2)
                except queue.Empty:
                    continue
                now = time.monotonic()
                if now < cool_until:
                    buf = np.zeros(0, dtype=np.float32)
                    signaled_transcribing = False
                    continue
                buf = np.concatenate([buf, block.reshape(-1).astype(np.float32)])
                while len(buf) >= need and not stop.is_set():
                    if time.monotonic() < cool_until:
                        buf = np.zeros(0, dtype=np.float32)
                        break
                    chunk = buf[:need]
                    buf = buf[need // 4 :]
                    if chunk_rms(chunk) < min_rms:
                        continue
                    if not signaled_transcribing:
                        signaled_transcribing = True
                        on_partial("Transcribing…")
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
                        cool_until = time.monotonic() + cooldown_sec
                        buf = np.zeros(0, dtype=np.float32)
                        signaled_transcribing = False
                        on_partial("Listening…")
                    else:
                        signaled_transcribing = False
                        on_partial("Listening…")
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
