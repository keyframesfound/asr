"""Whisper Large V3 Turbo — chunked live mic transcription."""
from __future__ import annotations

import os
import queue
import threading
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TQDM_DISABLE"] = "1"

import numpy as np

from .audio_util import chunk_rms, prepare_chunk
from .config import load_config
from .base import LiveEngine, OnFinal, OnPartial, SessionSummary
from .mic import open_input_stream

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "whisper-large-v3-turbo"
CHUNK_SEC = 4.0
# Higher than other engines — Whisper hallucinates on room tone / bleed.
WHISPER_MIN_RMS = 0.02
# After a committed line, ignore mic audio briefly so speaker/AEC bleed
# and overlapping hops cannot re-transcribe the same utterance.
POST_FINAL_COOLDOWN_SEC = 1.5
JUNK = {
    "",
    "you",
    "thank you",
    "thanks for watching",
    ".",
    "...",
    "字幕",
    "字幕by",
    "thanks",
    "thank you for watching",
    "please subscribe",
}

_PATCHED = False
_MODEL = None
_PROCESSOR = None
_DEVICE = None
_DTYPE = None


def _patch_tqdm_no_mp() -> None:
    """Stop tqdm from creating a multiprocessing RLock (breaks under Textual)."""
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True
    try:
        from transformers.utils.logging import disable_progress_bar

        disable_progress_bar()
    except Exception:
        pass

    import threading as _threading

    try:
        import tqdm.std as tqdm_std

        def _noop_create_mp_lock(cls):
            return None

        tqdm_std.tqdm.create_mp_lock = classmethod(_noop_create_mp_lock)
        tqdm_std.tqdm.mp_lock = None
        tqdm_std.tqdm._lock = _threading.RLock()

        @classmethod
        def _get_lock(cls):
            if getattr(cls, "_lock", None) is None:
                cls._lock = _threading.RLock()
            return cls._lock

        tqdm_std.tqdm.get_lock = _get_lock
    except Exception:
        pass

    try:
        from functools import partialmethod
        from tqdm import tqdm

        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
    except Exception:
        pass


def unload() -> None:
    """Drop the warm Whisper model so another engine can own RAM/GPU."""
    global _MODEL, _PROCESSOR, _DEVICE, _DTYPE
    _MODEL = None
    _PROCESSOR = None
    _DEVICE = None
    _DTYPE = None


def preload() -> str:
    """Load model on the calling thread (prefer main, before Textual)."""
    global _MODEL, _PROCESSOR, _DEVICE, _DTYPE
    _patch_tqdm_no_mp()
    if _MODEL is not None:
        return f"already on {_DEVICE}"

    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Missing model dir: {MODEL_DIR}. Run: python scripts/download_models.py"
        )

    _DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    _DTYPE = torch.float16 if _DEVICE == "mps" else torch.float32
    _MODEL = AutoModelForSpeechSeq2Seq.from_pretrained(
        str(MODEL_DIR),
        dtype=_DTYPE,
        low_cpu_mem_usage=False,
    )
    _MODEL.to(_DEVICE)
    _MODEL.eval()
    _PROCESSOR = AutoProcessor.from_pretrained(str(MODEL_DIR))
    return f"loaded on {_DEVICE}"


def _is_junk(text: str) -> bool:
    t = text.strip().lower().strip(" .,!?")
    return t in JUNK or len(t) <= 1


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
                f"Model not found at {MODEL_DIR}. "
                "Run: python scripts/download_models.py"
            )
            return summary

        try:
            import torch
        except ImportError as exc:
            summary.error = f"Missing dependency: {exc}. pip install torch transformers"
            return summary

        try:
            on_partial(f"Loading {self.name}…")
            info = preload()
            on_partial(f"Ready ({info})")
        except Exception as exc:
            summary.error = f"Model load failed: {exc}"
            return summary

        assert _MODEL is not None and _PROCESSOR is not None
        cfg = load_config()
        min_rms = float(cfg.get("vad_min_rms", WHISPER_MIN_RMS))
        cooldown_sec = float(cfg.get("post_final_cooldown_sec", POST_FINAL_COOLDOWN_SEC))

        model, processor, device, dtype = _MODEL, _PROCESSOR, _DEVICE, _DTYPE

        def transcribe(audio: np.ndarray) -> str:
            audio = prepare_chunk(audio, min_rms=min_rms)
            if audio is None:
                return ""
            inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
            input_features = inputs.input_features.to(device=device, dtype=dtype)
            with torch.inference_mode():
                predicted_ids = model.generate(
                    input_features,
                    language="en",
                    task="transcribe",
                    condition_on_prev_tokens=False,
                    use_cache=False,
                )
            text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            text = (text or "").strip()
            if _is_junk(text):
                return ""
            return text

        stop = stop_event or threading.Event()
        need = int(CHUNK_SEC * sample_rate)
        buf = np.zeros(0, dtype=np.float32)
        cool_until = 0.0
        last_final = ""

        blocksize = int(cfg.get("mic_blocksize", 4096))
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
                    # Drop bleed / self-echo while cooldown is active.
                    buf = np.zeros(0, dtype=np.float32)
                    signaled_transcribing = False
                    continue
                buf = np.concatenate([buf, block.reshape(-1).astype(np.float32)])
                while len(buf) >= need and not stop.is_set():
                    if time.monotonic() < cool_until:
                        buf = np.zeros(0, dtype=np.float32)
                        break
                    chunk = buf[:need]
                    # Non-overlapping hops — 75% overlap was re-saying the same line.
                    buf = buf[need:]
                    # Peek RMS before heavy work so UI can show Transcribing…
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
                    if not text:
                        signaled_transcribing = False
                        on_partial("Listening…")
                        continue
                    # Skip near-duplicate of the last commit (echo / re-hear).
                    if text.lower() == last_final.lower():
                        cool_until = time.monotonic() + cooldown_sec
                        buf = np.zeros(0, dtype=np.float32)
                        continue
                    last_final = text
                    summary.final_count += 1
                    summary.finals.append(text)
                    on_final(text)
                    cool_until = time.monotonic() + cooldown_sec
                    buf = np.zeros(0, dtype=np.float32)
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
        return summary
