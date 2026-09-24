"""Parakeet TDT — Apple Silicon via parakeet-mlx (streaming, GPU-only, self-contained)."""
from __future__ import annotations

import os
import queue
import threading
import time
from pathlib import Path

import numpy as np

from .audio_util import prepare_chunk
from .base import LiveEngine, OnFinal, OnPartial, SessionSummary
from .config import load_config
from .mic import open_input_stream

# MLX weights (downloaded on first use into models/parakeet-mlx).
MODEL_ID = "mlx-community/parakeet-tdt-0.6b-v3"
CACHE_DIR = Path(__file__).resolve().parents[1] / "models" / "parakeet-mlx"
# Legacy Hex CoreML tree (optional; this engine does not use it).
COREML_DIR = (
    Path(__file__).resolve().parents[1] / "models" / "parakeet-unified-en-0.6b-coreml"
)
# Default feed size; overridden by config.json parakeet_feed_sec at run time.
FEED_SEC = 0.4
# Prefer not emitting tiny mid-word finals unless punctuation (overridden by config).
MIN_FINAL_CHARS = 8
_PUNCT_END = tuple(".?!。？！…,;:，、")

_MODEL = None
_MODEL_THREAD: int | None = None
_MODEL_STREAM = None  # mx.Stream used for load + inference on this thread


def weights_cached() -> bool:
    """True when MLX weights are already on disk (project or HF hub cache)."""
    repo_dir_name = "models--" + MODEL_ID.replace("/", "--")
    candidates = [CACHE_DIR / repo_dir_name]
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    candidates.append(hf_home / "hub" / repo_dir_name)
    for base in candidates:
        if not base.exists():
            continue
        for cfg in base.glob("snapshots/*/config.json"):
            parent = cfg.parent
            if (parent / "model.safetensors").exists() or any(parent.glob("*.safetensors")):
                return True
    return False


def unload() -> None:
    """Drop the warm Parakeet model so another engine can own GPU/RAM."""
    global _MODEL, _MODEL_THREAD, _MODEL_STREAM
    _MODEL = None
    _MODEL_THREAD = None
    _MODEL_STREAM = None


def _bind_mlx_gpu_on_this_thread():
    """Bind MLX to GPU on this thread; refuse Stream(cpu, …) for live inference.

    MLX default streams are thread-local. Textual runs engines on a worker
    thread; restarting live creates a new thread. Arrays loaded on another
    thread still reference Stream(cpu, N) from that thread and raise:
    There is no Stream(cpu, N) in current thread.
    """
    import mlx.core as mx

    try:
        gpu = mx.gpu
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "MLX GPU device unavailable. Parakeet live ASR requires Apple Silicon GPU "
            f"(mlx.gpu). Details: {exc}"
        ) from exc

    try:
        mx.set_default_device(gpu)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to set MLX default device to GPU: {exc}"
        ) from exc

    device = mx.default_device()
    # Compare by type name — Device equality can be finicky across builds.
    dev_str = str(device).lower()
    if "cpu" in dev_str and "gpu" not in dev_str:
        try:
            mx.set_default_device(mx.gpu)
            device = mx.default_device()
            dev_str = str(device).lower()
        except Exception:
            pass
    if "cpu" in dev_str and "gpu" not in dev_str:
        raise RuntimeError(
            f"MLX default device is CPU ({device}); Parakeet refuses CPU for live "
            "latency. Ensure you are on Apple Silicon with a working MLX GPU build."
        )

    stream = mx.new_stream(device)
    stream_str = str(stream).lower()
    if "cpu" in stream_str and "gpu" not in stream_str:
        raise RuntimeError(
            f"Refusing Stream(cpu): got {stream}. Parakeet live ASR requires GPU streams."
        )
    return mx, stream, device


# Back-compat alias used by prove_parakeet_thread.py
def _bind_mlx_on_this_thread():
    mx, stream, _device = _bind_mlx_gpu_on_this_thread()
    return mx, stream


def preload() -> str:
    """Load model on the calling thread; store GPU stream for later inference."""
    global _MODEL, _MODEL_THREAD, _MODEL_STREAM
    tid = threading.get_ident()
    if _MODEL is not None and _MODEL_THREAD == tid and _MODEL_STREAM is not None:
        return f"already loaded ({MODEL_ID})"
    # Model (and its streams) from another thread cannot be reused.
    if _MODEL is not None and _MODEL_THREAD != tid:
        unload()

    from parakeet_mlx import from_pretrained

    mx, stream, device = _bind_mlx_gpu_on_this_thread()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with mx.stream(stream):
        _MODEL = from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    _MODEL_THREAD = tid
    _MODEL_STREAM = stream
    return f"loaded {MODEL_ID} on {device}"


def _tokens_to_text(tokens, sentence_cfg) -> str:
    if not tokens:
        return ""
    from parakeet_mlx.alignment import sentences_to_result, tokens_to_sentences

    return (sentences_to_result(tokens_to_sentences(list(tokens), sentence_cfg)).text or "").strip()


def _should_emit_final(delta: str, min_chars: int) -> bool:
    """Emit when delta is long enough or ends with punctuation (latency > perfection)."""
    d = (delta or "").strip()
    if not d:
        return False
    if len(d) >= min_chars:
        return True
    return d[-1] in _PUNCT_END


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
            import mlx.core as mx  # noqa: F401
            from parakeet_mlx import from_pretrained  # noqa: F401
        except ImportError:
            n = len(list(COREML_DIR.rglob("*.mlmodelc"))) if COREML_DIR.exists() else 0
            summary.error = (
                "Parakeet needs parakeet-mlx (Apple Silicon). "
                "Install with: pip install parakeet-mlx"
                + (f" — CoreML Hex weights also present ({n} bundles)." if n else "")
            )
            return summary

        cfg = load_config()
        feed_sec = float(cfg.get("parakeet_feed_sec", FEED_SEC))
        cooldown_sec = float(cfg.get("post_final_cooldown_sec", 1.25))
        min_final_chars = int(cfg.get("min_final_chars", MIN_FINAL_CHARS))
        blocksize = int(cfg.get("mic_blocksize", 4096))
        min_rms = float(cfg.get("vad_min_rms", 0.02))

        try:
            on_partial("Loading Parakeet (MLX GPU)…")
            info = preload()
            on_partial(f"Ready ({info})")
        except Exception as exc:
            summary.error = f"Model load failed: {exc}"
            return summary

        assert _MODEL is not None and _MODEL_STREAM is not None
        model = _MODEL
        stream = _MODEL_STREAM
        import mlx.core as mx

        stop = stop_event or threading.Event()
        need = max(1, int(feed_sec * sample_rate))
        buf = np.zeros(0, dtype=np.float32)
        stream_mic, q = open_input_stream(sample_rate=sample_rate, blocksize=blocksize)
        stream_mic.start()
        on_partial("Listening…")
        emitted_final = ""
        last_partial = ""
        cool_until = 0.0
        signaled_transcribing = False

        try:
            with mx.stream(stream):
                with model.transcribe_stream() as asr:
                    sentence_cfg = asr.decoding_config.sentence
                    while not stop.is_set():
                        try:
                            block = q.get(timeout=0.2)
                        except queue.Empty:
                            continue
                        now = time.monotonic()
                        if now < cool_until:
                            # Discard mic during post-final echo cooldown.
                            buf = np.zeros(0, dtype=np.float32)
                            signaled_transcribing = False
                            continue
                        buf = np.concatenate([buf, block.reshape(-1).astype(np.float32)])
                        while len(buf) >= need and not stop.is_set():
                            if time.monotonic() < cool_until:
                                buf = np.zeros(0, dtype=np.float32)
                                break
                            chunk = buf[:need]
                            buf = buf[need:]
                            audio = prepare_chunk(chunk, min_rms=min_rms)
                            if audio is None:
                                # Below VAD — stay in listening unless we already have draft text.
                                if not last_partial and not signaled_transcribing:
                                    pass
                                continue

                            if not signaled_transcribing and not last_partial:
                                signaled_transcribing = True
                                on_partial("Transcribing…")

                            asr.add_audio(mx.array(audio))

                            # Finals = committed tokens; drafts change every chunk.
                            final_text = _tokens_to_text(asr.finalized_tokens, sentence_cfg)
                            draft_text = _tokens_to_text(asr.draft_tokens, sentence_cfg)

                            if final_text and final_text != emitted_final:
                                if emitted_final and final_text.startswith(emitted_final):
                                    delta = final_text[len(emitted_final) :].strip()
                                else:
                                    delta = final_text.strip()
                                if delta and _should_emit_final(delta, min_final_chars):
                                    emitted_final = final_text
                                    summary.final_count += 1
                                    summary.finals.append(delta)
                                    on_final(delta)
                                    cool_until = time.monotonic() + cooldown_sec
                                    buf = np.zeros(0, dtype=np.float32)
                                    signaled_transcribing = False
                                    last_partial = ""
                                    on_partial("Listening…")

                            display = draft_text or ""
                            # Prefer live draft on caption; settle with final if no draft.
                            if not display and final_text:
                                # Show only the unsettled tail / latest final for caption.
                                if emitted_final and final_text.startswith(emitted_final):
                                    display = final_text[len(emitted_final) :].strip() or final_text
                                else:
                                    display = final_text
                            if display and display != last_partial:
                                last_partial = display
                                summary.partial_count += 1
                                on_partial(display)
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
        return summary
