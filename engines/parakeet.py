"""Parakeet TDT — Apple Silicon via parakeet-mlx (streaming, GPU-only, self-contained)."""
from __future__ import annotations

import os
import queue
import threading
import time
from pathlib import Path

import numpy as np

from .audio_util import (
    join_transcript_parts,
    plan_silent_chunks,
    polish_session,
    prepare_chunk,
)
from .base import LiveEngine, OnFinal, OnPartial, SessionSummary
from .config import load_config
from .mic import open_input_stream

# MLX weights in models/parakeet-mlx (scripts/download_models.py, or first use).
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
_MODEL_THREAD: int | None = None  # id(threading.current_thread()), not get_ident()
_MODEL_STREAM = None  # mx.Stream used for load + inference on this thread
# One live Parakeet worker at a time. A restart must not touch MLX until the
# previous thread has dropped its stream.
_WORKER_LOCK = threading.Lock()


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


def _thread_key() -> int:
    """Identity of this worker. ``get_ident()`` can be reused after a thread exits."""
    return id(threading.current_thread())


def _drop_parakeet_array_caches() -> None:
    """Forget mx.arrays parakeet_mlx cached on a previous worker.

    ``parakeet_mlx.audio`` memoizes window functions (``hanning`` and friends)
    with ``lru_cache``. Those arrays are permanently bound to the stream that
    created them — typically ``Stream(gpu, 0)`` on the first live thread.
    The next worker then dies with
    ``There is no Stream(gpu, 0) in current thread``.
    """
    import sys

    modules = [
        mod
        for name, mod in list(sys.modules.items())
        if name == "parakeet_mlx" or name.startswith("parakeet_mlx.")
    ]
    for mod in modules:
        for value in vars(mod).values():
            clear = getattr(value, "cache_clear", None)
            if not callable(clear):
                continue
            try:
                clear()
            except Exception:
                pass


def unload() -> None:
    """Drop the warm Parakeet model so another engine or thread can own the GPU."""
    global _MODEL, _MODEL_THREAD, _MODEL_STREAM
    _MODEL = None
    _MODEL_THREAD = None
    _MODEL_STREAM = None
    _drop_parakeet_array_caches()


def _clear_mlx_streams(mx) -> None:
    """Destroy streams created on this thread (``mx.clear_streams`` when present)."""
    sync = getattr(mx, "synchronize", None)
    if callable(sync):
        try:
            sync()
        except Exception:
            pass
    clear = getattr(mx, "clear_streams", None)
    if callable(clear):
        try:
            clear()
        except Exception:
            pass


def _end_worker_stream() -> None:
    """Release this worker's model and GPU stream before the thread exits.

    The next live session runs on a new thread and must create its own stream.
    Leaving ``Stream(gpu, 0)`` (and arrays that point at it) behind makes
    restart raise ``There is no Stream(gpu, 0) in current thread``.
    """
    mx = None
    try:
        import mlx.core as mx
    except Exception:
        mx = None
    stream = _MODEL_STREAM
    if mx is not None and stream is not None:
        try:
            mx.synchronize(stream)
        except Exception:
            pass
    unload()
    if mx is not None:
        _clear_mlx_streams(mx)


def _realize_on_stream(mx, model) -> None:
    """Evaluate parameters on the stream that just loaded them."""
    try:
        params = model.parameters()
    except Exception:
        return
    leaves = None
    try:
        from mlx.utils import tree_flatten

        leaves = [leaf for _name, leaf in tree_flatten(params)]
    except Exception:
        leaves = None
    if not leaves:
        return
    try:
        mx.eval(*leaves)
    except Exception:
        pass


def _bind_mlx_gpu_on_this_thread():
    """Create a new GPU stream on this thread; refuse Stream(cpu, …).

    MLX streams are thread-local. Textual starts a new worker each time live
    ASR is toggled. ``mx.new_stream`` here must run on that worker — a stream
    object left over from the previous thread is ``Stream(gpu, N)`` with no
    command encoder on the new thread.
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
    set_default = getattr(mx, "set_default_stream", None)
    if callable(set_default):
        try:
            set_default(stream)
        except Exception:
            pass
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
    key = _thread_key()
    if _MODEL is not None and _MODEL_THREAD == key and _MODEL_STREAM is not None:
        return f"already loaded ({MODEL_ID})"
    # Model, window caches, and streams from another thread cannot be reused.
    unload()

    mx, stream, device = _bind_mlx_gpu_on_this_thread()
    # Import / cached mx.arrays must be rebuilt on this thread's stream.
    _drop_parakeet_array_caches()
    from parakeet_mlx import from_pretrained

    _drop_parakeet_array_caches()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with mx.stream(stream):
        model = from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
        _realize_on_stream(mx, model)
    _MODEL = model
    _MODEL_THREAD = key
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

    def polish(
        self, audio: np.ndarray, *, sample_rate: int = 16000
    ) -> str | None:
        """Batch-decode the whole session with the warm model.

        Full-utterance context — the accuracy a push-to-talk dictation app
        gets: every millisecond of raw mic audio, decoded in one pass with
        pause-snapped pieces for long recordings. Runs on this worker's GPU
        stream while the worker lock is still held.
        """
        if _MODEL is None or _MODEL_STREAM is None or audio.size < sample_rate:
            return None
        import mlx.core as mx
        from parakeet_mlx.audio import get_logmel

        model, stream = _MODEL, _MODEL_STREAM
        pieces: list[str] = []
        with mx.stream(stream):
            for start, end in plan_silent_chunks(audio, sample_rate, max_sec=100.0):
                mel = get_logmel(mx.array(audio[start:end]), model.preprocessor_config)
                result = model.generate(mel)[0]
                text = (getattr(result, "text", "") or "").strip()
                if text:
                    pieces.append(text)
        return join_transcript_parts(pieces) or None

    def run(
        self,
        on_partial: OnPartial,
        on_final: OnFinal,
        *,
        sample_rate: int = 16000,
        stop_event: threading.Event | None = None,
    ) -> SessionSummary:
        """Run one live session, then drop this thread's MLX stream.

        Stop/restart starts a new worker. The lock waits until the previous
        worker has called ``clear_streams`` so the new thread can bind a
        fresh GPU stream instead of touching ``Stream(gpu, 0)``.
        """
        with _WORKER_LOCK:
            try:
                return self._run_locked(
                    on_partial,
                    on_final,
                    sample_rate=sample_rate,
                    stop_event=stop_event,
                )
            finally:
                _end_worker_stream()

    def _run_locked(
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
        # Every raw mic block, kept out of the live path's gating/cooldown so
        # the post-stop polish pass can re-decode the complete session.
        session: list[np.ndarray] = []
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
                        session.append(block.reshape(-1).astype(np.float32))
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
        polish_session(
            self,
            session,
            sample_rate=sample_rate,
            on_partial=on_partial,
            summary=summary,
        )
        return summary
