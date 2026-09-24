"""Microphone capture helpers (sounddevice).

Input selection prefers a physical mic (AirPods, headset, Built-in, MacBook, USB)
and will not silently open a blocklisted loopback when another input exists.
"""
from __future__ import annotations

import logging
import queue
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from . import level as level_bus
from .config import load_audio_config

logger = logging.getLogger("engines.mic")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_handler)
logger.propagate = False


@dataclass(frozen=True)
class InputChoice:
    """Mic chosen for a live session.

    ``index`` is None when PortAudio should use its default because no input
    could be listed. ``blocked_only`` means every listed input matched the
    loopback blocklist; the choice is logged, not silent.
    """

    index: int | None
    name: str
    blocked_only: bool = False


_LAST_CHOICE: InputChoice | None = None

# Last successfully opened input (for UI/debug). Index is None for PortAudio default.
LAST_INPUT_DEVICE: int | None = None
LAST_INPUT_NAME: str | None = None


def last_input_choice() -> InputChoice | None:
    return _LAST_CHOICE


def pick_input_device() -> int | None:
    """Physical-ish input index, or None when PortAudio should use its default."""
    return select_input_device().index


def list_input_devices() -> list[dict]:
    """Every recordable input as {index, name, max_input_channels} for the picker UI."""
    try:
        devices = sd.query_devices()
    except Exception as exc:
        logger.warning("Could not query input devices (%s)", exc)
        return []
    return _normalize_inputs(list(devices))


def listening_label(choice: InputChoice | None = None) -> str:
    """One status line so a session shows which mic was opened."""
    if choice is None:
        choice = _LAST_CHOICE
    if choice is None:
        return "Listening…"
    index = "default" if choice.index is None else str(choice.index)
    return f"Listening on {choice.name} (index {index})…"


def _name_has(name: str, pattern: str) -> bool:
    needle = pattern.strip().casefold()
    if not needle:
        return False
    return needle in name.casefold()


def _blocked(name: str, blocklist: Sequence[str]) -> bool:
    return any(_name_has(name, pattern) for pattern in blocklist)


def _normalize_inputs(devices: Sequence[Mapping]) -> list[dict]:
    inputs: list[dict] = []
    for i, dev in enumerate(devices):
        if not hasattr(dev, "get"):
            continue
        try:
            channels = int(dev.get("max_input_channels") or 0)
        except (TypeError, ValueError):
            continue
        if channels <= 0:
            continue
        name = str(dev.get("name") or "").strip() or f"input {i}"
        try:
            index = int(dev.get("index", i))
        except (TypeError, ValueError):
            index = i
        inputs.append({"index": index, "name": name, "max_input_channels": channels})
    return inputs


def select_input_device(
    devices: Sequence[Mapping] | None = None,
    *,
    blocklist: Sequence[str] | None = None,
    prefer: Sequence[str] | None = None,
) -> InputChoice:
    """Pick an input device index and name.

    Prefer the first ``prefer`` pattern that matches a non-blocklisted input
    (list order is priority: AirPods before Built-in before MacBook before USB).
    If nothing preferred matches, use the first non-blocklisted input. A
    blocklisted device is returned only when every input is blocklisted.
    """
    cfg = load_audio_config()
    if blocklist is None:
        blocklist = cfg.device_blocklist
    if prefer is None:
        prefer = cfg.device_prefer

    if devices is None:
        try:
            devices = sd.query_devices()
        except Exception as exc:
            logger.warning("Could not query input devices (%s); using PortAudio default", exc)
            return InputChoice(index=None, name="default")

    inputs = _normalize_inputs(devices)
    if not inputs:
        return InputChoice(index=None, name="default")

    # Explicit config override wins over blocklist and prefer hints.
    override = (cfg.input_device or "").strip()
    if override:
        for dev in inputs:
            if dev["name"].casefold() == override.casefold():
                return InputChoice(index=int(dev["index"]), name=str(dev["name"]))
        logger.warning(
            "input_device %r not found among inputs; falling back to auto selection",
            override,
        )

    allowed = [dev for dev in inputs if not _blocked(dev["name"], blocklist)]
    blocked_only = not allowed
    pool = allowed if allowed else inputs

    chosen = None
    for pattern in prefer:
        for dev in pool:
            if _name_has(dev["name"], pattern):
                chosen = dev
                break
        if chosen is not None:
            break
    if chosen is None:
        chosen = pool[0]

    return InputChoice(
        index=int(chosen["index"]),
        name=str(chosen["name"]),
        blocked_only=blocked_only,
    )


def log_input_choice(choice: InputChoice) -> None:
    """Log the chosen device once per stream open."""
    index = "default" if choice.index is None else str(choice.index)
    if choice.blocked_only:
        logger.warning(
            "Mic input: %s (index %s); every input matches the loopback blocklist",
            choice.name,
            index,
        )
        return
    logger.info("Mic input: %s (index %s)", choice.name, index)


def open_input_stream(
    sample_rate: int = 16000,
    channels: int = 1,
    blocksize: int = 4096,
    *,
    choice: InputChoice | None = None,
):
    """Open a 16 kHz mono float32 input. Logs the device name and index once."""
    global _LAST_CHOICE, LAST_INPUT_DEVICE, LAST_INPUT_NAME
    if choice is None:
        choice = select_input_device()
    _LAST_CHOICE = choice
    LAST_INPUT_DEVICE = choice.index
    LAST_INPUT_NAME = choice.name
    log_input_choice(choice)

    q: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata, frames, time_info, status):  # noqa: ARG001
        if status:
            pass
        block = indata.copy()
        try:
            flat = block.reshape(-1)
            if flat.size:
                rms = float(np.sqrt(np.mean(np.square(flat, dtype=np.float64))))
                peak = float(np.max(np.abs(flat)))
                level_bus.update_level(rms, peak)
        except Exception:
            pass
        q.put(block)

    kwargs = dict(
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
        blocksize=blocksize,
        callback=callback,
    )
    if choice.index is not None:
        kwargs["device"] = choice.index
    stream = sd.InputStream(**kwargs)
    return stream, q


def float32_to_pcm16(block: np.ndarray) -> bytes:
    clipped = np.clip(block.reshape(-1), -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def iter_pcm16(q: queue.Queue, stop_flag) -> Iterator[bytes]:
    while not stop_flag.is_set():
        try:
            block = q.get(timeout=0.2)
        except queue.Empty:
            continue
        yield float32_to_pcm16(block)
