"""Microphone capture helpers (sounddevice)."""
from __future__ import annotations

import queue
from typing import Iterator

import numpy as np
import sounddevice as sd

# Virtual / loopback-style devices that often hear speaker output or mix.
_BLOCKLIST_SUBSTR = (
    "eshare",
    "teams",
    "zoom",
    "steam streaming",
    "blackhole",
    "loopback",
    "soundflower",
    "aggregate",
    "multi-output",
)

# Last successfully opened input device index (for UI/debug).
LAST_INPUT_DEVICE: int | None = None
LAST_INPUT_NAME: str | None = None


def pick_input_device() -> int | None:
    """Return a physical-ish input device index, or None for PortAudio default."""
    devices = sd.query_devices()
    default_in, _default_out = sd.default.device
    candidates: list[tuple[int, int, str]] = []

    for idx, dev in enumerate(devices):
        if int(dev.get("max_input_channels") or 0) < 1:
            continue
        name = str(dev.get("name") or "")
        low = name.lower()
        if any(b in low for b in _BLOCKLIST_SUBSTR):
            continue
        if "airpods" in low or "headset" in low:
            pri = 0
        elif "macbook" in low and "microphone" in low:
            pri = 2
        elif "microphone" in low or "mic" in low:
            pri = 1
        else:
            pri = 3
        if default_in is not None and idx == int(default_in):
            pri -= 1
        candidates.append((pri, idx, name))

    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def open_input_stream(sample_rate: int = 16000, channels: int = 1, blocksize: int = 4096):
    """Open a mono input stream. Returns (stream, queue) — never uses loopback devices."""
    global LAST_INPUT_DEVICE, LAST_INPUT_NAME
    q: queue.Queue[np.ndarray] = queue.Queue()
    device = pick_input_device()
    LAST_INPUT_DEVICE = device
    try:
        LAST_INPUT_NAME = (
            str(sd.query_devices(device)["name"]) if device is not None else "system default"
        )
    except Exception:
        LAST_INPUT_NAME = str(device)

    def callback(indata, frames, time_info, status):  # noqa: ARG001
        if status:
            pass
        q.put(indata.copy())

    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
        blocksize=blocksize,
        callback=callback,
        device=device,
    )
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
