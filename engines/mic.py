"""Microphone capture helpers (sounddevice)."""
from __future__ import annotations

import queue
from typing import Iterator

import numpy as np
import sounddevice as sd


def open_input_stream(sample_rate: int = 16000, channels: int = 1, blocksize: int = 3200):
    q: queue.Queue[np.ndarray] = queue.Queue()

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
