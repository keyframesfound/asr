"""Parakeet must bind a fresh MLX GPU stream after stop/restart.

No Apple GPU or parakeet-mlx install required: mlx and the audio caches are fakes.
The regression is a second worker reusing Stream(gpu, 0) from the thread that
already exited, which raises ``There is no Stream(gpu, 0) in current thread``.
"""
from __future__ import annotations

import sys
import threading
import types
import unittest
from contextlib import contextmanager
from unittest import mock

# engines.mic imports sounddevice at import time. Stub it before parakeet loads.
if "sounddevice" not in sys.modules:
    sys.modules["sounddevice"] = types.ModuleType("sounddevice")

import engines.parakeet as pk  # noqa: E402
from engines.base import SessionSummary  # noqa: E402


class _StreamCtx:
    def __init__(self, stream: object) -> None:
        self.stream = stream

    def __enter__(self) -> object:
        return self.stream

    def __exit__(self, *args: object) -> bool:
        return False


class _Window:
    def __init__(self) -> None:
        self.clears = 0

    def cache_clear(self) -> None:
        self.clears += 1

    def __call__(self, size: int) -> tuple:
        return ("window", size, threading.get_ident())


class _Mx:
    def __init__(self) -> None:
        self.gpu = "gpu"
        self.streams: list[str] = []
        self.cleared = 0
        self.default = None

    def set_default_device(self, device: object) -> None:
        return None

    def default_device(self) -> str:
        return "gpu"

    def new_stream(self, device: object) -> str:
        stream = f"Stream(gpu, {len(self.streams)})"
        self.streams.append(stream)
        return stream

    def set_default_stream(self, stream: object) -> None:
        self.default = stream

    def stream(self, stream: object) -> _StreamCtx:
        return _StreamCtx(stream)

    def synchronize(self, stream: object | None = None) -> None:
        return None

    def clear_streams(self) -> None:
        self.cleared += 1

    def eval(self, *args: object) -> None:
        return None


def _install_fakes(mx: _Mx, loads: list[int], window: _Window) -> None:
    mlx = types.ModuleType("mlx")
    core = types.ModuleType("mlx.core")
    for name in (
        "gpu",
        "set_default_device",
        "default_device",
        "new_stream",
        "set_default_stream",
        "stream",
        "synchronize",
        "clear_streams",
        "eval",
    ):
        setattr(core, name, getattr(mx, name))
    mlx.core = core
    sys.modules["mlx"] = mlx
    sys.modules["mlx.core"] = core

    pkg = types.ModuleType("parakeet_mlx")

    def from_pretrained(*args: object, **kwargs: object) -> object:
        loads.append(threading.get_ident())
        return types.SimpleNamespace(parameters=lambda: {})

    pkg.from_pretrained = from_pretrained
    audio = types.ModuleType("parakeet_mlx.audio")
    audio.hanning = window
    audio.hamming = window
    audio.blackman = window
    audio.bartlett = window
    pkg.audio = audio
    sys.modules["parakeet_mlx"] = pkg
    sys.modules["parakeet_mlx.audio"] = audio


@contextmanager
def _fakes():
    mx = _Mx()
    loads: list[int] = []
    window = _Window()
    saved = {
        name: sys.modules.get(name)
        for name in (
            "mlx",
            "mlx.core",
            "parakeet_mlx",
            "parakeet_mlx.audio",
        )
    }
    pk.unload()
    _install_fakes(mx, loads, window)
    try:
        yield mx, loads, window
    finally:
        pk.unload()
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


@contextmanager
def _collapsed_parakeet_ident():
    """One OS ident for engines.parakeet calls only.

    ``engines.parakeet.threading`` is the real threading module. Replacing
    ``get_ident`` outright makes ``Thread.join`` resolve the worker as the
    current thread (``RuntimeError: cannot join current thread``) and races
    with removal from ``threading._active``. Thread bookkeeping keeps the
    real ident; Parakeet still sees a reused ident so a ``get_ident()`` key
    cannot pass this test.
    """
    real = threading.get_ident

    def get_ident() -> int:
        caller = sys._getframe(1).f_globals.get("__name__")
        if caller == "engines.parakeet":
            return 1
        return real()

    with mock.patch("threading.get_ident", get_ident):
        yield


class ParakeetStreamRestartTest(unittest.TestCase):
    def _finish(self, thread: threading.Thread, done: threading.Event) -> None:
        self.assertTrue(done.wait(5), "worker did not finish")
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())

    def test_second_worker_gets_a_new_stream_and_reloads(self) -> None:
        with _fakes() as (mx, loads, window):
            with _collapsed_parakeet_ident():
                errors: list[BaseException] = []

                def worker(*, release: bool, done: threading.Event) -> None:
                    try:
                        info = pk.preload()
                        if "already loaded" in info:
                            raise AssertionError(f"warm model reused across workers: {info}")
                        if release:
                            pk._end_worker_stream()
                    except BaseException as exc:  # noqa: BLE001
                        errors.append(exc)
                    finally:
                        done.set()

                # Leave the model warm, as a crashed or not-yet-joined worker would.
                first_done = threading.Event()
                first = threading.Thread(
                    target=worker, kwargs={"release": False, "done": first_done}
                )
                first.start()
                self._finish(first, first_done)
                second_done = threading.Event()
                second = threading.Thread(
                    target=worker, kwargs={"release": True, "done": second_done}
                )
                second.start()
                self._finish(second, second_done)

            self.assertEqual(errors, [])
            self.assertEqual(len(loads), 2)
            self.assertEqual(mx.streams, ["Stream(gpu, 0)", "Stream(gpu, 1)"])
            self.assertGreaterEqual(mx.cleared, 1)
            self.assertGreaterEqual(window.clears, 1)
            self.assertIsNone(pk._MODEL)
            self.assertIsNone(pk._MODEL_STREAM)

    def test_same_thread_keeps_the_warm_model(self) -> None:
        with _fakes() as (mx, loads, _window):
            first = pk.preload()
            second = pk.preload()
            self.assertIn("loaded", first)
            self.assertIn("already loaded", second)
            self.assertEqual(len(loads), 1)
            self.assertEqual(len(mx.streams), 1)

    def test_run_releases_stream_when_the_session_ends(self) -> None:
        summary = SessionSummary(engine="Parakeet Unified EN")
        engine = pk.ParakeetEngine()
        with mock.patch.object(pk.ParakeetEngine, "_run_locked", return_value=summary) as locked:
            with mock.patch.object(pk, "_end_worker_stream") as end:
                got = engine.run(lambda _text: None, lambda _text: None)
        self.assertIs(got, summary)
        locked.assert_called_once()
        end.assert_called_once()


if __name__ == "__main__":
    unittest.main()
