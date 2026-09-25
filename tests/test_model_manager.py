"""Model manager — weights presence, download lifecycle, picker wiring."""
from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from engines import weights as weights_bus


def _patch_hubs(testcase: unittest.TestCase) -> dict[str, Path]:
    """Point the local-engine entries at temp dirs — never touch real weights."""
    tmp = tempfile.TemporaryDirectory()
    testcase.addCleanup(tmp.cleanup)
    base = Path(tmp.name)
    paths = {
        "sensevoice": base / "sensevoice-small",
        "whisper": base / "whisper-large-v3-turbo",
        "parakeet": base / "parakeet-mlx",
    }
    patched = dict(weights_bus.HUB_REPOS)
    patched["sensevoice"] = ("FunAudioLLM/SenseVoiceSmall", paths["sensevoice"], "dir", "model.pt")
    patched["whisper"] = ("openai/whisper-large-v3-turbo", paths["whisper"], "dir", "model.safetensors")
    patched["parakeet"] = ("mlx-community/parakeet-tdt-0.6b-v3", paths["parakeet"], "hub-cache", "")
    patcher = mock.patch.dict(weights_bus.HUB_REPOS, patched, clear=True)
    patcher.start()
    testcase.addCleanup(patcher.stop)
    # Keep tests off the network and in-process: the GitHub release source is
    # disabled unless a test opts in with a file:// base URL, and the child
    # downloader runs in-thread so the mocks above apply to it.
    env = mock.patch.dict(
        os.environ, {"ASR_WEIGHTS_BASE_URL": "off", "ASR_WEIGHTS_INPROC": "1"}
    )
    env.start()
    testcase.addCleanup(env.stop)
    # Keep the expected-size lookup off the network; tests set it explicitly.
    total = mock.patch.object(weights_bus, "_hf_total_bytes", return_value=0)
    total.start()
    testcase.addCleanup(total.stop)
    weights_bus._states.clear()
    weights_bus._meters.clear()
    return paths


class HumanSizeTest(unittest.TestCase):
    def test_units(self) -> None:
        self.assertEqual(weights_bus.human_size(0), "0 B")
        self.assertEqual(weights_bus.human_size(2048), "2.0 KB")
        self.assertEqual(weights_bus.human_size(int(1.5 * 1024**3)), "1.5 GB")


class WeightsStatusTest(unittest.TestCase):
    def setUp(self) -> None:
        self.paths = _patch_hubs(self)

    def test_cloud_engine_has_no_weights(self) -> None:
        self.assertFalse(weights_bus.is_local_engine("iflytek"))
        self.assertTrue(weights_bus.weights_present("iflytek"))

    def test_human_eta_units(self) -> None:
        self.assertEqual(weights_bus.human_eta(45), "45s")
        self.assertEqual(weights_bus.human_eta(200), "3m 20s")
        self.assertEqual(weights_bus.human_eta(3900), "1h 05m")

    def test_meter_speed_over_window(self) -> None:
        meter = weights_bus._Meter()
        self.assertEqual(meter.feed(0), 0.0)
        time.sleep(0.3)
        self.assertGreater(meter.feed(3_000_000), 1_000_000)

    def test_note_progress_updates_state(self) -> None:
        with mock.patch.dict(
            weights_bus._states, {"whisper": weights_bus.DownloadState(running=True)}
        ):
            weights_bus._set_total("whisper", 4096)
            weights_bus._note_progress("whisper", 2048)
            state = weights_bus.download_state("whisper")
            self.assertEqual(state.total, 4096)
            self.assertEqual(state.downloaded, 2048)

    def test_parakeet_cache_requires_weights_blob(self) -> None:
        """Config-only leftovers must not verify; a real blob must."""
        snap = (
            self.paths["parakeet"]
            / "models--mlx-community--parakeet-tdt-0.6b-v3"
            / "snapshots"
            / "abc123"
        )
        snap.mkdir(parents=True)
        (snap / "config.json").write_bytes(b"{}")
        self.assertFalse(weights_bus.download_ok("parakeet"))
        (snap / "model.safetensors").write_bytes(b"w")
        self.assertTrue(weights_bus.download_ok("parakeet"))
        (snap / "model.safetensors").write_bytes(b"")
        self.assertFalse(weights_bus.download_ok("parakeet"))

    def test_hf_total_reaches_state(self) -> None:
        """The expected repo size flows into DownloadState for % and ETA."""

        def ok_snapshot(repo_id, local_dir=None, cache_dir=None):
            (local_dir / "model.pt").write_bytes(b"z" * 512)

        with mock.patch(
            "huggingface_hub.snapshot_download", ok_snapshot
        ), mock.patch.object(weights_bus, "_hf_total_bytes", return_value=4096):
            self.assertTrue(weights_bus.download_weights_async("sensevoice"))
            self._wait_idle("sensevoice")
        state = weights_bus.download_state("sensevoice")
        self.assertEqual(state.error, "")
        self.assertEqual(state.total, 4096)

    def test_presence_size_remove(self) -> None:
        target = self.paths["sensevoice"]
        self.assertFalse(weights_bus.weights_present("sensevoice"))
        self.assertEqual(weights_bus.weights_size("sensevoice"), 0)

        # Regression: a half-finished download leaves config files but no
        # weights blob — that must not count as installed.
        target.mkdir(parents=True)
        (target / "config.yaml").write_bytes(b"y" * 512)
        self.assertFalse(weights_bus.weights_present("sensevoice"))

        (target / "model.pt").write_bytes(b"x" * 2048)
        self.assertTrue(weights_bus.weights_present("sensevoice"))
        self.assertEqual(weights_bus.weights_size("sensevoice"), 2560)

        freed = weights_bus.remove_weights("sensevoice")
        self.assertEqual(freed, "2.5 KB")
        self.assertFalse(target.exists())
        self.assertFalse(weights_bus.weights_present("sensevoice"))

    def test_remove_rejects_cloud(self) -> None:
        with self.assertRaises(ValueError):
            weights_bus.remove_weights("iflytek")

    def test_download_rejects_cloud(self) -> None:
        with self.assertRaises(ValueError):
            weights_bus.download_weights_async("iflytek")

    def test_download_lifecycle(self) -> None:
        """In-flight downloads block a second start; completion clears state."""
        target = self.paths["sensevoice"]
        release = threading.Event()
        started = threading.Event()

        def fake_snapshot(repo_id, local_dir=None, cache_dir=None):
            started.set()
            release.wait(5)
            (local_dir / "model.pt").write_bytes(b"z" * 512)

        with mock.patch("huggingface_hub.snapshot_download", fake_snapshot):
            self.assertTrue(weights_bus.download_weights_async("sensevoice"))
            self.assertTrue(started.wait(5))
            self.assertFalse(weights_bus.download_weights_async("sensevoice"))

            release.set()
            state = weights_bus.download_state("sensevoice")
            for _ in range(100):
                state = weights_bus.download_state("sensevoice")
                if not state.running:
                    break
                time.sleep(0.05)
        self.assertEqual(state.error, "")
        self.assertTrue(weights_bus.weights_present("sensevoice"))

    def test_release_tar_primary_hf_fallback(self) -> None:
        """Release tar wins for dir engines; the HF hub covers its absence."""
        target = self.paths["whisper"]
        base = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(base, ignore_errors=True))
        payload = base / "payload" / target.name
        payload.mkdir(parents=True)
        (payload / "model.safetensors").write_bytes(b"w" * 4096)
        (payload / "config.json").write_bytes(b"{}")
        with tarfile.open(base / f"{target.name}.tar", "w") as tf:
            tf.add(payload, arcname=target.name)

        env = mock.patch.dict(
            os.environ, {"ASR_WEIGHTS_BASE_URL": f"file://{base}"}
        )
        env.start()
        self.addCleanup(env.stop)

        # Release tar available → it is used and HF is never touched.
        with mock.patch("huggingface_hub.snapshot_download") as hub:
            self.assertTrue(weights_bus.download_weights_async("whisper"))
            self._wait_idle("whisper")
            hub.assert_not_called()
        state = weights_bus.download_state("whisper")
        self.assertEqual(state.error, "")
        self.assertTrue(weights_bus.weights_present("whisper"))
        self.assertFalse((base / f"{target.name}.tar.part").exists())
        shutil.rmtree(target, ignore_errors=True)

        # The tar source disappears → HF takes over.
        (base / f"{target.name}.tar").unlink()

        def ok_snapshot(repo_id, local_dir=None, cache_dir=None):
            (local_dir / "model.safetensors").write_bytes(b"h" * 4096)

        with mock.patch("huggingface_hub.snapshot_download", ok_snapshot):
            self.assertTrue(weights_bus.download_weights_async("whisper"))
            self._wait_idle("whisper")
        self.assertTrue(weights_bus.weights_present("whisper"))

    def test_stall_watchdog(self) -> None:
        """A download that stops moving is flagged instead of spinning forever."""
        target = self.paths["sensevoice"]
        target.mkdir(parents=True)
        release = threading.Event()

        def hang_snapshot(repo_id, local_dir=None, cache_dir=None):
            (local_dir / "config.yaml").write_bytes(b"y")
            release.wait(10)

        with mock.patch("huggingface_hub.snapshot_download", hang_snapshot), mock.patch.object(
            weights_bus, "STALL_TIMEOUT_SEC", 1.0
        ):
            self.assertTrue(weights_bus.download_weights_async("sensevoice"))
            self._wait_idle("sensevoice")
        state = weights_bus.download_state("sensevoice")
        self.assertIn("stalled", state.error)
        release.set()  # let the abandoned worker thread drain

    def _wait_idle(self, code: str) -> None:
        state = weights_bus.download_state(code)
        for _ in range(300):
            state = weights_bus.download_state(code)
            if not state.running:
                return
            time.sleep(0.05)
        self.fail(f"download for {code} never went idle")


class PickerWiringTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.paths = _patch_hubs(self)

    async def test_statuses_install_and_remove_flow(self) -> None:
        from tui_app import AudioLiveApp, ModelPickerScreen

        sensevoice_dir = self.paths["sensevoice"]
        sensevoice_dir.mkdir(parents=True)
        (sensevoice_dir / "model.pt").write_bytes(b"x" * 10)

        started: list[str] = []
        fake_start = mock.patch.object(
            weights_bus,
            "download_weights_async",
            side_effect=lambda code: started.append(code) or True,
        )
        fake_start.start()
        self.addCleanup(fake_start.stop)

        app = AudioLiveApp()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            self.assertIsInstance(app.screen, ModelPickerScreen)

            whisper_status = str(app.screen.query_one("#eng-whisper .model-status").render())
            self.assertIn("not installed", whisper_status)
            sensevoice_status = str(app.screen.query_one("#eng-sensevoice .model-status").render())
            self.assertIn("installed", sensevoice_status)
            iflytek_status = str(app.screen.query_one("#eng-iflytek .model-status").render())
            self.assertIn("cloud", iflytek_status)

            # Navigate to Whisper (missing) and try to start — stay on the picker.
            await pilot.press("down", "enter")
            await pilot.pause()
            self.assertIsInstance(app.screen, ModelPickerScreen)

            # i starts the download via the weights bus.
            await pilot.press("i")
            await pilot.pause()
            self.assertEqual(started, ["whisper"])

            # u requires a second press to actually delete.
            await pilot.press("down")  # whisper → sensevoice
            await pilot.press("u")
            await pilot.pause()
            self.assertTrue(sensevoice_dir.exists())
            await pilot.press("u")
            await pilot.pause()
            self.assertFalse(sensevoice_dir.exists())

    async def test_row_shows_percentage_speed_and_eta(self) -> None:
        from tui_app import ModelPickerScreen

        screen = ModelPickerScreen()
        weights_bus._states["whisper"] = weights_bus.DownloadState(
            running=True, downloaded=500, total=1000, speed=100.0
        )
        try:
            text = str(screen._progress_text(weights_bus.download_state("whisper")))
            self.assertIn("downloading… 50%", text)
            self.assertIn("100 B/s", text)
            self.assertIn("~5s left", text)

            # No expected size known → fall back to a plain byte counter.
            weights_bus._states["whisper"] = weights_bus.DownloadState(
                running=True, downloaded=2048
            )
            text = str(screen._progress_text(weights_bus.download_state("whisper")))
            self.assertIn("downloading… 2.0 KB", text)
            self.assertNotIn("%", text)
            self.assertNotIn("left", text)
        finally:
            weights_bus._states.pop("whisper", None)

    async def test_row_flips_to_installed_without_restart(self) -> None:
        """Regression: a finished download must repaint its row live."""
        target = self.paths["sensevoice"]
        release = threading.Event()
        started = threading.Event()

        def slow_snapshot(repo_id, local_dir=None, cache_dir=None):
            started.set()
            release.wait(5)
            (local_dir / "model.pt").write_bytes(b"z" * 512)

        from tui_app import AudioLiveApp, ModelPickerScreen

        with mock.patch("huggingface_hub.snapshot_download", slow_snapshot):
            app = AudioLiveApp()
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                self.assertIsInstance(app.screen, ModelPickerScreen)
                await pilot.press("down", "down", "i")  # select sensevoice, download
                self.assertTrue(started.wait(5))
                release.set()
                status = ""
                for _ in range(80):
                    await pilot.pause(0.25)
                    status = str(
                        app.screen.query_one("#eng-sensevoice .model-status").render()
                    )
                    if "installed ·" in status:
                        break
                self.assertIn("installed ·", status)
                self.assertFalse(weights_bus.download_state("sensevoice").running)


if __name__ == "__main__":
    unittest.main()
