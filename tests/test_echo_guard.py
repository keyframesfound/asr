"""Device picker, post-final cooldown, and RMS gate. No mic or model required."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from engines.audio_util import (
    MAX_GAIN,
    PostFinalCooldown,
    accept_input_block,
    chunk_rms,
    prepare_chunk,
)
from engines.config import (
    DEFAULT_COOLDOWN_MS,
    DEFAULT_DEVICE_BLOCKLIST,
    DEFAULT_DEVICE_PREFER,
    DEFAULT_MIN_RMS,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_TARGET_RMS,
    clear_audio_config_cache,
    load_audio_config,
)
from engines.mic import InputChoice, float32_to_pcm16, listening_label, select_input_device

ROOT = Path(__file__).resolve().parents[1]
BLOCKLIST = ["Zoom", "Teams", "Steam", "EShare"]
PREFER = ["AirPods", "Built-in", "MacBook", "USB"]


def _dev(index: int, name: str, channels: int = 1) -> dict:
    return {"index": index, "name": name, "max_input_channels": channels}


class DeviceSelectionTest(unittest.TestCase):
    def test_airpods_preferred_over_builtin_and_zoom(self) -> None:
        devices = [
            _dev(0, "Zoom Audio Device", 2),
            _dev(1, "MacBook Pro Microphone"),
            _dev(2, "Built-in Microphone"),
            _dev(3, "AirPods Pro"),
        ]
        choice = select_input_device(devices, blocklist=BLOCKLIST, prefer=PREFER)
        self.assertEqual(choice.index, 3)
        self.assertEqual(choice.name, "AirPods Pro")
        self.assertFalse(choice.blocked_only)

    def test_virtual_devices_from_main_stay_blocked(self) -> None:
        devices = [
            _dev(0, "BlackHole 2ch"),
            _dev(1, "Loopback Audio"),
            _dev(2, "Soundflower (2ch)"),
            _dev(3, "Built-in Microphone"),
        ]
        choice = select_input_device(devices)
        self.assertEqual(choice.name, "Built-in Microphone")
        self.assertFalse(choice.blocked_only)

    def test_zoom_audio_device_present_but_unused(self) -> None:
        devices = [
            _dev(0, "Zoom Audio Device", 2),
            _dev(4, "AirPods"),
        ]
        choice = select_input_device(devices)
        self.assertEqual((choice.index, choice.name), (4, "AirPods"))
        self.assertFalse(choice.blocked_only)

    def test_prefer_order_builtin_before_macbook_before_usb(self) -> None:
        devices = [
            _dev(1, "USB Audio Device"),
            _dev(2, "MacBook Pro Microphone"),
            _dev(4, "Built-in Microphone"),
        ]
        choice = select_input_device(devices, blocklist=BLOCKLIST, prefer=PREFER)
        self.assertEqual(choice.index, 4)
        self.assertEqual(choice.name, "Built-in Microphone")

    def test_fallback_first_non_blocklisted(self) -> None:
        devices = [
            _dev(0, "ZoomAudioDevice"),
            _dev(1, "Steam Streaming Microphone"),
            _dev(2, "External Headphones"),
            _dev(3, "Line In"),
        ]
        choice = select_input_device(devices, blocklist=BLOCKLIST, prefer=PREFER)
        self.assertEqual(choice.index, 2)
        self.assertEqual(choice.name, "External Headphones")
        self.assertFalse(choice.blocked_only)

    def test_blocklist_is_case_insensitive(self) -> None:
        devices = [
            _dev(0, "eshare audio"),
            _dev(1, "Microsoft TEAMS Audio"),
            _dev(2, "steam streaming mic"),
            _dev(3, "zoomus virtual"),
            _dev(8, "MacBook Pro Microphone"),
        ]
        choice = select_input_device(devices, blocklist=BLOCKLIST, prefer=PREFER)
        self.assertEqual(choice.index, 8)
        self.assertFalse(choice.blocked_only)

    def test_never_picks_blocklisted_when_another_input_exists(self) -> None:
        devices = [
            _dev(0, "ZoomAudioDevice", 2),
            _dev(1, "USB Zoom Mixer"),
            _dev(6, "External Microphone"),
        ]
        choice = select_input_device(devices, blocklist=BLOCKLIST, prefer=["USB", "AirPods"])
        self.assertEqual(choice.name, "External Microphone")
        self.assertEqual(choice.index, 6)
        self.assertFalse(choice.blocked_only)

    def test_blocklisted_only_when_nothing_else_exists(self) -> None:
        devices = [
            _dev(0, "ZoomAudioDevice"),
            _dev(1, "Steam Streaming Microphone"),
        ]
        choice = select_input_device(devices, blocklist=BLOCKLIST, prefer=PREFER)
        self.assertTrue(choice.blocked_only)
        self.assertEqual(choice.index, 0)
        self.assertEqual(choice.name, "ZoomAudioDevice")

    def test_output_only_devices_are_ignored(self) -> None:
        devices = [
            _dev(0, "AirPods", 0),
            _dev(1, "Zoom Audio Device", 0),
            _dev(2, "Built-in Microphone", 1),
        ]
        choice = select_input_device(devices, blocklist=BLOCKLIST, prefer=PREFER)
        self.assertEqual(choice.index, 2)
        self.assertEqual(choice.name, "Built-in Microphone")

    def test_no_inputs_uses_portaudio_default(self) -> None:
        choice = select_input_device([], blocklist=BLOCKLIST, prefer=PREFER)
        self.assertIsNone(choice.index)
        self.assertEqual(choice.name, "default")
        self.assertFalse(choice.blocked_only)

    def test_index_falls_back_to_enumeration_order(self) -> None:
        devices = [
            {"name": "Zoom Audio Device", "max_input_channels": 1},
            {"name": "MacBook Pro Microphone", "max_input_channels": 1},
        ]
        choice = select_input_device(devices, blocklist=BLOCKLIST, prefer=PREFER)
        self.assertEqual(choice.index, 1)


class StreamOpenTest(unittest.TestCase):
    def test_open_logs_chosen_device_once_and_passes_index(self) -> None:
        import engines.mic as mic

        devices = [
            _dev(0, "Zoom Audio Device", 2),
            _dev(3, "AirPods"),
        ]
        captured: dict = {}

        class _Stream:
            pass

        def _input_stream(**kwargs):
            captured.update(kwargs)
            return _Stream()

        with mock.patch.object(mic.sd, "query_devices", return_value=devices):
            with mock.patch.object(mic.sd, "InputStream", side_effect=_input_stream):
                with self.assertLogs("engines.mic", level="INFO") as logs:
                    _stream, _q = mic.open_input_stream()
        self.assertEqual(captured["device"], 3)
        self.assertEqual(captured["samplerate"], 16000)
        self.assertEqual(captured["channels"], 1)
        self.assertEqual(captured["dtype"], "float32")
        messages = [r.getMessage() for r in logs.records]
        self.assertEqual(messages, ["Mic input: AirPods (index 3)"])
        self.assertEqual(mic.last_input_choice().index, 3)
        self.assertEqual(listening_label(), "Listening on AirPods (index 3)…")

    def test_blocked_only_is_logged_not_silent(self) -> None:
        import engines.mic as mic

        choice = InputChoice(index=0, name="ZoomAudioDevice", blocked_only=True)
        with mock.patch.object(mic.sd, "InputStream", return_value=object()):
            with self.assertLogs("engines.mic", level="INFO") as logs:
                mic.open_input_stream(choice=choice)
        self.assertIn("blocklist", logs.records[0].getMessage())
        self.assertEqual(logs.records[0].levelname, "WARNING")

    def test_default_device_omits_device_kwarg(self) -> None:
        import engines.mic as mic

        captured: dict = {}

        def _input_stream(**kwargs):
            captured.update(kwargs)
            return object()

        with mock.patch.object(mic.sd, "query_devices", return_value=[]):
            with mock.patch.object(mic.sd, "InputStream", side_effect=_input_stream):
                mic.open_input_stream()
        self.assertNotIn("device", captured)
        self.assertEqual(listening_label(), "Listening on default (index default)…")


class CooldownTest(unittest.TestCase):
    def test_drops_until_cooldown_elapses(self) -> None:
        now = {"t": 10.0}
        cooldown = PostFinalCooldown(1000, clock=lambda: now["t"])
        audio = np.full(16, 0.2, dtype=np.float32)
        self.assertFalse(cooldown.active())
        self.assertIsNotNone(cooldown.gate(audio))
        cooldown.arm()
        self.assertTrue(cooldown.active())
        self.assertIsNone(cooldown.gate(audio))
        now["t"] = 10.999
        self.assertIsNone(cooldown.gate(audio))
        now["t"] = 11.0
        self.assertFalse(cooldown.active())
        self.assertIsNotNone(cooldown.gate(audio))

    def test_second_arm_does_not_extend_window(self) -> None:
        now = {"t": 0.0}
        cooldown = PostFinalCooldown(1000, clock=lambda: now["t"])
        cooldown.arm()
        now["t"] = 0.4
        cooldown.arm()
        now["t"] = 1.0
        self.assertFalse(cooldown.active())

    def test_zero_cooldown_never_drops(self) -> None:
        cooldown = PostFinalCooldown(0, clock=lambda: 0.0)
        cooldown.arm()
        self.assertFalse(cooldown.active())
        audio = np.full(8, 0.2, dtype=np.float32)
        self.assertIsNotNone(cooldown.gate(audio))

    def test_keep_loud_passes_speech_and_drops_echo(self) -> None:
        now = {"t": 0.0}
        cooldown = PostFinalCooldown(1200, clock=lambda: now["t"])
        cooldown.arm()
        loud = np.full(32, 0.2, dtype=np.float32)
        quiet_echo = np.full(32, 0.01, dtype=np.float32)
        self.assertIsNotNone(cooldown.gate(loud, keep_loud=True))
        self.assertIsNone(cooldown.gate(quiet_echo, keep_loud=True))
        self.assertIsNone(cooldown.gate(loud, keep_loud=False))

    def test_accept_input_block_clears_overlap_during_cooldown(self) -> None:
        now = {"t": 0.0}
        cooldown = PostFinalCooldown(800, clock=lambda: now["t"])
        buf = np.ones(10, dtype=np.float32)
        block = np.full(4, 0.5, dtype=np.float32)
        kept = accept_input_block(buf, block, cooldown)
        self.assertEqual(kept.shape, (14,))
        self.assertEqual(kept.dtype, np.float32)
        cooldown.arm()
        dropped = accept_input_block(kept, block, cooldown)
        self.assertEqual(dropped.size, 0)
        now["t"] = 0.8
        resumed = accept_input_block(dropped, block, cooldown)
        np.testing.assert_allclose(resumed, block)


class RmsGateTest(unittest.TestCase):
    def test_drops_below_min_rms(self) -> None:
        quiet = np.full(400, 0.001, dtype=np.float32)
        self.assertIsNone(prepare_chunk(quiet, min_rms=0.003, target_rms=0.025))

    def test_gain_is_capped(self) -> None:
        quiet_speech = np.full(400, 0.003, dtype=np.float32)
        out = prepare_chunk(quiet_speech, min_rms=0.003, target_rms=0.025)
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out.dtype, np.float32)
        self.assertAlmostEqual(chunk_rms(out), 0.003 * MAX_GAIN, places=5)
        self.assertLess(chunk_rms(out), 0.025)

    def test_reaches_target_when_gain_stays_under_cap(self) -> None:
        audio = np.full(400, 0.01, dtype=np.float32)
        out = prepare_chunk(audio, min_rms=0.003, target_rms=0.025)
        assert out is not None
        self.assertAlmostEqual(chunk_rms(out), 0.025, places=4)

    def test_loud_audio_is_not_gained(self) -> None:
        audio = np.linspace(-0.4, 0.4, 1000, dtype=np.float32)
        out = prepare_chunk(audio, min_rms=0.003, target_rms=0.025)
        assert out is not None
        np.testing.assert_allclose(out, audio.reshape(-1))

    def test_defaults_come_from_config(self) -> None:
        cfg = load_audio_config(ROOT / "config.json", use_cache=False)
        below = np.full(200, cfg.min_rms / 2, dtype=np.float32)
        self.assertIsNone(prepare_chunk(below))
        passing = np.full(200, np.float32(cfg.min_rms), dtype=np.float32)
        out = prepare_chunk(passing)
        assert out is not None
        expected = min(cfg.target_rms, chunk_rms(passing) * MAX_GAIN)
        self.assertAlmostEqual(chunk_rms(out), expected, places=4)

    def test_pcm16_conversion_unchanged(self) -> None:
        block = np.array([0.0, 1.0, -1.0], dtype=np.float32)
        pcm = np.frombuffer(float32_to_pcm16(block), dtype=np.int16)
        self.assertEqual(pcm.tolist(), [0, 32767, -32767])


class ConfigLoaderTest(unittest.TestCase):
    def tearDown(self) -> None:
        clear_audio_config_cache()

    def test_repo_config_defaults(self) -> None:
        cfg = load_audio_config(ROOT / "config.json", use_cache=False)
        self.assertEqual(cfg.sample_rate, 16000)
        self.assertEqual(cfg.cooldown_ms, 1200)
        self.assertGreaterEqual(cfg.cooldown_ms, 800)
        self.assertLessEqual(cfg.cooldown_ms, 1500)
        self.assertEqual(cfg.min_rms, DEFAULT_MIN_RMS)
        self.assertEqual(cfg.target_rms, DEFAULT_TARGET_RMS)
        self.assertGreater(cfg.min_rms, 0.001)
        self.assertLess(cfg.target_rms, 0.05)
        self.assertGreater(cfg.target_rms, cfg.min_rms)
        self.assertEqual(cfg.device_blocklist, DEFAULT_DEVICE_BLOCKLIST)
        self.assertEqual(cfg.device_prefer, DEFAULT_DEVICE_PREFER)

    def test_missing_file_uses_builtins(self) -> None:
        missing = Path(tempfile.gettempdir()) / "audio-cli-no-such-config.json"
        with self.assertNoLogs("engines.config", level="WARNING"):
            cfg = load_audio_config(missing, use_cache=False)
        self.assertEqual(cfg.sample_rate, DEFAULT_SAMPLE_RATE)
        self.assertEqual(cfg.cooldown_ms, DEFAULT_COOLDOWN_MS)
        self.assertEqual(cfg.min_rms, DEFAULT_MIN_RMS)
        self.assertEqual(cfg.target_rms, DEFAULT_TARGET_RMS)
        self.assertEqual(cfg.device_blocklist, DEFAULT_DEVICE_BLOCKLIST)

    def test_invalid_json_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertLogs("engines.config", level="WARNING"):
                cfg = load_audio_config(path, use_cache=False)
        self.assertEqual(cfg.cooldown_ms, DEFAULT_COOLDOWN_MS)
        self.assertEqual(cfg.sample_rate, 16000)

    def test_bad_fields_and_sample_rate_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "sample_rate": 48000,
                        "cooldown_ms": "fast",
                        "min_rms": -1,
                        "target_rms": 5,
                        "device_blocklist": "Zoom",
                        "device_prefer": None,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertLogs("engines.config", level="WARNING") as logs:
                cfg = load_audio_config(path, use_cache=False)
        self.assertEqual(cfg.sample_rate, 16000)
        self.assertEqual(cfg.cooldown_ms, DEFAULT_COOLDOWN_MS)
        self.assertEqual(cfg.min_rms, DEFAULT_MIN_RMS)
        self.assertEqual(cfg.target_rms, DEFAULT_TARGET_RMS)
        self.assertEqual(cfg.device_blocklist, DEFAULT_DEVICE_BLOCKLIST)
        self.assertEqual(cfg.device_prefer, DEFAULT_DEVICE_PREFER)
        self.assertTrue(any("16000" in r.getMessage() for r in logs.records))

    def test_explicit_overrides_and_empty_blocklist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "sample_rate": 16000,
                        "cooldown_ms": 800,
                        "min_rms": 0.01,
                        "target_rms": 0.02,
                        "device_blocklist": [],
                        "device_prefer": ["Headset"],
                    }
                ),
                encoding="utf-8",
            )
            cfg = load_audio_config(path, use_cache=False)
        self.assertEqual(cfg.cooldown_ms, 800)
        self.assertEqual(cfg.min_rms, 0.01)
        self.assertEqual(cfg.target_rms, 0.02)
        self.assertEqual(cfg.device_blocklist, ())
        self.assertEqual(cfg.device_prefer, ("Headset",))
        devices = [
            _dev(0, "Zoom Audio Device"),
            _dev(1, "Headset Mic"),
        ]
        choice = select_input_device(
            devices,
            blocklist=cfg.device_blocklist,
            prefer=cfg.device_prefer,
        )
        self.assertEqual(choice.name, "Headset Mic")
        only_zoom = select_input_device(
            [_dev(0, "Zoom Audio Device")],
            blocklist=cfg.device_blocklist,
            prefer=cfg.device_prefer,
        )
        self.assertEqual(only_zoom.name, "Zoom Audio Device")
        self.assertFalse(only_zoom.blocked_only)

    def test_target_below_min_disables_boost(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps({"min_rms": 0.02, "target_rms": 0.01, "cooldown_ms": 0}),
                encoding="utf-8",
            )
            cfg = load_audio_config(path, use_cache=False)
        self.assertEqual(cfg.min_rms, 0.02)
        self.assertEqual(cfg.target_rms, 0.02)
        self.assertEqual(cfg.cooldown_ms, 0)


if __name__ == "__main__":
    unittest.main()
