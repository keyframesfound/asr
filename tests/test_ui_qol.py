"""Level bus, caption reveal planner, config save, and input_device override."""
from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from engines import level as level_bus
from engines.config import AudioConfig, clear_audio_config_cache, save_config
from engines.mic import select_input_device
from tui_app import _parse_setting, _plan_reveal, _tokenize


def _dev(index: int, name: str, channels: int = 1) -> dict:
    return {"index": index, "name": name, "max_input_channels": channels}


class ParseSettingTest(unittest.TestCase):
    def test_float_clamps_to_range(self) -> None:
        from tui_app import SETTINGS

        spec = next(s for s in SETTINGS if s.key == "post_final_cooldown_sec")
        value, err = _parse_setting(spec, "99")
        self.assertIsNone(err)
        self.assertEqual(value, 5.0)
        value, _ = _parse_setting(spec, "-3")
        self.assertEqual(value, 0.0)

    def test_int_rejects_garbage(self) -> None:
        from tui_app import SETTINGS

        spec = next(s for s in SETTINGS if s.key == "mic_blocksize")
        value, err = _parse_setting(spec, "not-a-number")
        self.assertIsNone(value)
        self.assertIn("not a number", err)

    def test_choice_validates(self) -> None:
        from tui_app import SETTINGS

        spec = next(s for s in SETTINGS if s.key == "default_engine")
        value, err = _parse_setting(spec, "sensevoice")
        self.assertIsNone(err)
        self.assertEqual(value, "sensevoice")
        _v, err = _parse_setting(spec, "gpt")
        self.assertIsNotNone(err)


class LevelBusTest(unittest.TestCase):
    def setUp(self) -> None:
        level_bus.reset_level()

    def tearDown(self) -> None:
        level_bus.reset_level()

    def test_snapshot_reflects_last_update(self) -> None:
        t0 = time.monotonic()  # snapshot checks clip expiry on the real clock
        level_bus.update_level(0.2, 0.8, now=t0)
        rms, peak, ts, clipped = level_bus.snapshot()
        self.assertAlmostEqual(rms, 0.2)
        self.assertAlmostEqual(peak, 0.8)
        self.assertEqual(ts, t0)
        self.assertFalse(clipped)

    def test_peak_falls_toward_live_rms(self) -> None:
        t0 = time.monotonic()
        level_bus.update_level(0.5, 0.5, now=t0)
        level_bus.update_level(0.1, 0.1, now=t0 + 5.0)  # long gap: peak fully decayed
        rms, peak, _ts, _clipped = level_bus.snapshot()
        self.assertAlmostEqual(peak, 0.1)
        self.assertAlmostEqual(rms, 0.1)

    def test_clip_lamps_then_expires(self) -> None:
        t0 = time.monotonic()
        level_bus.update_level(0.9, 1.0, now=t0)
        self.assertTrue(level_bus.snapshot(now=t0)[3])
        level_bus.update_level(0.1, 0.1, now=t0 + 2.0)
        self.assertFalse(level_bus.snapshot(now=t0 + 2.0)[3])

    def test_reset_silences_everything(self) -> None:
        level_bus.update_level(0.5, 1.0, now=time.monotonic())
        level_bus.reset_level()
        self.assertEqual(level_bus.snapshot(), (0.0, 0.0, 0.0, False))


class TokenizeTest(unittest.TestCase):
    def test_round_trip_preserves_text_exactly(self) -> None:
        samples = [
            "hello world",
            "你好，世界。",
            "mixed 中文 and english",
            "  leading and trailing  ",
            "",
            "don't stop — now!",
        ]
        for text in samples:
            self.assertEqual("".join(_tokenize(text)), text, text)

    def test_cjk_chars_are_individual_tokens(self) -> None:
        self.assertEqual(_tokenize("你好"), ["你", "好"])

    def test_latin_words_keep_trailing_space(self) -> None:
        self.assertEqual(_tokenize("hi there"), ["hi ", "there"])


class PlanRevealTest(unittest.TestCase):
    def test_append_keeps_base_stable(self) -> None:
        base, pending = _plan_reveal("hello ", "hello world")
        self.assertEqual(base, "hello ")
        self.assertEqual("".join(pending), "world")

    def test_full_reveal_for_new_text(self) -> None:
        base, pending = _plan_reveal("", "new line")
        self.assertEqual(base, "")
        self.assertEqual("".join(pending), "new line")

    def test_identical_target_has_nothing_pending(self) -> None:
        base, pending = _plan_reveal("same", "same")
        self.assertEqual(base, "same")
        self.assertEqual(pending, [])

    def test_mid_word_growth_animates_remainder(self) -> None:
        base, pending = _plan_reveal("hello wor", "hello word")
        self.assertEqual(base, "hello wor")
        self.assertEqual("".join(pending), "d")

    def test_rewrite_trims_to_token_boundary(self) -> None:
        base, pending = _plan_reveal("hello word one", "hello word on")
        self.assertEqual(base, "hello word on")
        self.assertEqual(pending, [])

    def test_diverged_prefix_stays_stable(self) -> None:
        base, pending = _plan_reveal("the cat sat", "the cat is")
        self.assertEqual(base, "the cat ")
        self.assertEqual("".join(pending), "is")

    def test_join_of_base_and_pending_equals_target(self) -> None:
        cases = [
            ("hello wor", "hello word"),
            ("the cat sat", "the cat is"),
            ("", "你好世界"),
            ("你", "你好 world"),
        ]
        for shown, target in cases:
            base, pending = _plan_reveal(shown, target)
            self.assertEqual(base + "".join(pending), target)


class SaveConfigTest(unittest.TestCase):
    def tearDown(self) -> None:
        clear_audio_config_cache()

    def test_merge_preserves_comments_and_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "// note": "keep me",
                        "default_engine": "parakeet",
                        "vad_min_rms": 0.02,
                    }
                ),
                encoding="utf-8",
            )
            save_config({"vad_min_rms": 0.05, "min_final_chars": 4}, path=path)
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(raw["// note"], "keep me")
            self.assertEqual(raw["vad_min_rms"], 0.05)
            self.assertEqual(raw["min_final_chars"], 4)
            keys = list(json.loads(path.read_text(encoding="utf-8")).keys())
            self.assertEqual(keys, ["// note", "default_engine", "vad_min_rms", "min_final_chars"])

    def test_creates_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub" / "config.json"
            save_config({"min_final_chars": 3}, path=path)
            self.assertTrue(path.exists())
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["min_final_chars"], 3)


class InputDeviceOverrideTest(unittest.TestCase):
    def tearDown(self) -> None:
        clear_audio_config_cache()

    def _with_cfg(self, override: str):
        return AudioConfig(input_device=override)

    def test_explicit_device_wins_over_blocklist_and_prefer(self) -> None:
        devices = [
            _dev(0, "BlackHole 2ch"),
            _dev(1, "MacBook Pro Microphone"),
        ]
        with mock.patch(
            "engines.mic.load_audio_config", return_value=self._with_cfg("BlackHole 2ch")
        ):
            choice = select_input_device(devices)
        self.assertEqual((choice.index, choice.name), (0, "BlackHole 2ch"))
        self.assertFalse(choice.blocked_only)

    def test_match_is_case_insensitive_exact(self) -> None:
        devices = [_dev(2, "airpods pro"), _dev(1, "MacBook Pro Microphone")]
        with mock.patch(
            "engines.mic.load_audio_config", return_value=self._with_cfg("AirPods Pro")
        ):
            choice = select_input_device(devices)
        self.assertEqual(choice.index, 2)

    def test_unknown_name_falls_back_to_auto(self) -> None:
        devices = [_dev(1, "MacBook Pro Microphone")]
        with mock.patch(
            "engines.mic.load_audio_config", return_value=self._with_cfg("Ghost Mic")
        ):
            with self.assertLogs("engines.mic", level="WARNING"):
                choice = select_input_device(devices)
        self.assertEqual(choice.name, "MacBook Pro Microphone")

    def test_empty_override_uses_prefer_order(self) -> None:
        devices = [_dev(1, "MacBook Pro Microphone"), _dev(2, "AirPods")]
        with mock.patch(
            "engines.mic.load_audio_config", return_value=self._with_cfg("")
        ):
            choice = select_input_device(devices)
        self.assertEqual(choice.name, "AirPods")


if __name__ == "__main__":
    unittest.main()
