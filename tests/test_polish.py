"""Full-session polish: pause-snapped chunking, joins, config gate, plumbing."""
from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from engines import audio_util
from engines.base import SessionSummary


class IsCjkTest(unittest.TestCase):
    def test_ranges(self) -> None:
        self.assertTrue(audio_util.is_cjk("香"))
        self.assertTrue(audio_util.is_cjk("，"))
        self.assertFalse(audio_util.is_cjk("a"))
        self.assertFalse(audio_util.is_cjk(" "))


class JoinTranscriptPartsTest(unittest.TestCase):
    def test_latin_parts_get_spaces(self) -> None:
        self.assertEqual(
            audio_util.join_transcript_parts(["hello", "", "world today"]),
            "hello world today",
        )

    def test_cjk_seam_has_no_space(self) -> None:
        self.assertEqual(audio_util.join_transcript_parts(["今日", "好熱"]), "今日好熱")

    def test_mixed_seam(self) -> None:
        self.assertEqual(audio_util.join_transcript_parts(["我說", "hello"]), "我說hello")
        self.assertEqual(audio_util.join_transcript_parts(["hello", "世界"]), "hello世界")


class PlanSilentChunksTest(unittest.TestCase):
    SR = 16000

    def _audio_with_dip(self, total_sec: float, dip_sec: float) -> np.ndarray:
        n = int(total_sec * self.SR)
        audio = np.full(n, 0.1, dtype=np.float32)
        start = int(dip_sec * self.SR)
        width = int(0.4 * self.SR)
        audio[start : start + width] = 0.0
        return audio

    def test_empty_audio(self) -> None:
        self.assertEqual(audio_util.plan_silent_chunks(np.zeros(0), self.SR, max_sec=10), [])

    def test_short_audio_is_one_chunk(self) -> None:
        audio = np.ones(self.SR * 5, dtype=np.float32)
        chunks = audio_util.plan_silent_chunks(audio, self.SR, max_sec=10.0)
        self.assertEqual(chunks, [(0, audio.size)])

    def test_covers_whole_range_under_max(self) -> None:
        audio = np.ones(self.SR * 60, dtype=np.float32)  # constant tone, no dips
        chunks = audio_util.plan_silent_chunks(audio, self.SR, max_sec=10.0)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0][0], 0)
        self.assertEqual(chunks[-1][1], audio.size)
        for (_, end), (start, _) in zip(chunks, chunks[1:]):
            self.assertEqual(end, start)  # contiguous
        for start, end in chunks:
            self.assertLessEqual(end - start, int(10.0 * self.SR))

    def test_boundary_snaps_into_quiet_dip(self) -> None:
        # 30 s loud audio with a single silent gap at 10 s; 12 s max chunks.
        audio = self._audio_with_dip(30.0, dip_sec=10.0)
        chunks = audio_util.plan_silent_chunks(audio, self.SR, max_sec=12.0, min_sec=4.0)
        self.assertEqual(chunks[0][0], 0)
        first_end = chunks[0][1]
        dip_lo = int(10.0 * self.SR)
        dip_hi = dip_lo + int(0.4 * self.SR)
        self.assertGreaterEqual(first_end, dip_lo)
        self.assertLessEqual(first_end, dip_hi)


class PolishEnabledTest(unittest.TestCase):
    def test_default_on(self) -> None:
        with mock.patch.object(audio_util, "load_config", return_value={}):
            self.assertTrue(audio_util.polish_enabled())

    def test_respects_false(self) -> None:
        with mock.patch.object(audio_util, "load_config", return_value={"post_stop_polish": False}):
            self.assertFalse(audio_util.polish_enabled())


class _FakeEngine:
    def __init__(self, result: str | None = None, exc: Exception | None = None) -> None:
        self.result = result
        self.exc = exc
        self.calls: list[tuple] = []

    def polish(self, audio: np.ndarray, *, sample_rate: int = 16000) -> str | None:
        self.calls.append((audio.copy(), sample_rate))
        if self.exc is not None:
            raise self.exc
        return self.result


class PolishSessionTest(unittest.TestCase):
    SR = 16000

    def _blocks(self, sec: float) -> list[np.ndarray]:
        return [np.ones(self.SR, dtype=np.float32) * 0.1 for _ in range(int(sec))]

    def _run(self, engine: _FakeEngine, blocks: list[np.ndarray], error: str | None = None):
        summary = SessionSummary(engine="test", error=error)
        notes: list[str] = []
        audio_util.polish_session(
            engine,
            blocks,
            sample_rate=self.SR,
            on_partial=notes.append,
            summary=summary,
        )
        return summary, notes

    def test_polish_replaces_summary(self) -> None:
        engine = _FakeEngine(result="polished text")
        summary, notes = self._run(engine, self._blocks(3))
        self.assertEqual(summary.polished, "polished text")
        self.assertAlmostEqual(summary.recorded_sec, 3.0)
        self.assertEqual(notes, ["Polishing session audio…"])

    def test_skipped_when_disabled(self) -> None:
        engine = _FakeEngine(result="polished text")
        with mock.patch.object(audio_util, "load_config", return_value={"post_stop_polish": False}):
            summary, notes = self._run(engine, self._blocks(3))
        self.assertIsNone(summary.polished)
        self.assertEqual(engine.calls, [])
        self.assertEqual(notes, [])

    def test_skipped_on_engine_error(self) -> None:
        engine = _FakeEngine(result="polished text")
        summary, _ = self._run(engine, self._blocks(3), error="boom")
        self.assertIsNone(summary.polished)
        self.assertEqual(engine.calls, [])

    def test_skipped_when_too_short(self) -> None:
        engine = _FakeEngine(result="polished text")
        summary, _ = self._run(engine, self._blocks(0.2))
        self.assertIsNone(summary.polished)
        self.assertEqual(engine.calls, [])

    def test_polish_failure_keeps_live_transcript(self) -> None:
        engine = _FakeEngine(exc=RuntimeError("gpu sad"))
        summary, _ = self._run(engine, self._blocks(3))
        self.assertIsNone(summary.polished)  # best-effort: no raise, no replace


class ParakeetPolishGuardTest(unittest.TestCase):
    def test_no_warm_model_returns_none(self) -> None:
        import engines.parakeet as parakeet

        engine = parakeet.ParakeetEngine()
        with mock.patch.object(parakeet, "_MODEL", None), mock.patch.object(
            parakeet, "_MODEL_STREAM", None
        ):
            self.assertIsNone(engine.polish(np.ones(16000, dtype=np.float32)))

    def test_too_short_returns_none(self) -> None:
        import engines.parakeet as parakeet

        engine = parakeet.ParakeetEngine()
        with mock.patch.object(parakeet, "_MODEL", object()), mock.patch.object(
            parakeet, "_MODEL_STREAM", object()
        ):
            self.assertIsNone(engine.polish(np.ones(100, dtype=np.float32)))


class ClassifyPolishTest(unittest.TestCase):
    def test_polishing_status(self) -> None:
        from tui_app import _classify_partial

        self.assertEqual(_classify_partial("Polishing session audio…"), "polishing")
        self.assertEqual(_classify_partial("Transcribing…"), "transcribing")
        self.assertEqual(_classify_partial("hello there"), "draft")


class ExportBodyTest(unittest.TestCase):
    def test_plain_lines_no_timestamps(self) -> None:
        from tui_app import _export_body

        self.assertEqual(_export_body(["one two", "three"]), "one two\nthree\n")
        self.assertEqual(_export_body([]), "")
        self.assertNotIn("%H", _export_body(["x"]))


class SummaryShortTextTest(unittest.TestCase):
    def test_polished_preferred_over_preview(self) -> None:
        summary = SessionSummary(engine="Parakeet", finals=["raw draft text"])
        summary.polished = "Polished full text."
        text = summary.short_text()
        self.assertIn("Polished: Polished full text.", text)
        self.assertNotIn("Transcript preview", text)

    def test_defaults(self) -> None:
        summary = SessionSummary(engine="Parakeet")
        self.assertIsNone(summary.polished)
        self.assertEqual(summary.recorded_sec, 0.0)


if __name__ == "__main__":
    unittest.main()
