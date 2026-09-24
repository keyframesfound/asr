"""Presence checks for scripts/download_models.py. No network and no weights."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import download_models as dm  # noqa: E402


def _touch(path: Path, size: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.truncate(size)


def _plant_whisper(model_dir: Path, *, weight_bytes: int = dm.MIN_WEIGHT_BYTES) -> None:
    _touch(model_dir / "config.json")
    _touch(model_dir / "preprocessor_config.json")
    _touch(model_dir / "tokenizer.json")
    _touch(model_dir / "model.safetensors", weight_bytes)


def _plant_sensevoice(model_dir: Path, *, weight_bytes: int = dm.MIN_WEIGHT_BYTES) -> None:
    for name in dm.SENSEVOICE_FILES:
        if name == "model.pt":
            _touch(model_dir / name, weight_bytes)
        else:
            _touch(model_dir / name)


def _plant_parakeet(cache_dir: Path, *, weight_bytes: int = dm.MIN_WEIGHT_BYTES) -> Path:
    repo = cache_dir / ("models--" + dm.PARAKEET_REPO_ID.replace("/", "--"))
    snap = repo / "snapshots" / "rev"
    blob = repo / "blobs" / "weights"
    _touch(blob, weight_bytes)
    _touch(snap / "config.json")
    link = snap / "model.safetensors"
    link.symlink_to(Path("..") / ".." / "blobs" / "weights")
    return snap


class PresenceTest(unittest.TestCase):
    def test_empty_and_pointer_files_are_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            whisper = root / "whisper"
            sense = root / "sense"
            cache = root / "parakeet"
            self.assertFalse(dm.whisper_weights_ready(whisper))
            self.assertFalse(dm.sensevoice_weights_ready(sense))
            self.assertFalse(dm.parakeet_weights_ready(cache))

            _plant_whisper(whisper, weight_bytes=128)
            _plant_sensevoice(sense, weight_bytes=128)
            _plant_parakeet(cache, weight_bytes=128)
            self.assertFalse(dm.whisper_weights_ready(whisper))
            self.assertFalse(dm.sensevoice_weights_ready(sense))
            self.assertFalse(dm.parakeet_weights_ready(cache))

    def test_complete_trees_are_ready_including_parakeet_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            whisper = root / "whisper"
            sense = root / "sense"
            cache = root / "parakeet"
            _plant_whisper(whisper)
            _plant_sensevoice(sense)
            snap = _plant_parakeet(cache)
            self.assertTrue(dm.whisper_weights_ready(whisper))
            self.assertTrue(dm.sensevoice_weights_ready(sense))
            self.assertTrue(dm.parakeet_weights_ready(cache))
            self.assertTrue((snap / "model.safetensors").is_symlink())
            self.assertGreaterEqual(
                (snap / "model.safetensors").stat().st_size,
                dm.MIN_WEIGHT_BYTES,
            )

    def test_incomplete_marker_forces_redownload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            whisper = root / "whisper"
            _plant_whisper(whisper)
            _touch(whisper / "model.safetensors.incomplete", 32)
            self.assertFalse(dm.whisper_weights_ready(whisper))

    def test_paths_match_engine_modules(self) -> None:
        whisper = (ROOT / "engines" / "whisper.py").read_text()
        sense = (ROOT / "engines" / "sensevoice.py").read_text()
        parakeet = (ROOT / "engines" / "parakeet.py").read_text()
        models_readme = (ROOT / "models" / "README.md").read_text()
        self.assertIn(dm.WHISPER_DIRNAME, whisper)
        self.assertIn(dm.WHISPER_REPO_ID, models_readme)
        self.assertIn(dm.SENSEVOICE_DIRNAME, sense)
        self.assertIn(dm.SENSEVOICE_REPO_ID, models_readme)
        self.assertIn(dm.PARAKEET_REPO_ID, parakeet)
        self.assertIn(dm.PARAKEET_DIRNAME, parakeet)
        self.assertIn(
            'repo_dir_name = "models--" + MODEL_ID.replace("/", "--")',
            parakeet,
        )
        self.assertIn("snapshots/*/config.json", parakeet)

    def test_plan_is_three_local_engines_only(self) -> None:
        repos = [spec.repo_id for spec in dm.MODELS]
        self.assertEqual(
            repos,
            [dm.WHISPER_REPO_ID, dm.SENSEVOICE_REPO_ID, dm.PARAKEET_REPO_ID],
        )
        joined = " ".join(repos)
        self.assertNotIn("coreml", joined.lower())
        self.assertNotIn("iflytek", joined.lower())
        self.assertNotIn("hex", joined.lower())


class DownloadSkipTest(unittest.TestCase):
    def test_skips_present_weights_without_calling_downloader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _plant_whisper(dm.model_dest(root, "whisper"))
            _plant_sensevoice(dm.model_dest(root, "sensevoice"))
            _plant_parakeet(dm.model_dest(root, "parakeet"))

            def _boom(**_kwargs: object) -> str:
                raise AssertionError("downloader should not run when weights exist")

            code = dm.download_local_models(root, downloader=_boom)
            self.assertEqual(code, 0)

    def test_downloads_missing_engine_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls: list[dict[str, object]] = []

            def _fake(**kwargs: object) -> str:
                calls.append(dict(kwargs))
                repo = kwargs["repo_id"]
                if repo == dm.WHISPER_REPO_ID:
                    _plant_whisper(Path(str(kwargs["local_dir"])))
                elif repo == dm.SENSEVOICE_REPO_ID:
                    self.assertEqual(
                        tuple(kwargs["allow_patterns"]),
                        dm.SENSEVOICE_FILES,
                    )
                    _plant_sensevoice(Path(str(kwargs["local_dir"])))
                elif repo == dm.PARAKEET_REPO_ID:
                    self.assertNotIn("local_dir", kwargs)
                    _plant_parakeet(Path(str(kwargs["cache_dir"])))
                else:
                    raise AssertionError(repo)
                return "ok"

            self.assertEqual(dm.download_local_models(root, downloader=_fake), 0)
            self.assertEqual(
                [call["repo_id"] for call in calls],
                [dm.WHISPER_REPO_ID, dm.SENSEVOICE_REPO_ID, dm.PARAKEET_REPO_ID],
            )
            calls.clear()
            self.assertEqual(dm.download_local_models(root, downloader=_fake), 0)
            self.assertEqual(calls, [])

    def test_network_error_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def _fail(**_kwargs: object) -> str:
                raise OSError("HTTPSConnectionPool: connection timed out")

            code = dm.download_local_models(root, downloader=_fail)
            self.assertEqual(code, 1)

    def test_clears_offline_env_before_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            try:
                def _fail(**_kwargs: object) -> str:
                    self.assertNotIn("HF_HUB_OFFLINE", os.environ)
                    self.assertNotIn("TRANSFORMERS_OFFLINE", os.environ)
                    raise OSError("network is unreachable")

                self.assertEqual(dm.download_local_models(root, downloader=_fail), 1)
            finally:
                os.environ.pop("HF_HUB_OFFLINE", None)
                os.environ.pop("TRANSFORMERS_OFFLINE", None)


class GitignoreTest(unittest.TestCase):
    def test_model_weights_stay_untracked(self) -> None:
        text = (ROOT / ".gitignore").read_text()
        self.assertIn("models/*/", text)
        self.assertIn("!models/README.md", text)
        self.assertIn("models/parakeet-mlx/", text)
