"""Presence checks for scripts/download_models.py. No network and no weights."""
from __future__ import annotations

import io
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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
    if link.is_symlink():
        link.unlink()
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

            def _release_boom(_url: str, _dest: Path) -> None:
                raise AssertionError("release fetcher should not run when weights exist")

            code = dm.download_local_models(
                root, downloader=_boom, release_fetcher=_release_boom
            )
            self.assertEqual(code, 0)

    def test_release_first_for_all_engines_and_hub_unused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            release_urls: list[str] = []
            hub_repos: list[object] = []

            def _release(url: str, dest: Path) -> None:
                release_urls.append(url)
                if url == dm.tarball_url(dm.R2_BASE_URL, dm.WHISPER_DIRNAME):
                    self.assertEqual(dest, dm.model_dest(root, "whisper"))
                    _plant_whisper(dest)
                elif url == dm.tarball_url(dm.R2_BASE_URL, dm.SENSEVOICE_DIRNAME):
                    self.assertEqual(dest, dm.model_dest(root, "sensevoice"))
                    _plant_sensevoice(dest)
                elif url == dm.tarball_url(dm.R2_BASE_URL, dm.PARAKEET_DIRNAME):
                    self.assertEqual(dest, dm.model_dest(root, "parakeet"))
                    _plant_parakeet(dest)
                else:
                    raise AssertionError(url)

            def _hub(**_kwargs: object) -> str:
                raise AssertionError("hub should not run when tarballs succeed")

            self.assertEqual(
                dm.download_local_models(
                    root, downloader=_hub, release_fetcher=_release
                ),
                0,
            )
            self.assertEqual(
                release_urls,
                [
                    dm.tarball_url(dm.R2_BASE_URL, dm.WHISPER_DIRNAME),
                    dm.tarball_url(dm.R2_BASE_URL, dm.SENSEVOICE_DIRNAME),
                    dm.tarball_url(dm.R2_BASE_URL, dm.PARAKEET_DIRNAME),
                ],
            )
            self.assertEqual(hub_repos, [])

    def test_release_failure_falls_back_to_huggingface(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            release_urls: list[str] = []
            hub_calls: list[dict[str, object]] = []

            def _release(url: str, _dest: Path) -> None:
                release_urls.append(url)
                raise OSError("cdn unreachable")

            def _hub(**kwargs: object) -> str:
                hub_calls.append(dict(kwargs))
                repo = kwargs["repo_id"]
                if repo == dm.WHISPER_REPO_ID:
                    self.assertNotIn("cache_dir", kwargs)
                    self.assertNotIn("allow_patterns", kwargs)
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

            self.assertEqual(
                dm.download_local_models(
                    root, downloader=_hub, release_fetcher=_release
                ),
                0,
            )
            self.assertEqual(
                release_urls,
                [
                    dm.tarball_url(dm.R2_BASE_URL, dm.WHISPER_DIRNAME),
                    dm.tarball_url(dm.RELEASE_BASE_URL, dm.WHISPER_DIRNAME),
                    dm.tarball_url(dm.R2_BASE_URL, dm.SENSEVOICE_DIRNAME),
                    dm.tarball_url(dm.RELEASE_BASE_URL, dm.SENSEVOICE_DIRNAME),
                    dm.tarball_url(dm.R2_BASE_URL, dm.PARAKEET_DIRNAME),
                    dm.tarball_url(dm.RELEASE_BASE_URL, dm.PARAKEET_DIRNAME),
                ],
            )
            self.assertEqual(
                [call["repo_id"] for call in hub_calls],
                [dm.WHISPER_REPO_ID, dm.SENSEVOICE_REPO_ID, dm.PARAKEET_REPO_ID],
            )

    def test_unusable_release_tree_falls_back_to_huggingface(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hub_repos: list[object] = []
            planted: set[str] = set()

            def _release(url: str, dest: Path) -> None:
                key = url.rsplit("/", 1)[-1]
                if key in planted:
                    return  # second source, same unusable tree
                planted.add(key)
                if key == f"{dm.WHISPER_DIRNAME}.tar":
                    _plant_whisper(dest, weight_bytes=128)
                elif key == f"{dm.SENSEVOICE_DIRNAME}.tar":
                    _plant_sensevoice(dest, weight_bytes=128)
                elif key == f"{dm.PARAKEET_DIRNAME}.tar":
                    _plant_parakeet(dest, weight_bytes=128)
                else:
                    raise AssertionError(url)

            def _hub(**kwargs: object) -> str:
                repo = kwargs["repo_id"]
                hub_repos.append(repo)
                if repo == dm.WHISPER_REPO_ID:
                    _plant_whisper(Path(str(kwargs["local_dir"])))
                elif repo == dm.SENSEVOICE_REPO_ID:
                    _plant_sensevoice(Path(str(kwargs["local_dir"])))
                elif repo == dm.PARAKEET_REPO_ID:
                    _plant_parakeet(Path(str(kwargs["cache_dir"])))
                else:
                    raise AssertionError(repo)
                return "ok"

            self.assertEqual(
                dm.download_local_models(
                    root, downloader=_hub, release_fetcher=_release
                ),
                0,
            )
            self.assertEqual(
                hub_repos,
                [dm.WHISPER_REPO_ID, dm.SENSEVOICE_REPO_ID, dm.PARAKEET_REPO_ID],
            )

    def test_network_error_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def _release(_url: str, _dest: Path) -> None:
                raise OSError("connection timed out")

            def _fail(**_kwargs: object) -> str:
                raise OSError("HTTPSConnectionPool: connection timed out")

            code = dm.download_local_models(
                root, downloader=_fail, release_fetcher=_release
            )
            self.assertEqual(code, 1)

    def test_clears_offline_env_before_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            try:
                def _release(_url: str, _dest: Path) -> None:
                    self.assertNotIn("HF_HUB_OFFLINE", os.environ)
                    self.assertNotIn("TRANSFORMERS_OFFLINE", os.environ)
                    raise OSError("network is unreachable")

                def _fail(**_kwargs: object) -> str:
                    self.assertNotIn("HF_HUB_OFFLINE", os.environ)
                    self.assertNotIn("TRANSFORMERS_OFFLINE", os.environ)
                    raise OSError("network is unreachable")

                self.assertEqual(
                    dm.download_local_models(
                        root, downloader=_fail, release_fetcher=_release
                    ),
                    1,
                )
            finally:
                os.environ.pop("HF_HUB_OFFLINE", None)
                os.environ.pop("TRANSFORMERS_OFFLINE", None)


class ReleaseArchiveTest(unittest.TestCase):
    def test_tarball_urls_and_per_source_checksums(self) -> None:
        self.assertEqual(
            dm.tarball_url(dm.R2_BASE_URL, dm.WHISPER_DIRNAME),
            "https://pub-f6dba6d3598843a0bf81e6cb54c57d5b.r2.dev"
            "/models-v1/whisper-large-v3-turbo.tar",
        )
        self.assertEqual(
            dm.tarball_url(dm.RELEASE_BASE_URL, dm.SENSEVOICE_DIRNAME),
            "https://github.com/keyframesfound/asr/releases/download/"
            "models-v1/sensevoice-small.tar",
        )
        # R2 ships all three; the GitHub release only the dir engines.
        self.assertEqual(
            set(dm.R2_SHA256),
            {
                f"{dm.WHISPER_DIRNAME}.tar",
                f"{dm.SENSEVOICE_DIRNAME}.tar",
                f"{dm.PARAKEET_DIRNAME}.tar",
            },
        )
        self.assertNotIn(f"{dm.PARAKEET_DIRNAME}.tar", dm.RELEASE_SHA256)
        self.assertEqual(
            dm._pinned_sha256(dm.tarball_url(dm.R2_BASE_URL, dm.PARAKEET_DIRNAME)),
            dm.R2_SHA256[f"{dm.PARAKEET_DIRNAME}.tar"],
        )
        self.assertEqual(dm._pinned_sha256("https://example.com/anything.tar"), "")
        self.assertEqual(dm._tarball_dirname("parakeet"), dm.PARAKEET_DIRNAME)

    def test_source_chain_env_override(self) -> None:
        with mock.patch.dict(os.environ, {"ASR_WEIGHTS_BASE_URL": "off"}):
            self.assertEqual(dm._tarball_sources(), ())
        with mock.patch.dict(os.environ, {"ASR_WEIGHTS_BASE_URL": "hf"}):
            self.assertEqual(dm._tarball_sources(), ())
        with mock.patch.dict(
            os.environ, {"ASR_WEIGHTS_BASE_URL": "https://mirror.example/w"}
        ):
            self.assertEqual(dm._tarball_sources(), ("https://mirror.example/w",))
        with mock.patch.dict(os.environ, {"ASR_WEIGHTS_BASE_URL": ""}):
            self.assertEqual(
                dm._tarball_sources(), (dm.R2_BASE_URL, dm.RELEASE_BASE_URL)
            )

    def test_extract_strips_root_and_skips_appledouble(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage = root / "stage" / dm.WHISPER_DIRNAME
            _touch(stage / "config.json", 4)
            _touch(stage / "._config.json", 4)
            _touch(stage / "nested" / "note.txt", 3)
            archive = root / "model.tar"
            with tarfile.open(archive, "w") as tar:
                tar.add(stage, arcname=dm.WHISPER_DIRNAME)
            dest = root / "models" / dm.WHISPER_DIRNAME
            dm.extract_release_archive(archive, dest)
            self.assertEqual((dest / "config.json").read_bytes(), b"\x00" * 4)
            self.assertTrue((dest / "nested" / "note.txt").is_file())
            self.assertFalse((dest / "._config.json").exists())
            self.assertFalse((dest / dm.WHISPER_DIRNAME).exists())

    def test_extract_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "bad.tar"
            dest = root / "models" / dm.WHISPER_DIRNAME
            payload = b"nope"
            with tarfile.open(archive, "w") as tar:
                info = tarfile.TarInfo(f"{dm.WHISPER_DIRNAME}/../../escaped.txt")
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
            with self.assertRaises(ValueError):
                dm.extract_release_archive(archive, dest)
            self.assertFalse((root / "escaped.txt").exists())
            self.assertFalse((root / "models" / "escaped.txt").exists())


class GitignoreTest(unittest.TestCase):
    def test_model_weights_stay_untracked(self) -> None:
        text = (ROOT / ".gitignore").read_text()
        self.assertIn("models/*/", text)
        self.assertIn("!models/README.md", text)
        self.assertIn("models/parakeet-mlx/", text)
