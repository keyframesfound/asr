# Local models

Weights live here (gitignored except this file). Manage them from inside the
app: the model picker shows install state per engine — **i** downloads the
highlighted model, **u** (twice) uninstalls it. SenseVoice/Whisper pull a
tarball from this repo's GitHub Releases (tag `models-v1`) with the HF hub as
fallback; Parakeet comes from the HF hub on first use. Sources:

| Folder | Source |
|--------|--------|
| `parakeet-mlx/` | mlx-community/parakeet-tdt-0.6b-v3 (HF hub; > 2 GiB, exceeds GitHub release asset cap) |
| `sensevoice-small/` | Releases `models-v1` ← FunAudioLLM/SenseVoiceSmall |
| `whisper-large-v3-turbo/` | Releases `models-v1` ← openai/whisper-large-v3-turbo |
| `parakeet-unified-en-0.6b-coreml/` | FluidInference/parakeet-unified-en-0.6b-coreml (legacy, unused) |

```bash
python scripts/download_models.py
```

That script is safe to re-run. It skips an engine when its files are already here. iFlytek is cloud-only (nothing to download). Hex / CoreML trees are not fetched; Parakeet uses `parakeet-mlx` only.

Whisper and SenseVoice prefer the GitHub release `https://github.com/keyframesfound/asr/releases/download/models-v1/<dirname>.tar` (`whisper-large-v3-turbo.tar`, `sensevoice-small.tar`). The tarball matches `models/<dirname>/`. Hugging Face LFS can stall after a few MB when `cdn-lfs.huggingface.co` is unreachable; if the release fetch fails, the script falls back to the Hugging Face snapshot. Parakeet stays on the Hugging Face hub cache (no GitHub asset; the weights are over the 2 GiB release limit).

| Folder | Hugging Face repo |
|--------|-------------------|
| `sensevoice-small/` | FunAudioLLM/SenseVoiceSmall |
| `parakeet-mlx/` | mlx-community/parakeet-tdt-0.6b-v3 (hub cache layout) |
| `whisper-large-v3-turbo/` | openai/whisper-large-v3-turbo |

About 5 GB total. A leftover `parakeet-unified-en-0.6b-coreml/` directory is unused by the portable engine and is not part of setup.
