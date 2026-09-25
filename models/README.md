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

Do not commit model binaries to GitHub.
