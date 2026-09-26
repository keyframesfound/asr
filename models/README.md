# Local models

Weights live here (gitignored except this file). Manage them from inside the
app: the model picker shows install state per engine — **i** downloads the
highlighted model, **u** (twice) uninstalls it. All three pull a tarball from
the self-hosted sources in `engines/weights.py` — R2 bucket `asr-weights`
first, the GitHub Releases tag `models-v1` second — with the HF hub as last
resort. Sources:

| Folder | Upstream model |
|--------|----------------|
| `parakeet-mlx/` | mlx-community/parakeet-tdt-0.6b-v3 (HF hub cache layout) |
| `sensevoice-small/` | FunAudioLLM/SenseVoiceSmall |
| `whisper-large-v3-turbo/` | openai/whisper-large-v3-turbo |
| `parakeet-unified-en-0.6b-coreml/` | FluidInference/parakeet-unified-en-0.6b-coreml (legacy, unused) |

The R2 bucket (`pub-f6dba6d3598843a0bf81e6cb54c57d5b.r2.dev`, APAC) holds
`models-v1/<name>.tar` for all three engines. GitHub still carries the
SenseVoice/Whisper tars under the `models-v1` release; Parakeet exceeds the
2 GiB release-asset cap there, so its GitHub entry 404s and falls through.
Re-upload with `wrangler r2 object put asr-weights/models-v1/<name>.tar
--file <tar> --remote`.

Do not commit model binaries to GitHub.
