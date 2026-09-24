# Local models

Weights are gitignored. Do not commit model binaries.

After `pip install -r requirements.txt`:

```bash
python scripts/download_models.py
```

That script is safe to re-run. It skips an engine when its files are already here. iFlytek is cloud-only (nothing to download). Hex / CoreML trees are not fetched; Parakeet uses `parakeet-mlx` only.

| Folder | Hugging Face repo |
|--------|-------------------|
| `sensevoice-small/` | FunAudioLLM/SenseVoiceSmall |
| `parakeet-mlx/` | mlx-community/parakeet-tdt-0.6b-v3 (hub cache layout) |
| `whisper-large-v3-turbo/` | openai/whisper-large-v3-turbo |

About 5 GB total. A leftover `parakeet-unified-en-0.6b-coreml/` directory is unused by the portable engine and is not part of setup.
