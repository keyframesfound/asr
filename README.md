# Audio Live Transcription

Interactive terminal UI (TUI) for live microphone transcription — Cantonese–English workflows.

Arrow-key model picker (OpenCode-style), then a live listening view. Press **m** anytime to switch models.

## Engines

1. **iFlytek live ASR** — cloud streaming 语音听写（流式版）. Credentials in local `.env` (never committed).
2. **SenseVoice Small** — local FunAudioLLM under `models/sensevoice-small` (needs `funasr`).
3. **Parakeet Unified EN** — FluidInference / Hex CoreML under `models/parakeet-unified-en-0.6b-coreml`.
4. **Whisper Large V3 Turbo** — local under `models/whisper-large-v3-turbo`.

Model weights are **not** in git. Keep them under this project’s `models/` on your machine.

## Setup

```bash
cd ~/iflytek-live-asr   # or: git clone https://github.com/keyframesfound/audio-live-transcription
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # fill iFlytek keys for cloud ASR
```

## Run (interactive TUI)

```bash
./run
# or
python main.py
```

### Controls

| Key | Action |
|-----|--------|
| ↑ / ↓ | Move in the model list |
| Enter | Start listening with the selected model |
| m | Reopen the model picker (switch mid-session) |
| Ctrl+C | Stop the current listening session |
| q / Esc | Quit (Esc on picker) |

Live partials and finals appear in the listening view.

### Legacy non-TUI CLI

```bash
python main.py --cli
python main.py --cli --engine whisper
python main.py --list
```

## Secrets

- `.env` is gitignored. Do not commit API keys.
- Use `.env.example` for `XFYUN_APP_ID`, `XFYUN_API_KEY`, `XFYUN_API_SECRET`.
