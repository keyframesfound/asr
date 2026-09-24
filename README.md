# Audio Live Transcription

Terminal-only live microphone transcription for Cantonese–English work.

## Engines

1. **iFlytek live ASR** — cloud streaming 语音听写（流式版）. Credentials stay in a local `.env` (never committed).
2. **SenseVoice Small** — local FunAudioLLM weights under `models/sensevoice-small` (needs `funasr`).
3. **Parakeet Unified EN** — FluidInference / Hex CoreML package under `models/parakeet-unified-en-0.6b-coreml` (run via Hex/FluidAudio; CLI reports model presence).
4. **Whisper Large V3 Turbo** — local `openai/whisper-large-v3-turbo` under `models/whisper-large-v3-turbo`.

Model weights are **not** in git. Keep them under `~/iflytek-live-asr/models` (or this repo’s `models/`) on your machine.

## Setup

```bash
cd ~/iflytek-live-asr   # or clone this repo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # fill iFlytek keys for cloud ASR
```

## Run

```bash
./run
# or
python main.py
# or skip the menu:
python main.py --engine whisper
```

Speak into the mic. Transcripts print to stdout. **Ctrl+C** stops and prints a short summary.

## Secrets

- `.env` is gitignored. Do not commit API keys.
- Use `.env.example` as a template for `XFYUN_APP_ID`, `XFYUN_API_KEY`, `XFYUN_API_SECRET`.
