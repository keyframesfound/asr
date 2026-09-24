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

### Hong Kong CJK font（繁體中文・香港）

Captions and the transcript log use the terminal’s font. Textual CSS cannot set a typeface (`font-family` is rejected as an invalid property, and the TUI will not start), so this app does not declare one.

For Hong Kong Traditional Chinese, point **Terminal.app** at **PingFang HK**（蘋方-港）. If that face is missing, use **Noto Sans HK**, then **Noto Sans TC**.

1. Terminal → **Settings…**（設定，⌘,）→ **Profiles**（描述檔）→ **Text**（文字）.
2. Under Font（字體）, click **Change…** and choose **PingFang HK**, Regular or Medium, about 14–16 pt.
3. Open a new window so the profile font applies, then run `./run`.

If borders look uneven, set character spacing to 1 and nudge line spacing (about 0.8). To screenshot-check, leave PingFang HK selected, start a listening session, and capture the view while a Chinese line is on the caption or in the log.

### Legacy non-TUI CLI

```bash
python main.py --cli
python main.py --cli --engine whisper
python main.py --list
```

## Secrets

- `.env` is gitignored. Do not commit API keys.
- Use `.env.example` for `XFYUN_APP_ID`, `XFYUN_API_KEY`, `XFYUN_API_SECRET`.
