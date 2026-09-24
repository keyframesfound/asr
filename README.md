# asr

Minimal terminal UI for live microphone transcription — Cantonese–English workflows.

Claude Code / OpenCode–inspired TUI: slim chrome, keyboard-first. Arrow-key model picker, then a calm live caption + transcript log. Press **m** anytime to switch models.

## Engines

1. **iFlytek live ASR** — cloud streaming 语音听写（流式版）. Credentials in local `.env` (never committed).
2. **SenseVoice Small** — local FunAudioLLM under `models/sensevoice-small` (needs `funasr`).
3. **Parakeet Unified EN** — local Parakeet TDT via **parakeet-mlx** (`mlx-community/parakeet-tdt-0.6b-v3`); weights download into `models/parakeet-mlx` on first use (Apple Silicon).
4. **Whisper Large V3 Turbo** — local under `models/whisper-large-v3-turbo`.

Model weights are **not** in git. Keep them under this project’s `models/` on your machine.

## Setup

```bash
cd ~/iflytek-live-asr   # or: git clone https://github.com/keyframesfound/asr
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # fill iFlytek keys for cloud ASR
```

## Run (interactive TUI)

```bash
asr          # global command → ./run → TUI
./run
# or
python main.py
```

### Controls

| Key | Action |
|-----|--------|
| ↑ / ↓ | Move in the model list |
| Enter | Start listening with the selected model |
| Space | Start / stop live |
| e | Export transcript as `.txt` |
| m | Reopen the model picker (switch mid-session) |
| Ctrl+C | Stop the current listening session |
| q / Esc | Quit (Esc on picker) |

Caption shows **Listening…** / **Transcribing…**, then drafts; finals land in the dim-timestamped log.

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


## Config

Tunable knobs live in `config.json` (keys starting with `//` are comments):

| Key | Default | Meaning |
|-----|---------|---------|
| `default_engine` | `parakeet` | Model picker / CLI default (English live). Use `sensevoice` for Cantonese. |
| `parakeet_feed_sec` | `0.4` | Audio seconds per Parakeet `add_audio` (~0.3–0.5). |
| `post_final_cooldown_sec` | `1.25` | Discard mic audio after each final (echo/self-print). |
| `mic_blocksize` | `4096` | sounddevice frames @ 16 kHz (fewer Python wakeups). |
| `vad_min_rms` | `0.02` | Speech energy gate. |
| `min_final_chars` | `8` | Prefer not emitting tiny Parakeet finals unless punctuated. |


## Secrets

- `.env` is gitignored. Do not commit API keys.
- Use `.env.example` for `XFYUN_APP_ID`, `XFYUN_API_KEY`, `XFYUN_API_SECRET`.
