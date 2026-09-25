# asr

Minimal terminal UI for live microphone transcription — Cantonese–English workflows.

Claude Code / OpenCode–inspired TUI: slim chrome, keyboard-first. Arrow-key model picker, then a calm live caption + transcript log. Press **m** anytime to switch models.

## Engines

1. **iFlytek live ASR** — cloud streaming 语音听写（流式版）. Credentials in local `.env` (never committed).
2. **SenseVoice Small** — local FunAudioLLM under `models/sensevoice-small` (needs `funasr`).
3. **Parakeet Unified EN** — local Parakeet TDT via **parakeet-mlx** (`mlx-community/parakeet-tdt-0.6b-v3`); weights live in `models/parakeet-mlx` (Apple Silicon).
4. **Whisper Large V3 Turbo** — local under `models/whisper-large-v3-turbo`.

Model weights are **not** in git. `scripts/download_models.py` fetches the three local engines into `models/` after pip install. Whisper and SenseVoice prefer the GitHub [`models-v1`](https://github.com/keyframesfound/asr/releases/tag/models-v1) release tarballs (`whisper-large-v3-turbo.tar`, `sensevoice-small.tar`), which unpack to `models/<dirname>/`. Hugging Face LFS (`cdn-lfs.huggingface.co`) can stall after a few MB on some networks; if that release fetch fails, the script falls back to the Hugging Face snapshot. Parakeet stays on Hugging Face (the weights are over GitHub’s 2 GiB asset limit). iFlytek stays cloud-only. If Parakeet’s cache is still empty, `./run` can download it on first use; it does not re-download once the weights are present.

## Setup

Local Whisper, SenseVoice, and Parakeet weights are about **5 GB** and need a network connection. They are not committed. iFlytek has no download.

```bash
cd ~/iflytek-live-asr   # or: git clone https://github.com/keyframesfound/asr
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py
cp .env.example .env    # fill iFlytek keys for cloud ASR
```

`scripts/setup.sh` runs that sequence (venv, pip install, download). Re-running the download skips engines whose weights are already on disk. `./run` does not start a multi-GB download when those files are present.

| Model | Folder | Hugging Face repo | Approx. size |
|-------|--------|-------------------|--------------|
| Whisper Large V3 Turbo | `models/whisper-large-v3-turbo` | `openai/whisper-large-v3-turbo` | 1.6 GB |
| SenseVoice Small | `models/sensevoice-small` | `FunAudioLLM/SenseVoiceSmall` | 0.9 GB |
| Parakeet TDT 0.6B (`parakeet-mlx`) | `models/parakeet-mlx` | `mlx-community/parakeet-tdt-0.6b-v3` | 2.5 GB |

## Run (interactive TUI)

```bash
asr          # global command → ./run → TUI
./run
# or
python main.py
```

### Controls

Bottom action bar buttons are clickable; the same actions work from the keyboard.

| Key | Action |
|-----|--------|
| ↑ / ↓ | Move in the model list |
| Enter | Start listening with the selected model |
| Space | Start / stop live (transcript and last caption are kept when you stop) |
| e | Export transcript as `.txt` |
| a | Export the session recording as MP3 (WAV fallback if ffmpeg lacks MP3) |
| s | Settings — edit config.json values in the app |
| d | Mic picker — choose the input device (the **Mics** button does the same) |
| x | Clear all (transcript + counts + caption) |
| c | Copy the last transcript line to the clipboard |
| m / Esc | Reopen the model picker (switch mid-session; Esc on the listening view) |
| Esc | Back — closes the picker / settings / mic screen, returns to listening |
| Ctrl+C | Stop the current listening session |
| q | Quit (on any screen) |

Bottom action bar buttons are all clickable: **● Live / ■ Stop**, **Export**, **MP3**, **Clear all**, **Mics**, **Models**, **Settings**. In the model picker, unselected engines are dimmed grey and the highlighted row is bright with an accent bar.

While a model loads, a loading screen takes over and the mic is only opened once the model reports ready. When you stop a session the loading screen returns while the full audio is re-decoded, then the transcript log expands to replace the caption — the finished record becomes the main view.

Caption shows **Listening…** / **Transcribing…**, then drafts appear word by word (CJK character by character) with a soft fade-in; finals land in the transcript log as plain lines. The status line shows the state plus the mic in use (`listening · AirPods Pro`), and the meter row's ● dot + zone-colored bar pulse with the input level so you can see it is listening (○ = no signal), with dB, CLIP, and transcript counts.

### Full-session polish (stop = re-decode)

While live, the app records every raw mic block. When you stop (**space**), a loading screen shows the polish pass — it batch-decodes the whole recording in one pass (full-utterance context, pause-aware boundaries, no early-frozen partials, no discarded windows) and replaces the transcript with the polished text. The transcript log then expands to replace the caption, so the finished record is the main view. This is the accuracy model of push-to-talk dictation apps, layered on top of the live captions. The same recording can be saved as MP3 with the **MP3** button (or **a**). Disable polish with `post_stop_polish: false` (config.json) — on speakers (no headphones) room echo lands in the raw recording and can duplicate lines in the polished output.

### Settings & mic picker

Press **s** to edit the knobs from `config.json` in-app (written back with the comment keys preserved; values apply to the next session): default engine, Parakeet feed, post-final cooldown, min final chars, speech gate, mic blocksize, echo cooldown, noise floor, AGC target.

Press **d** to pick an input device. The choice is saved as `input_device` in `config.json` and wins over the blocklist/prefer heuristics; choose **System default (auto)** to go back.

### Transcription quality knobs

If finals feel like they are cutting words after a sentence, lower `post_final_cooldown_sec` (audio is discarded that long after each final). If ambient noise produces phantom text, raise `vad_min_rms`. Both are editable live in the settings screen (**s**) and apply next session.

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
| `post_stop_polish` | `true` | On stop, re-decode the whole recorded session in one batch pass and replace the transcript. |
| `mic_blocksize` | `4096` | sounddevice frames @ 16 kHz (fewer Python wakeups). |
| `vad_min_rms` | `0.02` | Speech energy gate passed by the live engines. |
| `min_final_chars` | `8` | Prefer not emitting tiny Parakeet finals unless punctuated. |
| `sample_rate` | `16000` | Fixed. 16 kHz mono float32; other values are ignored. |
| `cooldown_ms` | `1200` | Shared helper window after a final (800–1500; `0` disables). Engines read `post_final_cooldown_sec`. |
| `min_rms` | `0.003` | `prepare_chunk` floor when a caller does not pass its own gate. |
| `target_rms` | `0.025` | Quiet speech is gained toward this, capped at 4×. |
| `device_blocklist` | Zoom, Teams, Steam, EShare, BlackHole, Loopback, Soundflower, Aggregate, Multi-output | Case-insensitive. Never chosen when another input exists. |
| `device_prefer` | AirPods, Headset, Built-in, MacBook, USB | Ordered. First non-blocklisted match wins. |
| `input_device` | *(empty)* | Exact input name from the mic picker (d). Overrides blocklist/prefer; empty = auto. |


## Secrets

- `.env` is gitignored. Do not commit API keys.
- Use `.env.example` for `XFYUN_APP_ID`, `XFYUN_API_KEY`, `XFYUN_API_SECRET`.
