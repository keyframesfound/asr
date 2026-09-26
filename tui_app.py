"""Interactive TUI for asr (Textual) — Claude Code / OpenCode inspired."""
from __future__ import annotations

import math
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from rich.align import Align
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen, Screen
from textual.timer import Timer
from textual.widgets import Button, Input, ListItem, ListView, Static

from engines import level as level_bus
from engines import weights as weights_bus
from engines.audio_util import is_cjk as _is_cjk
from engines.config import load_audio_config, load_config, save_config

# Patch tqdm before any engine load — Textual FDs break multiprocessing locks.
try:
    from engines.whisper import _patch_tqdm_no_mp

    _patch_tqdm_no_mp()
except Exception:
    pass

ROOT = Path(__file__).resolve().parent

# Status / caption markers emitted by engines (not speech text).
_STATUS_LISTENING = ("listening…", "listening...", "listening")
_STATUS_TRANSCRIBING = ("transcribing…", "transcribing...", "transcribing")
_STATUS_LOADING_PREFIXES = ("loading ", "ready (")
# Accent: calm amber (used sparingly — LIVE chip, focus row, key glyphs)
_ACCENT = "#e8a87c"
_MUTED = "#6b6b6b"
_SECONDARY = "#8a8a8a"


@dataclass(frozen=True)
class EngineOption:
    code: str
    title: str
    blurb: str


# Parakeet first — preferred English live default. SenseVoice remains Cantonese path.
ENGINES: list[EngineOption] = [
    EngineOption(
        "parakeet",
        "Parakeet Unified EN",
        "Default English live — Parakeet TDT (MLX GPU), low-latency drafts",
    ),
    EngineOption(
        "whisper",
        "Whisper Large V3 Turbo",
        "Local openai/whisper-large-v3-turbo — models/whisper-large-v3-turbo",
    ),
    EngineOption(
        "sensevoice",
        "SenseVoice Small",
        "Cantonese / multilingual — FunAudioLLM (models/sensevoice-small)",
    ),
    EngineOption("iflytek", "iFlytek live ASR", "Cloud streaming 语音听写 — needs .env keys"),
]


def default_engine_code() -> str:
    try:
        from engines.config import load_config

        code = str(load_config().get("default_engine") or "parakeet").strip().lower()
        if any(o.code == code for o in ENGINES):
            return code
    except Exception:
        pass
    return "parakeet"


def _build_engine(code: str):
    if code == "iflytek":
        from engines.iflytek import IflytekEngine

        return IflytekEngine()
    if code == "sensevoice":
        from engines.sensevoice import SenseVoiceEngine

        return SenseVoiceEngine()
    if code == "parakeet":
        from engines.parakeet import ParakeetEngine

        return ParakeetEngine()
    if code == "whisper":
        from engines.whisper import WhisperTurboEngine

        return WhisperTurboEngine()
    raise ValueError(f"Unknown engine: {code}")


def _pick_save_path(default_name: str) -> Path | None:
    """Native macOS save panel; Cancel returns None. Fallback: ~/Documents."""
    try:
        import subprocess

        script = (
            "try\n"
            f"  set theFile to choose file name with prompt \"Save transcript as\" default name \"{default_name}\"\n"
            "  return POSIX path of theFile\n"
            "on error\n"
            "  return \"\"\n"
            "end try\n"
        )
        out = subprocess.check_output(["osascript", "-e", script], text=True).strip()
        if not out:
            return None
        path = Path(out).expanduser()
        default_suffix = Path(default_name).suffix.lower() or ".txt"
        if path.suffix.lower() != default_suffix:
            path = path.with_suffix(default_suffix)
        return path
    except Exception:
        docs = Path.home() / "Documents"
        docs.mkdir(parents=True, exist_ok=True)
        return docs / default_name


def _classify_partial(text: str) -> str:
    """Return 'listening' | 'transcribing' | 'polishing' | 'loading' | 'draft'."""
    t = (text or "").strip()
    low = t.lower()
    if not t:
        return "listening"
    if low in _STATUS_LISTENING or low.startswith("listening on "):
        return "listening"
    if low in _STATUS_TRANSCRIBING:
        return "transcribing"
    if low.startswith("polishing"):
        return "polishing"
    if any(low.startswith(p) for p in _STATUS_LOADING_PREFIXES):
        return "loading"
    return "draft"


def _clock_str() -> str:
    return datetime.now().strftime("%H:%M")


# —— caption reveal: pure helpers (unit-tested) ——

def _tokenize(text: str) -> list[str]:
    """Split into reveal chunks: latin words (trailing space attached), one CJK char each.

    ``"".join(_tokenize(t)) == t`` always holds.
    """
    tokens: list[str] = []
    buf: list[str] = []
    for ch in text:
        if _is_cjk(ch):
            if buf:
                tokens.append("".join(buf))
                buf.clear()
            tokens.append(ch)
        elif ch.isspace():
            buf.append(ch)
            tokens.append("".join(buf))
            buf.clear()
        else:
            buf.append(ch)
    if buf:
        tokens.append("".join(buf))
    return tokens


def _plan_reveal(shown: str, target: str) -> tuple[str, list[str]]:
    """Diff a caption update into (stable base, chunks still to animate).

    Pure append keeps every shown char in place (no flashing); a rewrite only
    moves text after the longest token-aligned common prefix.
    """
    if target.startswith(shown):
        return shown, _tokenize(target[len(shown) :])
    k = 0
    for a, b in zip(shown, target):
        if a != b:
            break
        k += 1
    toks = _tokenize(target)
    acc = 0
    idx = 0
    while idx < len(toks) and acc + len(toks[idx]) <= k:
        acc += len(toks[idx])
        idx += 1
    return "".join(toks[:idx]), toks[idx:]


def _export_body(lines: list[str]) -> str:
    """Transcript file body: plain text lines, one final per line."""
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def _wav_bytes(audio: object, sample_rate: int) -> bytes:
    """16 kHz mono float32 recording → 16-bit PCM WAV file bytes (stdlib only)."""
    import io
    import wave

    pcm = (np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0) * 32767.0)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate))
        wav.writeframes(pcm.astype("<i2").tobytes())
    return buf.getvalue()


def _encode_mp3(wav_path: Path, out_path: Path) -> bool:
    """ffmpeg WAV→MP3; False when ffmpeg or libmp3lame is unavailable."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    try:
        proc = subprocess.run(
            [
                ffmpeg, "-y", "-loglevel", "error",
                "-i", str(wav_path),
                "-codec:a", "libmp3lame", "-qscale:a", "4",
                str(out_path),
            ],
            capture_output=True,
            timeout=120,
        )
    except Exception:
        return False
    return proc.returncode == 0 and out_path.exists()


def _copy_to_clipboard(text: str) -> bool:
    """Best-effort system clipboard (macOS/Linux/Windows); False if unsupported."""
    for cmd in ("pbcopy", "wl-copy", "xclip", "clip.exe"):
        if shutil.which(cmd) is None:
            continue
        argv = [cmd] if cmd != "xclip" else ["xclip", "-selection", "clipboard"]
        try:
            subprocess.run(
                argv, input=text.encode("utf-8"), check=True, timeout=5
            )
            return True
        except Exception:
            return False
    return False


# —— editable settings (settings screen) ——

@dataclass(frozen=True)
class SettingSpec:
    key: str
    label: str
    kind: str  # "float" | "int" | "choice"
    lo: float = 0.0
    hi: float = 0.0
    choices: tuple[str, ...] = ()
    fmt: str = "{:.2f}"


SETTINGS: tuple[SettingSpec, ...] = (
    SettingSpec(
        "default_engine",
        "Default engine",
        "choice",
        choices=("parakeet", "whisper", "sensevoice", "iflytek"),
        fmt="{}",
    ),
    SettingSpec("parakeet_feed_sec", "Parakeet feed (s per chunk)", "float", 0.2, 1.0),
    SettingSpec("post_final_cooldown_sec", "Post-final cooldown (s)", "float", 0.0, 5.0),
    SettingSpec("min_final_chars", "Min chars per final", "int", 1, 64, fmt="{:d}"),
    SettingSpec("vad_min_rms", "Speech gate (RMS)", "float", 0.0, 0.5, fmt="{:.3f}"),
    SettingSpec("mic_blocksize", "Mic blocksize (frames)", "int", 512, 16384, fmt="{:d}"),
    SettingSpec("cooldown_ms", "Echo cooldown (ms)", "int", 0, 5000, fmt="{:d}"),
    SettingSpec("min_rms", "Noise floor min RMS", "float", 0.0001, 1.0, fmt="{:.4f}"),
    SettingSpec("target_rms", "AGC target RMS", "float", 0.0001, 1.0, fmt="{:.4f}"),
)


def _settings_values() -> dict[str, object]:
    """Current, validated values for every SettingSpec (engine + audio views merged)."""
    values: dict[str, object] = dict(load_config())
    audio = load_audio_config()
    values["cooldown_ms"] = audio.cooldown_ms
    values["min_rms"] = audio.min_rms
    values["target_rms"] = audio.target_rms
    return values


def _parse_setting(spec: SettingSpec, raw: str) -> tuple[object | None, str | None]:
    """Validate/clamp raw editor input. Returns (value, error); error None when valid."""
    text = raw.strip()
    if spec.kind == "choice":
        if text in spec.choices:
            return text, None
        return None, f"choose one of: {', '.join(spec.choices)}"
    try:
        value: float | int = float(text) if spec.kind == "float" else int(text)
    except ValueError:
        return None, f"{raw!r} is not a number"
    value = min(spec.hi, max(spec.lo, value))
    return value, None


class ModelPicked(Message):
    def __init__(self, code: str) -> None:
        super().__init__()
        self.code = code


class PartialText(Message):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class FinalText(Message):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class SessionDone(Message):
    def __init__(
        self,
        summary: str,
        error: str | None = None,
        *,
        polished: str | None = None,
        recorded_sec: float = 0.0,
        audio: object | None = None,
        sample_rate: int = 16000,
    ) -> None:
        super().__init__()
        self.summary = summary
        self.error = error
        self.polished = polished
        self.recorded_sec = recorded_sec
        self.audio = audio
        self.sample_rate = sample_rate


class TopBar(Static):
    """Slim custom chrome: asr · context · clock.

    The label lives on ``_bar_context``. ``MessagePump._context`` is the
    context manager wrapped around the message loop (``with self._context():``).
    Storing a string under that name makes the pump raise
    ``TypeError: 'str' object is not callable`` and the screen never paints.
    """

    def __init__(self, context: str = "", *, show_live: bool = False) -> None:
        super().__init__(id="top-bar")
        self._bar_context = context
        self._show_live = show_live
        self._live = False
        self._live_since: float | None = None

    def on_mount(self) -> None:
        self._render_bar()
        # 1 s tick so the LIVE elapsed timer stays current.
        self.set_interval(1.0, self._tick_clock)

    def _tick_clock(self) -> None:
        self._render_bar()

    def set_context(self, context: str) -> None:
        self._bar_context = context
        self._render_bar()

    def set_live(self, live: bool) -> None:
        self._live = live
        self._live_since = time.monotonic() if live else None
        self._render_bar()

    def _live_chip(self) -> Text:
        if not self._live:
            return Text("○ idle", style=_MUTED)
        chip = Text("● LIVE", style="bold #e05c5c")
        if self._live_since is not None:
            secs = max(0, int(time.monotonic() - self._live_since))
            chip.append(f" {secs // 60:02d}:{secs % 60:02d}", style="#e05c5c")
        return chip

    def _render_bar(self) -> None:
        parts: list[Text] = [Text("asr", style=f"bold {_ACCENT}")]
        if self._bar_context:
            parts.append(Text("  ·  ", style=_MUTED))
            parts.append(Text(self._bar_context, style="#e8e8e8"))
        if self._show_live:
            parts.append(Text("  ·  ", style=_MUTED))
            parts.append(self._live_chip())
        # Clock pushed right via spacer in CSS layout — append dim clock
        parts.append(Text("  ·  ", style=_MUTED))
        parts.append(Text(_clock_str(), style=_MUTED))
        self.update(Text.assemble(*parts))


class KeyHint(Static):
    """Sparse bottom keyhint row — keys in bright chips so they stay legible."""

    def __init__(self, hints: list[tuple[str, str]], **kwargs) -> None:
        pieces: list[Text] = []
        for i, (key, label) in enumerate(hints):
            if i:
                pieces.append(Text("    ", style=_MUTED))
            pieces.append(Text(f" {key} ", style=f"bold {_ACCENT}", end=""))
            pieces.append(Text(" ", end=""))
            pieces.append(Text(label, style="bold #a8a8a8"))
        super().__init__(Text.assemble(*pieces), id=kwargs.pop("id", "keyhint"), **kwargs)


def _rms_to_level(rms: float) -> float:
    """Map RMS to 0–1 with a -60 dB floor so speech sits mid-bar."""
    if rms <= 0.0:
        return 0.0
    db = 20.0 * math.log10(min(rms, 1.0))
    return min(1.0, max(0.0, (db + 60.0) / 60.0)) ** 0.9


class LevelMeter(Static):
    """Slim live mic row: state dot · level bar · dB · transcript counts.

    Reads engines.level (fed by the sounddevice callback) so it works for
    every engine without touching their loops. A solid ● means audio is
    arriving from the mic; ○ means idle/no signal.
    """

    BAR_SLOTS = 30
    STALE_SEC = 0.4

    def __init__(self, **kwargs) -> None:
        super().__init__("", id=kwargs.pop("id", "meter"), **kwargs)
        self._smooth = 0.0
        self._counts = (0, 0)

    def on_mount(self) -> None:
        self.set_interval(1 / 15, self._tick)

    def set_counts(self, lines: int, words: int) -> None:
        self._counts = (lines, words)

    def _tick(self) -> None:
        rms, peak, ts, clipped = level_bus.snapshot()
        fresh = ts > 0.0 and (time.monotonic() - ts) < self.STALE_SEC
        target = _rms_to_level(rms) if fresh else 0.0
        # Fast attack, slow release keeps the bar calm.
        self._smooth = max(target, self._smooth * 0.85)
        peak_level = _rms_to_level(peak) if fresh else 0.0
        db = (
            max(-60, min(0, round(20.0 * math.log10(max(rms, 1e-5)))))
            if fresh
            else None
        )
        self.update(self._render_row(fresh, peak_level, db, clipped))

    def _render_row(self, fresh: bool, peak_level: float, db: int | None, clipped: bool) -> Text:
        row = Text()
        if fresh:
            row.append("●", style="bold #e05c5c" if clipped else f"bold {_ACCENT}")
        else:
            row.append("○", style=_MUTED)
        row.append(" ", style=_MUTED)

        filled = round(self._smooth * self.BAR_SLOTS)
        peak_idx = round(peak_level * self.BAR_SLOTS)
        for i in range(self.BAR_SLOTS):
            frac = (i + 1) / self.BAR_SLOTS
            if fresh and i < filled:
                if clipped or frac >= 0.92:
                    color = "#e05c5c"
                elif frac >= 0.7:
                    color = _ACCENT
                else:
                    color = "#8a8a8a"
                row.append("█", style=color)
            elif i == peak_idx and i >= filled and peak_level > 0.01:
                row.append("▍", style=_ACCENT)  # peak-hold tick
            else:
                row.append("·", style="#2e2e2e")

        if fresh and db is not None:
            row.append(f"  {db:>3} dB", style=_SECONDARY)
        else:
            row.append("  — dB", style="#3a3a3a")
        if clipped:
            row.append("  CLIP", style="bold #e05c5c")
        lines, words = self._counts
        if lines:
            row.append(f"   {lines} lines · {words} words", style="#4a4a4a")
        return row


class AnimatedCaption(Static):
    """Big centered caption; each word/CJK char fades in instead of the line flashing.

    Stable prefix never re-renders with a different style, so drafts that grow
    word-by-word read as smooth typing. New chunks run dim → mid → base color
    over ~3 ticks; long backlogs reveal faster to catch up.
    """

    TICK_SEC = 0.055

    def __init__(self, **kwargs) -> None:
        super().__init__("", **kwargs)
        self._base = ""  # text fully settled
        self._shown = ""  # base + fading + pending (everything queued)
        self._pending: list[str] = []
        self._fading: list[list] = []  # [chunk, stage]
        self._final = False
        self._timer: Timer | None = None

    def set_speech(self, text: str, *, final: bool = False) -> None:
        text = (text or "").rstrip()
        if not text:
            return
        append = bool(self._shown) and text.startswith(self._shown)
        if final:
            self._final = True
        elif not append:
            self._final = False
        if not append:
            # Rewrite: stale fade-ins no longer belong to the target.
            self._fading.clear()
        self._base, self._pending = _plan_reveal(self._shown, text)
        self._shown = text
        self._kick()

    def set_status(self, text: str, *, muted: bool = False) -> None:
        """Listening… / Transcribing… / cleared — italic, centered, no animation."""
        self._cancel()
        style = f"italic {_MUTED}" if muted else f"italic {_SECONDARY}"
        self.set_class(muted or not text, "caption-empty")
        self.set_class(True, "caption-status")
        self.update(Align.center(Text(text, style=style), vertical="middle"))

    def clear_caption(self) -> None:
        self._cancel()
        self.set_class(True, "caption-empty")
        self.set_class(True, "caption-status")
        self.update(Align.center(Text("", style=f"italic {_MUTED}"), vertical="middle"))

    def _cancel(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self._base = ""
        self._shown = ""
        self._pending.clear()
        self._fading.clear()
        self._final = False

    def _kick(self) -> None:
        if self._timer is None:
            self._timer = self.set_timer(self.TICK_SEC, self._tick)

    def _tick(self) -> None:
        self._timer = None
        if self._pending:
            step = 1 + len(self._pending) // 12  # catch up when a long draft lands
            for _ in range(min(step, len(self._pending))):
                self._fading.append([self._pending.pop(0), 0])
        merged = ""
        for chunk in self._fading:
            chunk[1] += 1
        still: list[list] = []
        for chunk, stage in self._fading:
            if stage > 1:
                merged += chunk
            else:
                still.append([chunk, stage])
        self._fading = still
        if merged:
            self._base += merged
        self._render_caption()
        if self._pending or self._fading:
            self._kick()

    def _render_caption(self) -> None:
        base_style = "bold #e8e8e8" if self._final else "#d4d4d4"
        shades = ("#6a6a6a", "#b0b0b0") if self._final else ("#565656", "#a4a4a4")
        line = Text(self._base, style=base_style)
        for chunk, stage in self._fading:
            line.append(chunk, style=shades[min(stage, len(shades) - 1)])
        self.set_class(False, "caption-empty")
        self.set_class(False, "caption-status")
        self.update(Align.center(line, vertical="middle"))


class LoadingPanel(Static):
    """Centered spinner for model load / polish — shown instead of the caption."""

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    TICK_SEC = 0.08

    def __init__(self, **kwargs) -> None:
        super().__init__("", id=kwargs.pop("id", "loading-panel"), **kwargs)
        self._frame = 0
        self._label = "Loading…"
        self._sub = ""

    def on_mount(self) -> None:
        self.display = False  # hidden until a layout helper shows it
        self.set_interval(self.TICK_SEC, self._tick)
        self._render_panel()

    def set_label(self, label: str, sub: str = "") -> None:
        self._label = (label or "").strip() or "Loading…"
        self._sub = (sub or "").strip()
        self._render_panel()

    def _tick(self) -> None:
        self._frame = (self._frame + 1) % len(self.FRAMES)
        self._render_panel()

    def _render_panel(self) -> None:
        body = Text.assemble(
            (self.FRAMES[self._frame], f"bold {_ACCENT}"),
            ("  " + self._label, "#e8e8e8"),
        )
        if self._sub:
            body.append("\n")
            body.append(Text(self._sub, style=f"italic {_MUTED}"))
        self.update(Align.center(body, vertical="middle"))


class ActionButton(Button):
    """Click-only button — never takes focus, so keys keep working."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.can_focus = False


class ModelPickerScreen(Screen):
    """Arrow-key model picker. Defaults to config default_engine (parakeet).

    Local engines show install state; enter (or i) downloads missing weights,
    u removes them again (second press confirms). Enter on a not-installed
    model starts the download and stays here instead of starting a doomed
    session.
    """

    BINDINGS = [
        Binding("i", "download_model", "Download", show=False),
        Binding("u", "remove_model", "Remove", show=False),
        Binding("escape", "back", "Back", show=False),
        Binding("q", "app.quit", "Quit", show=False),
    ]

    # A second u press within this window confirms the removal.
    REMOVE_CONFIRM_SEC = 6.0
    NOTE_TTL_SEC = 6.0

    def __init__(self, can_go_back: bool = False) -> None:
        super().__init__()
        # True only when pushed over a listening screen: esc returns there.
        # On the startup picker esc is inert (q quits) so escape mashing
        # can never exit the app.
        self._can_go_back = can_go_back

    def compose(self) -> ComposeResult:
        yield TopBar("models")
        with Vertical(id="picker-wrap"):
            yield Static("Select model", id="picker-title")
            default = default_engine_code()
            items = []
            for opt in ENGINES:
                title = Text(opt.title)
                if opt.code == default:
                    title.append("  · default", style=f"italic {_ACCENT}")
                item = ListItem(
                    Vertical(
                        Static(title, classes="model-title"),
                        Static(opt.blurb, classes="model-blurb"),
                        Static("", classes="model-status"),
                        classes="model-copy",
                    ),
                    id=f"eng-{opt.code}",
                    classes="model-item",
                )
                items.append(item)
            yield ListView(*items, id="model-list")
            yield Static("", id="picker-note")
        hints = [("↑↓", "navigate"), ("enter", "select")]
        if self._can_go_back:
            hints.append(("esc", "back"))
        hints.extend([("i", "download"), ("u", "remove"), ("q", "quit")])
        yield KeyHint(hints, id="keyhint")

    def on_mount(self) -> None:
        lv = self.query_one("#model-list", ListView)
        lv.focus()
        code = default_engine_code()
        for i, opt in enumerate(ENGINES):
            if opt.code == code:
                lv.index = i
                break
        self._armed_remove = ""
        self._armed_at = 0.0
        self._note_text = ""
        self._note_at = 0.0
        self._downloading: set[str] = set()
        if weights_bus.is_local_engine(code) and not weights_bus.weights_present(code):
            self._flash(f"{self._title(code)} (default) is not installed — press enter to download")
        self._refresh_rows()
        self.set_interval(0.5, self._refresh_rows)

    def action_back(self) -> None:
        if self._can_go_back:
            self.app.pop_screen()

    # -- helpers -----------------------------------------------------------

    def _title(self, code: str) -> str:
        for opt in ENGINES:
            if opt.code == code:
                return opt.title
        return code

    def _selected_code(self) -> str:
        lv = self.query_one("#model-list", ListView)
        idx = lv.index or 0
        if 0 <= idx < len(lv.children):
            return (lv.children[idx].id or "").removeprefix("eng-")
        return ""

    def _flash(self, msg: str) -> None:
        self._note_text = msg
        self._note_at = time.monotonic()
        self.query_one("#picker-note", Static).update(Text(msg, style=f"italic {_ACCENT}"))

    def _progress_text(self, state) -> str:
        if state.total > 0:
            pct = min(100, int(100 * state.downloaded / state.total))
            bits = [f"downloading… {pct}%"]
        else:
            bits = [f"downloading… {weights_bus.human_size(state.downloaded)}"]
        if state.speed > 0:
            bits.append(f"{weights_bus.human_size(state.speed)}/s")
        if state.total > 0 and state.speed > 0 and state.downloaded < state.total:
            eta = (state.total - state.downloaded) / state.speed
            bits.append(f"~{weights_bus.human_eta(eta)} left")
        return " · ".join(bits)

    def _status_text(self, code: str) -> Text:
        if not weights_bus.is_local_engine(code):
            return Text("cloud · nothing to download", style=f"italic {_MUTED}")
        state = weights_bus.download_state(code)
        if state.running:
            return Text(self._progress_text(state), style=f"italic {_ACCENT}")
        if state.error:
            msg = state.error if len(state.error) <= 64 else state.error[:61] + "…"
            return Text(f"download failed — {msg} · press enter to retry", style="italic #c96f6f")
        if weights_bus.weights_present(code):
            size = weights_bus.human_size(weights_bus.weights_size(code))
            return Text(f"installed · {size}", style=f"italic {_MUTED}")
        return Text("not installed · press enter to download", style=f"italic {_ACCENT}")

    def _refresh_rows(self) -> None:
        for opt in ENGINES:
            try:
                row = self.query_one(f"#eng-{opt.code} .model-status", Static)
            except Exception:
                continue
            state = weights_bus.download_state(opt.code)
            # Announce the moment an in-flight download lands or fails — the
            # row text below updates on the same tick.
            if opt.code in self._downloading and not state.running:
                self._downloading.discard(opt.code)
                if not state.error:
                    size = weights_bus.human_size(state.downloaded)
                    self._flash(f"downloaded {self._title(opt.code)} · {size} ✓")
            elif state.running:
                self._downloading.add(opt.code)
            row.update(self._status_text(opt.code))
        if self._note_text and time.monotonic() - self._note_at > self.NOTE_TTL_SEC:
            self._note_text = ""
            self.query_one("#picker-note", Static).update("")
        if self._armed_remove and time.monotonic() - self._armed_at > self.REMOVE_CONFIRM_SEC:
            self._armed_remove = ""

    # -- keys ---------------------------------------------------------------

    @on(ListView.Selected)
    def on_selected(self, event: ListView.Selected) -> None:
        item_id = event.item.id or ""
        code = item_id.removeprefix("eng-")
        if not code:
            return
        if weights_bus.is_local_engine(code) and not weights_bus.weights_present(code):
            # Enter on a missing model downloads it (same as i) instead of
            # starting a doomed session; it lands back here when done.
            self.action_download_model()
            return
        self.post_message(ModelPicked(code))

    def action_download_model(self) -> None:
        code = self._selected_code()
        if not weights_bus.is_local_engine(code):
            self._flash(f"{self._title(code)} streams from the cloud — no weights")
            return
        if weights_bus.weights_present(code):
            self._flash(f"{self._title(code)} is already installed")
            return
        if weights_bus.download_weights_async(code):
            self._armed_remove = ""
            self._flash(f"downloading {self._title(code)}…")
        else:
            self._flash(f"{self._title(code)} is already downloading")

    def action_remove_model(self) -> None:
        code = self._selected_code()
        if not weights_bus.is_local_engine(code):
            self._flash(f"{self._title(code)} streams from the cloud — nothing to remove")
            return
        if not weights_bus.weights_present(code):
            self._flash(f"{self._title(code)} is not installed")
            return
        if self._armed_remove != code:
            self._armed_remove, self._armed_at = code, time.monotonic()
            size = weights_bus.human_size(weights_bus.weights_size(code))
            self._flash(f"press u again to remove {self._title(code)} ({size})")
            return
        self._armed_remove = ""
        freed = weights_bus.remove_weights(code)
        self._flash(f"removed {self._title(code)} — freed {freed}")


class ListeningScreen(Screen):
    """Centered live caption + export + start/stop live ASR.

    Caption and status use the terminal typeface. Textual CSS cannot
    set PingFang HK; see the stylesheet comment and README "Hong Kong CJK font".
    """

    BINDINGS = [
        Binding("m", "pick_model", "Models", show=False),
        Binding("escape", "pick_model", "Models", show=False),
        Binding("space", "toggle_live", "Start/Stop", show=False),
        Binding("e", "export_txt", "Export", show=False),
        Binding("a", "export_audio", "MP3", show=False),
        Binding("s", "settings", "Settings", show=False),
        Binding("d", "pick_device", "Mic", show=False),
        Binding("x", "clear_transcript", "Clear", show=False),
        Binding("c", "copy_last", "Copy", show=False),
        Binding("q", "app.quit", "Quit", show=False),
        Binding("ctrl+c", "stop_session", "Stop", show=False, priority=True),
    ]

    def __init__(self, engine_code: str) -> None:
        super().__init__()
        self.engine_code = engine_code
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # MessagePump._running is the message-loop flag (is_running / idle).
        # A session flag under that name stops the pump from seeing idle work
        # as soon as live ASR is toggled off.
        self._session_running = False
        self._opt = next(o for o in ENGINES if o.code == engine_code)
        self._lines: list[str] = []
        self._caption = ""
        self._words = 0
        # Raw session recording (for MP3 export), filled at SessionDone.
        self._audio: object | None = None
        self._audio_sr = 16000
        self._cleared = False  # user cleared during/after the last session
        # listening | transcribing | polishing | live | stopped | loading
        self._ui_state = "stopped"
        # Which main view fills the listen area: loading | live
        self._view = "live"

    def compose(self) -> ComposeResult:
        yield TopBar(self._opt.title, show_live=True)
        with Vertical(id="listen-wrap"):
            yield LevelMeter()
            yield Static("", id="status", classes="status-line")
            yield LoadingPanel(id="loading-panel")
            with Vertical(id="caption-stage"):
                yield AnimatedCaption(id="caption")
            # Hidden partial sink — engines still write here; keep for status hooks
            yield Static("", id="partial", classes="partial-hidden")
        with Horizontal(id="action-bar"):
            yield ActionButton("● Live", id="btn-live")
            yield ActionButton("Export", id="btn-export")
            yield ActionButton("MP3", id="btn-mp3")
            yield ActionButton("Clear all", id="btn-clear")
            yield ActionButton("Mics", id="btn-mics")
            yield ActionButton("Models", id="btn-models")
            yield ActionButton("Settings", id="btn-settings")

    def on_mount(self) -> None:
        # Auto-start so selecting a model feels immediate; toggle can stop/restart.
        self._start_engine()

    def _top(self) -> TopBar:
        return self.query_one("#top-bar", TopBar)

    # —— main view layouts: loading | live ——

    def _set_layout(self, view: str) -> None:
        """Swap which view fills the listen area; no-op when unchanged."""
        if self._view == view:
            return
        self._view = view
        try:
            panel = self.query_one("#loading-panel", LoadingPanel)
            caption = self.query_one("#caption-stage")
        except Exception:
            return  # not mounted yet
        panel.display = view == "loading"
        caption.display = view == "live"

    def _show_loading(self, label: str, sub: str = "") -> None:
        self._set_layout("loading")
        try:
            self.query_one("#loading-panel", LoadingPanel).set_label(label, sub)
        except Exception:
            pass

    def _show_live_layout(self) -> None:
        self._set_layout("live")

    def _set_live_ui(self, running: bool) -> None:
        self._session_running = running
        self._top().set_live(running)
        try:
            self.query_one("#btn-live", ActionButton).label = "■ Stop" if running else "● Live"
        except Exception:
            pass  # action bar not mounted yet

    @on(Button.Pressed, "#btn-live")
    def _btn_live(self) -> None:
        self._toggle_live()

    @on(Button.Pressed, "#btn-export")
    def _btn_export(self) -> None:
        self._export_txt()

    @on(Button.Pressed, "#btn-mp3")
    def _btn_mp3(self) -> None:
        self._export_audio()

    @on(Button.Pressed, "#btn-mics")
    def _btn_mics(self) -> None:
        self.action_pick_device()

    @on(Button.Pressed, "#btn-clear")
    def _btn_clear(self) -> None:
        self.action_clear_transcript()

    @on(Button.Pressed, "#btn-models")
    def _btn_models(self) -> None:
        self.action_pick_model()

    @on(Button.Pressed, "#btn-settings")
    def _btn_settings(self) -> None:
        self.action_settings()

    def _mic_name(self) -> str:
        """Mic actually opened by the last session ('unknown mic' before that)."""
        try:
            from engines.mic import last_input_choice

            choice = last_input_choice()
            if choice is not None and choice.name:
                return choice.name
        except Exception:
            pass
        return "unknown mic"

    def _set_status_line(self, state: str, detail: str = "") -> None:
        self._ui_state = state
        status = self.query_one("#status", Static)
        if state == "listening":
            status.update(Text(f"listening · {self._mic_name()}", style="#9a9a9a"))
        elif state == "transcribing":
            status.update(Text("transcribing…", style=f"italic {_SECONDARY}"))
        elif state == "polishing":
            status.update(Text("polishing full-session audio…", style=f"italic {_SECONDARY}"))
        elif state == "live":
            preview = detail if len(detail) <= 80 else detail[:77] + "…"
            status.update(
                Text(preview, style=_SECONDARY) if preview else Text("live", style=_MUTED)
            )
        elif state == "loading":
            status.update(
                Text(f"loading {detail or self._opt.title}…", style=f"italic {_MUTED}")
            )
        elif state == "stopped":
            msg = (detail or "").removeprefix("Status: ").strip()
            if not msg:
                msg = "stopped — space to resume"
            status.update(Text(msg, style=_MUTED))
        else:
            status.update(Text(str(state), style=_MUTED))

    def _set_caption_indicator(self, kind: str, text: str = "") -> None:
        """Update big centered caption for listening / transcribing / draft / final."""
        cap = self.query_one("#caption", AnimatedCaption)
        if kind == "listening":
            # Keep settled speech text visible; only show the indicator when empty.
            if not self._caption:
                self._caption = ""
                cap.set_status("Listening…", muted=True)
        elif kind == "transcribing":
            self._caption = ""
            cap.set_status("Transcribing…")
        elif kind == "clear":
            self._caption = ""
            cap.clear_caption()
        else:
            # draft or final speech text
            self._caption = text
            cap.set_speech(text, final=(kind == "final"))

    def _start_engine(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        level_bus.reset_level()
        self._set_live_ui(True)
        self._audio = None
        self._cleared = False
        self._set_status_line("loading", self._opt.title)
        # Loading screen first; the mic only opens once the model is ready.
        self._show_loading(
            f"Loading {self._opt.title}…", sub="mic opens once the model is ready"
        )
        self._set_caption_indicator("listening")
        self.query_one("#partial", Static).update("")
        app = self.app
        engine_code = self.engine_code

        def on_partial(text: str) -> None:
            # No stop guard: engines emit the "Polishing…" status after stop,
            # and the engine itself honors stop_event before anything else.
            app.call_from_thread(self.post_message, PartialText(text))

        def on_final(text: str) -> None:
            app.call_from_thread(self.post_message, FinalText(text))

        def worker() -> None:
            try:
                # One warm model: drop others before loading the selected engine.
                from engines.warm import drop_all_models

                drop_all_models(keep=engine_code)
                engine = _build_engine(engine_code)
                summary = engine.run(on_partial, on_final, stop_event=self._stop)
                app.call_from_thread(
                    self.post_message,
                    SessionDone(
                        summary.short_text(),
                        summary.error,
                        polished=getattr(summary, "polished", None),
                        recorded_sec=float(getattr(summary, "recorded_sec", 0.0) or 0.0),
                        audio=getattr(summary, "audio", None),
                        sample_rate=int(getattr(summary, "audio_sample_rate", 16000) or 16000),
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                app.call_from_thread(
                    self.post_message,
                    SessionDone(f"Engine crashed: {exc}", str(exc)),
                )

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()

    def _request_stop(self) -> None:
        self._stop.set()

    def action_pick_model(self) -> None:
        self._request_stop()
        self._set_live_ui(False)
        self._set_status_line("stopped", "stopped")
        self._set_caption_indicator("clear")
        # Stack the picker on top (like Settings/Mics) so esc returns to this
        # screen with its transcript. The warm model survives too: the session
        # worker drops non-matching models when a new engine starts.
        self.app.push_screen(ModelPickerScreen(can_go_back=True))

    def action_stop_session(self) -> None:
        self._request_stop()
        self._set_live_ui(False)
        self._set_status_line("stopped", "stopping…")
        # Caption keeps the last text so stopping never wipes the record.

    def action_toggle_live(self) -> None:
        self._toggle_live()

    def action_export_txt(self) -> None:
        self._export_txt()

    def action_export_audio(self) -> None:
        self._export_audio()

    def _export_audio(self) -> None:
        """Save the session recording as MP3 (ffmpeg) — WAV fallback."""
        audio = self._audio
        if audio is None or not np.asarray(audio).size:
            self._flash("no session recording — run live first", color=_MUTED)
            return
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = _pick_save_path(f"session-{self.engine_code}-{stamp}.mp3")
        if path is None:
            self._flash("export cancelled", color=_MUTED)
            return
        try:
            wav = _wav_bytes(audio, self._audio_sr)
            if path.suffix.lower() == ".wav":
                path.write_bytes(wav)
                self._flash(f"exported → {path}")
                return
            import tempfile

            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            try:
                tmp.write(wav)
                tmp.close()
                if _encode_mp3(Path(tmp.name), path):
                    self._flash(f"exported → {path}")
                    return
            finally:
                try:
                    Path(tmp.name).unlink()
                except Exception:
                    pass
            fallback = path.with_suffix(".wav")
            fallback.write_bytes(wav)
            self._flash(
                f"mp3 encoder unavailable — saved {fallback.name}", color=_SECONDARY
            )
        except Exception as exc:  # noqa: BLE001
            self._flash(f"export failed — {exc}", color="#e05c5c")

    def _flash(self, msg: str, *, color: str = _SECONDARY) -> None:
        self.query_one("#status", Static).update(Text(msg, style=color))

    def action_settings(self) -> None:
        self.app.push_screen(SettingsScreen())

    def action_pick_device(self) -> None:
        self.app.push_screen(DeviceScreen())

    def action_clear_transcript(self) -> None:
        if not self._lines:
            self._flash("nothing to clear", color=_MUTED)
            return
        cleared = len(self._lines)
        self._lines.clear()
        self._words = 0
        self._audio = None
        self._cleared = True  # a pending polish result should not resurrect text
        self.query_one("#meter", LevelMeter).set_counts(0, 0)
        self._set_caption_indicator("clear")
        self._show_live_layout()
        self._flash(f"cleared {cleared} lines", color=_MUTED)

    def action_copy_last(self) -> None:
        if not self._lines:
            self._flash("nothing to copy yet", color=_MUTED)
            return
        text = self._lines[-1]
        if _copy_to_clipboard(text):
            self._flash(f"copied — {text[:60]}{'…' if len(text) > 60 else ''}")
        else:
            self._flash("clipboard not available", color="#e05c5c")

    def _toggle_live(self) -> None:
        if self._session_running:
            self._request_stop()
            self._set_live_ui(False)
            self._set_status_line("stopped")
            # Caption keeps the last text so stopping never wipes the record.
            self.query_one("#partial", Static).update("")
        else:
            self._start_engine()

    def _export_txt(self) -> None:
        if not self._lines:
            self.query_one("#status", Static).update(
                Text("nothing to export yet", style=_MUTED)
            )
            return
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        default = f"transcript-{self.engine_code}-{stamp}.txt"
        path = _pick_save_path(default)
        if path is None:
            self.query_one("#status", Static).update(Text("export cancelled", style=_MUTED))
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_export_body(self._lines), encoding="utf-8")
            self.query_one("#status", Static).update(
                Text(f"exported → {path}", style=_SECONDARY)
            )
        except Exception as exc:  # noqa: BLE001
            self.query_one("#status", Static).update(
                Text(f"export failed — {exc}", style="#e05c5c")
            )

    @on(PartialText)
    def show_partial(self, event: PartialText) -> None:
        kind = _classify_partial(event.text)
        if kind == "loading":
            if event.text.lower().startswith("ready"):
                self._show_live_layout()  # model ready → live view
            else:
                self._show_loading(event.text)
            self._set_status_line("loading", event.text)
            self.query_one("#partial", Static).update(f"~ {event.text}")
        elif kind == "polishing":
            self._show_loading("Polishing session audio…", sub="re-decoding the full recording")
            self._set_status_line("polishing")
            self.query_one("#partial", Static).update("")
        else:
            # Anything live (listening / transcribing / draft) shows the live view.
            self._show_live_layout()
            if kind == "listening":
                self._set_status_line("listening")
                # Keep settled final on caption; only show Listening… when empty.
                if not self._caption:
                    self._set_caption_indicator("listening")
                self.query_one("#partial", Static).update("")
            elif kind == "transcribing":
                self._set_status_line("transcribing")
                self._set_caption_indicator("transcribing")
                self.query_one("#partial", Static).update("~ Transcribing…")
            else:
                # Real draft → big centered caption immediately + subtle status.
                self._set_status_line("live", event.text)
                self._set_caption_indicator("draft", event.text)
                self.query_one("#partial", Static).update(f"~ {event.text}")

    @on(FinalText)
    def show_final(self, event: FinalText) -> None:
        text = (event.text or "").strip()
        if not text:
            return
        self.query_one("#partial", Static).update("")
        self._lines.append(text)
        self._words += len(text.split())
        self.query_one("#meter", LevelMeter).set_counts(len(self._lines), self._words)
        # Settle caption to final text — the only on-screen transcript is the
        # big caption; session history lives in _lines for export/copy.
        self._set_caption_indicator("final", text)
        self._set_status_line("listening")  # ready for next utterance after final

    @on(SessionDone)
    def session_done(self, event: SessionDone) -> None:
        self._set_live_ui(False)
        self._audio = getattr(event, "audio", None)
        self._audio_sr = int(getattr(event, "sample_rate", 16000) or 16000)
        polished = (event.polished or "").strip()
        if polished and not event.error and not self._cleared:
            # Full-session re-decode won: it becomes the whole transcript —
            # even when the live loop committed zero finals. It settles into
            # the big caption; export/copy still read _lines.
            self._lines = [polished]
            self._words = len(polished.split())
            self.query_one("#meter", LevelMeter).set_counts(1, self._words)
            self._set_caption_indicator("final", polished)
            self._show_live_layout()
            secs = f"{event.recorded_sec:.0f}s" if event.recorded_sec else "session"
            self._set_status_line(
                "stopped", f"polished — full audio re-decoded ({secs})"
            )
            return
        self._show_live_layout()
        if event.error:
            short = str(event.error).split(". ")[0].rstrip(".")
            self._set_status_line("stopped", f"error — {short}")
            return
        self._set_status_line("stopped")


class EditValueScreen(ModalScreen[str | None]):
    """Inline value editor; Enter dismisses with the raw text, Esc with None."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False, priority=True)]

    def __init__(self, title: str, value: str) -> None:
        super().__init__()
        self._title = title
        self._value = value

    def compose(self) -> ComposeResult:
        with Vertical(id="edit-card"):
            yield Static(self._title, id="edit-title")
            yield Input(value=self._value, id="edit-input")
            yield Static("enter save · esc cancel", id="edit-hint")

    def on_mount(self) -> None:
        self.query_one("#edit-input", Input).focus()

    @on(Input.Submitted)
    def _submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip())

    def action_cancel(self) -> None:
        self.dismiss(None)


class SettingsScreen(Screen):
    """Arrow-key settings editor; writes config.json (comments preserved)."""

    BINDINGS = [
        Binding("escape", "back", "Back", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._specs: list[SettingSpec] = list(SETTINGS)

    def compose(self) -> ComposeResult:
        yield TopBar("settings")
        with Vertical(id="picker-wrap"):
            yield Static(
                "Settings  ·  saved to config.json  ·  values apply next session",
                id="picker-title",
            )
            items = [
                ListItem(Static("", classes="model-title"), id=f"set-{spec.key}")
                for spec in self._specs
            ]
            yield ListView(*items, id="settings-list")
        yield KeyHint(
            [("↑↓", "navigate"), ("enter", "edit"), ("esc", "back")],
            id="keyhint",
        )

    def on_mount(self) -> None:
        self._refresh_rows()
        lv = self.query_one("#settings-list", ListView)
        lv.focus()

    def _refresh_rows(self) -> None:
        values = _settings_values()
        lv = self.query_one("#settings-list", ListView)
        width = max(len(spec.label) for spec in self._specs)
        for i, spec in enumerate(self._specs):
            value = values.get(spec.key, "?")
            text = f"{spec.label:<{width}}   {spec.fmt.format(value)}"
            lv.children[i].query_one(".model-title", Static).update(
                Text(text)
            )

    def _flash(self, msg: str, *, color: str = _ACCENT) -> None:
        self.query_one("#picker-title", Static).update(msg)

    @on(ListView.Selected)
    def _selected(self, event: ListView.Selected) -> None:
        item_id = event.item.id or ""
        key = item_id.removeprefix("set-")
        spec = next((s for s in self._specs if s.key == key), None)
        if spec is None:
            return
        current = _settings_values().get(spec.key, "")
        if spec.kind == "choice":
            hint = f"{spec.label}  ({', '.join(spec.choices)})"
        else:
            hint = f"{spec.label}  ({spec.lo}–{spec.hi})"
        self.app.push_screen(
            EditValueScreen(hint, spec.fmt.format(current)),
            callback=lambda raw, s=spec: self._apply(s, raw),
        )

    def _apply(self, spec: SettingSpec, raw: str | None) -> None:
        if raw is None or raw == "":
            return
        value, error = _parse_setting(spec, raw)
        if error is not None:
            self._flash(f"? {error}", color="#e05c5c")
            return
        try:
            save_config({spec.key: value})
        except Exception as exc:  # noqa: BLE001
            self._flash(f"save failed — {exc}", color="#e05c5c")
            return
        self._refresh_rows()
        self._flash(f"saved · {spec.key} = {spec.fmt.format(value)}")

    def action_back(self) -> None:
        self.app.pop_screen()


class DeviceScreen(Screen):
    """Input device picker; the choice is saved to config.json input_device."""

    BINDINGS = [
        Binding("escape", "back", "Back", show=False),
    ]

    def compose(self) -> ComposeResult:
        yield TopBar("input device")
        with Vertical(id="picker-wrap"):
            yield Static("Select microphone  ·  applies next session", id="picker-title")
            from engines.mic import list_input_devices

            devices = list_input_devices()
            current = load_audio_config().input_device
            items: list[ListItem] = []
            auto_title = Text("System default (auto)")
            if not current:
                auto_title.append("  · current", style=f"italic {_ACCENT}")
            items.append(
                ListItem(
                    Vertical(
                        Static(auto_title, classes="model-title"),
                        Static("Let the app pick (blocklist + prefer order)", classes="model-blurb"),
                    ),
                    id="dev-auto",
                    classes="model-item",
                )
            )
            for dev in devices:
                title = Text(str(dev["name"]))
                if current and str(dev["name"]).casefold() == current.casefold():
                    title.append("  · current", style=f"italic {_ACCENT}")
                blurb = f"index {dev['index']} · {dev['max_input_channels']} ch"
                items.append(
                    ListItem(
                        Vertical(
                            Static(title, classes="model-title"),
                            Static(blurb, classes="model-blurb"),
                        ),
                        id=f"dev-{dev['index']}",
                        classes="model-item",
                    )
                )
            if not devices:
                items.append(
                    ListItem(
                        Static("No input devices found", classes="model-title"),
                        id="dev-none",
                        classes="model-item",
                    )
                )
            yield ListView(*items, id="device-list")
        yield KeyHint(
            [("↑↓", "navigate"), ("enter", "select"), ("esc", "back")],
            id="keyhint",
        )

    def on_mount(self) -> None:
        lv = self.query_one("#device-list", ListView)
        lv.focus()

    @on(ListView.Selected)
    def _selected(self, event: ListView.Selected) -> None:
        from engines.mic import list_input_devices

        item_id = event.item.id or ""
        if item_id == "dev-none":
            return
        if item_id == "dev-auto":
            save_config({"input_device": ""})
            self.app.pop_screen()
            return
        idx = item_id.removeprefix("dev-")
        try:
            dev_index = int(idx)
        except ValueError:
            return
        dev = next(
            (d for d in list_input_devices() if int(d["index"]) == dev_index), None
        )
        if dev is None:
            return
        save_config({"input_device": str(dev["name"])})
        self.app.pop_screen()

    def action_back(self) -> None:
        self.app.pop_screen()


class AudioLiveApp(App[None]):
    """Minimal full-screen transcription TUI (asr)."""

    TITLE = "asr"
    CSS = """
    Screen {
        background: #0a0a0a;
        color: #e8e8e8;
    }

    /* —— slim top bar —— */
    #top-bar {
        height: 1;
        padding: 0 2;
        color: #e8e8e8;
        background: #0a0a0a;
        text-style: none;
    }

    /* —— sparse bottom keyhint —— */
    #keyhint {
        height: 1;
        padding: 0 2;
        color: #8a8a8a;
        background: #0a0a0a;
        dock: bottom;
    }

    #picker-wrap, #listen-wrap {
        padding: 1 2;
        height: 1fr;
        background: #0a0a0a;
    }

    #picker-title {
        color: #8a8a8a;
        text-style: none;
        margin-bottom: 1;
        padding: 0 1;
    }

    /* —— model list ——
       Vertical defaults to height: 1fr. Inside a ListItem that fills the
       viewport, so each engine row becomes one screen tall and the other
       models sit below the fold behind the scrollbar. Keep the copy at
       content height so all four engines are on screen. */
    #model-list {
        height: 1fr;
        background: #0a0a0a;
        border: none;
        padding: 0;
    }
    ListView > .model-item {
        height: auto;
        padding: 0 1;
        margin: 0 0 1 0;
        background: #0a0a0a;
        border-left: solid #0a0a0a;
    }
    .model-copy {
        height: auto;
        width: 1fr;
    }
    .model-title, .model-blurb {
        height: auto;
    }
    .model-title {
        color: #8a8a8a;
        text-style: none;
    }
    .model-blurb {
        color: #565656;
        text-style: none;
        margin-top: 0;
    }
    .model-status {
        height: auto;
        margin-top: 0;
    }
    /* install/remove hints from the picker (i / u) */
    #picker-note {
        height: 1;
        padding: 0 1;
        color: #e8a87c;
    }
    /* ListItem sets `-highlight` (single dash) on the highlighted row. */
    ListItem.-highlight {
        background: #1a1a1a;
        border-left: solid #e8a87c;
    }
    ListItem.-highlight .model-title {
        color: #e8e8e8;
        text-style: bold;
    }
    ListItem.-highlight .model-blurb {
        color: #8a8a8a;
    }

    /* —— status (dim, one line) —— */
    #status, .status-line {
        height: 1;
        color: #6b6b6b;
        margin-bottom: 0;
        padding: 0 1;
    }

    /* —— loading screen (model load / polish) —— */
    #loading-panel {
        height: 1fr;
        color: #e8e8e8;
        align: center middle;
        text-align: center;
    }

    /* —— live mic level row —— */
    #meter {
        height: 1;
        padding: 0 1;
        color: #8a8a8a;
        background: #0a0a0a;
    }

    /* —— bottom action bar: clickable, prominent —— */
    #action-bar {
        height: 3;
        dock: bottom;
        padding: 0 1;
        background: #0a0a0a;
    }
    #action-bar ActionButton {
        min-width: 10;
        height: 3;
        margin: 0 1 0 0;
        padding: 0 2;
        background: #161616;
        color: #c8c8c8;
        border: round #333333;
        text-style: bold;
    }
    #action-bar ActionButton:hover {
        background: #1f1f1f;
        border: round #e8a87c;
        color: #e8e8e8;
    }
    #action-bar #btn-live {
        color: #e8a87c;
        border: round #4a3a2e;
    }
    #action-bar #btn-clear {
        color: #e05c5c;
        border: round #4a2a2a;
    }

    /* —— settings / device edit modal —— */
    #edit-card {
        width: 64;
        height: auto;
        background: #141414;
        border: round #3a3a3a;
        padding: 1 2;
    }
    #edit-title {
        color: #e8e8e8;
        margin-bottom: 1;
    }
    #edit-input {
        background: #0f0f0f;
        color: #e8e8e8;
        border: none;
    }
    #edit-hint {
        margin-top: 1;
        color: #6b6b6b;
    }

    /* —— caption stage: fills the listen area; the big live caption is the
       only on-screen transcript (history lives in _lines for export) ——
       Textual CSS has no font-family or font-size (declaring it is an error
       and the TUI will not start); the caption inherits Terminal.app's font.
       Preferred stack for 繁體中文（香港）: PingFang HK（蘋方-港）, Noto Sans HK,
       Noto Sans TC — set that face in Terminal.app, README "Hong Kong CJK font". */
    #caption-stage {
        height: 1fr;
        margin: 0 0 1 0;
        padding: 1 2;
        border: none;
        background: #0a0a0a;
        content-align: center middle;
        align: center middle;
    }
    #caption {
        width: 1fr;
        height: 1fr;
        content-align: center middle;
        text-align: center;
        color: #e8e8e8;
        padding: 1 2;
        text-wrap: wrap;
    }
    #caption.caption-empty {
        color: #6b6b6b;
        text-style: italic;
    }
    #caption.caption-status {
        color: #8a8a8a;
        text-style: italic;
    }

    /* keep partial widget for hooks; hide visually (live text is on #caption) */
    #partial, .partial-hidden {
        height: 0;
        display: none;
    }
    """

    BINDINGS = [Binding("q", "quit", "Quit", show=False)]

    def on_mount(self) -> None:
        self.push_screen(ModelPickerScreen())

    @on(ModelPicked)
    def start_listening(self, event: ModelPicked) -> None:
        if isinstance(self.screen, ModelPickerScreen):
            self.pop_screen()
        if isinstance(self.screen, ListeningScreen):
            if self.screen.engine_code == event.code:
                return  # same model — keep the screen, transcript, warm load
            self.switch_screen(ListeningScreen(event.code))
        else:
            self.push_screen(ListeningScreen(event.code))


def run_tui() -> None:
    AudioLiveApp().run()


if __name__ == "__main__":
    run_tui()
