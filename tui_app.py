"""Interactive TUI for asr (Textual) — Claude Code / OpenCode inspired."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rich.align import Align
from rich.markup import escape
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Label, ListItem, ListView, RichLog, Static

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
        if path.suffix.lower() != ".txt":
            path = path.with_suffix(".txt")
        return path
    except Exception:
        docs = Path.home() / "Documents"
        docs.mkdir(parents=True, exist_ok=True)
        return docs / default_name


def _classify_partial(text: str) -> str:
    """Return 'listening' | 'transcribing' | 'loading' | 'draft'."""
    t = (text or "").strip()
    low = t.lower()
    if not t:
        return "listening"
    if low in _STATUS_LISTENING or low.startswith("listening on "):
        return "listening"
    if low in _STATUS_TRANSCRIBING:
        return "transcribing"
    if any(low.startswith(p) for p in _STATUS_LOADING_PREFIXES):
        return "loading"
    return "draft"


def _clock_str() -> str:
    return datetime.now().strftime("%H:%M")


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
    def __init__(self, summary: str, error: str | None = None) -> None:
        super().__init__()
        self.summary = summary
        self.error = error


@dataclass
class TranscriptLine:
    ts: datetime
    text: str


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

    def on_mount(self) -> None:
        self._render_bar()
        self.set_interval(30.0, self._tick_clock)

    def _tick_clock(self) -> None:
        self._render_bar()

    def set_context(self, context: str) -> None:
        self._bar_context = context
        self._render_bar()

    def set_live(self, live: bool) -> None:
        self._live = live
        self._render_bar()

    def _render_bar(self) -> None:
        parts: list[Text] = [Text("asr", style=f"bold {_ACCENT}")]
        if self._bar_context:
            parts.append(Text("  ·  ", style=_MUTED))
            parts.append(Text(self._bar_context, style="#e8e8e8"))
        if self._show_live:
            parts.append(Text("  ·  ", style=_MUTED))
            if self._live:
                parts.append(Text("● LIVE", style="bold #e05c5c"))
            else:
                parts.append(Text("○ idle", style=_MUTED))
        # Clock pushed right via spacer in CSS layout — append dim clock
        parts.append(Text("  ·  ", style=_MUTED))
        parts.append(Text(_clock_str(), style=_MUTED))
        self.update(Text.assemble(*parts))


class KeyHint(Static):
    """Sparse bottom keyhint row."""

    def __init__(self, hints: list[tuple[str, str]], **kwargs) -> None:
        # hints: [(key, label), ...]
        pieces: list[Text] = []
        for i, (key, label) in enumerate(hints):
            if i:
                pieces.append(Text("  ·  ", style=_MUTED))
            pieces.append(Text(key, style=f"bold {_ACCENT}"))
            pieces.append(Text(f" {label}", style=_SECONDARY))
        super().__init__(Text.assemble(*pieces), id=kwargs.pop("id", "keyhint"), **kwargs)


class ModelPickerScreen(Screen):
    """Arrow-key model picker. Defaults to config default_engine (parakeet)."""

    BINDINGS = [
        Binding("escape", "app.quit", "Quit", show=False),
        Binding("q", "app.quit", "Quit", show=False),
    ]

    def compose(self) -> ComposeResult:
        yield TopBar("models")
        with Vertical(id="picker-wrap"):
            yield Static("Select model", id="picker-title")
            items = []
            for opt in ENGINES:
                item = ListItem(
                    Vertical(
                        Static(opt.title, classes="model-title"),
                        Static(opt.blurb, classes="model-blurb"),
                        classes="model-copy",
                    ),
                    id=f"eng-{opt.code}",
                    classes="model-item",
                )
                items.append(item)
            yield ListView(*items, id="model-list")
        yield KeyHint(
            [("↑↓", "navigate"), ("enter", "select"), ("q", "quit")],
            id="keyhint",
        )

    def on_mount(self) -> None:
        lv = self.query_one("#model-list", ListView)
        lv.focus()
        code = default_engine_code()
        for i, opt in enumerate(ENGINES):
            if opt.code == code:
                lv.index = i
                break

    @on(ListView.Selected)
    def on_selected(self, event: ListView.Selected) -> None:
        item_id = event.item.id or ""
        code = item_id.removeprefix("eng-")
        if code:
            self.post_message(ModelPicked(code))


class ListeningScreen(Screen):
    """Centered live caption + scrolling log + export + start/stop live ASR.

    Caption and transcript use the terminal typeface. Textual CSS cannot
    set PingFang HK; see the stylesheet comment and README "Hong Kong CJK font".
    """

    BINDINGS = [
        Binding("m", "pick_model", "Models", show=False),
        Binding("escape", "pick_model", "Models", show=False),
        Binding("space", "toggle_live", "Start/Stop", show=False),
        Binding("e", "export_txt", "Export", show=False),
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
        self._lines: list[TranscriptLine] = []
        self._caption = ""
        # listening | transcribing | live | stopped | loading
        self._ui_state = "stopped"

    def compose(self) -> ComposeResult:
        yield TopBar(self._opt.title, show_live=True)
        with Vertical(id="listen-wrap"):
            yield Static("", id="status", classes="status-line")
            with Vertical(id="caption-stage"):
                yield Static("", id="caption", classes="caption-empty")
            yield Static("", id="stage-rule", classes="rule")
            with VerticalScroll(id="log-scroll"):
                yield RichLog(
                    id="transcript",
                    highlight=False,
                    markup=True,
                    wrap=True,
                    max_lines=2000,
                )
            # Hidden partial sink — engines still write here; keep for status hooks
            yield Static("", id="partial", classes="partial-hidden")
        yield KeyHint(
            [("space", "live"), ("e", "export"), ("m", "models"), ("q", "quit")],
            id="keyhint",
        )

    def on_mount(self) -> None:
        # Auto-start so selecting a model feels immediate; toggle can stop/restart.
        self._start_engine()

    def _top(self) -> TopBar:
        return self.query_one("#top-bar", TopBar)

    def _set_live_ui(self, running: bool) -> None:
        self._session_running = running
        self._top().set_live(running)

    def _set_status_line(self, state: str, detail: str = "") -> None:
        self._ui_state = state
        status = self.query_one("#status", Static)
        if state == "listening":
            status.update(Text("listening", style=f"italic {_MUTED}"))
        elif state == "transcribing":
            status.update(Text("transcribing…", style=f"italic {_SECONDARY}"))
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
        cap = self.query_one("#caption", Static)
        if kind == "listening":
            self._caption = ""
            cap.set_class(True, "caption-empty")
            cap.set_class(True, "caption-status")
            cap.update(
                Align.center(Text("Listening…", style=f"italic {_SECONDARY}"), vertical="middle")
            )
        elif kind == "transcribing":
            self._caption = ""
            cap.set_class(True, "caption-empty")
            cap.set_class(True, "caption-status")
            cap.update(
                Align.center(Text("Transcribing…", style=f"italic {_SECONDARY}"), vertical="middle")
            )
        elif kind == "clear":
            self._caption = ""
            cap.set_class(True, "caption-empty")
            cap.set_class(True, "caption-status")
            cap.update(Align.center(Text("", style=f"italic {_MUTED}")))
        else:
            # draft or final speech text
            self._caption = text
            cap.set_class(False, "caption-empty")
            cap.set_class(False, "caption-status")
            style = "#d4d4d4" if kind == "draft" else "bold #e8e8e8"
            cap.update(Align.center(Text(text, style=style), vertical="middle"))

    def _start_engine(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._set_live_ui(True)
        self._set_status_line("loading", self._opt.title)
        self._set_caption_indicator("listening")
        self.query_one("#partial", Static).update("")
        app = self.app
        engine_code = self.engine_code

        def on_partial(text: str) -> None:
            if not self._stop.is_set():
                app.call_from_thread(self.post_message, PartialText(text))

        def on_final(text: str) -> None:
            if not self._stop.is_set():
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
                    SessionDone(summary.short_text(), summary.error),
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
        # Drop warm model when leaving so the next pick loads cleanly.
        try:
            from engines.warm import drop_all_models

            drop_all_models(keep=None)
        except Exception:
            pass
        self.app.pop_screen()
        self.app.push_screen(ModelPickerScreen())

    def action_stop_session(self) -> None:
        self._request_stop()
        self._set_live_ui(False)
        self._set_status_line("stopped", "stopping…")
        self._set_caption_indicator("clear")

    def action_toggle_live(self) -> None:
        self._toggle_live()

    def action_export_txt(self) -> None:
        self._export_txt()

    def _toggle_live(self) -> None:
        if self._session_running:
            self._request_stop()
            self._set_live_ui(False)
            self._set_status_line("stopped")
            self._set_caption_indicator("clear")
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
            body = "\n".join(
                f"[{line.ts.strftime('%Y-%m-%d %H:%M:%S')}] {line.text}"
                for line in self._lines
            )
            path.write_text(body + "\n", encoding="utf-8")
            self.query_one("#status", Static).update(
                Text(f"exported → {path}", style=_SECONDARY)
            )
            self.query_one("#transcript", RichLog).write(
                f"[dim]Exported {len(self._lines)} lines to {escape(str(path))}[/]"
            )
        except Exception as exc:  # noqa: BLE001
            self.query_one("#status", Static).update(
                Text(f"export failed — {exc}", style="#e05c5c")
            )

    @on(PartialText)
    def show_partial(self, event: PartialText) -> None:
        kind = _classify_partial(event.text)
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
        elif kind == "loading":
            self._set_status_line("loading", event.text)
            self.query_one("#partial", Static).update(f"~ {event.text}")
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
        self._lines.append(TranscriptLine(ts=datetime.now(), text=text))
        # Settle caption to final text; log gets the timestamped line.
        self._set_caption_indicator("final", text)
        self._set_status_line("listening")  # ready for next utterance after final
        self.query_one("#transcript", RichLog).write(
            f"[dim]{datetime.now().strftime('%H:%M:%S')}[/]  {escape(text)}"
        )

    @on(SessionDone)
    def session_done(self, event: SessionDone) -> None:
        self._set_live_ui(False)
        self._set_caption_indicator("clear")
        log = self.query_one("#transcript", RichLog)
        if event.error:
            self._set_status_line("stopped", f"error — {event.error}")
            log.write(f"[red]{escape(event.error)}[/]")
        else:
            self._set_status_line("stopped")
        if event.summary:
            log.write(f"[dim]{escape(event.summary)}[/]")


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
        padding: 1 3;
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
        color: #e8e8e8;
        text-style: none;
    }
    .model-blurb {
        color: #6b6b6b;
        text-style: none;
        margin-top: 0;
    }
    ListItem.--highlight {
        background: #1a1a1a;
        border-left: solid #e8a87c;
    }
    ListItem.--highlight .model-title {
        color: #e8e8e8;
        text-style: bold;
    }
    ListItem.--highlight .model-blurb {
        color: #8a8a8a;
    }

    /* —— status (dim, one line) —— */
    #status, .status-line {
        height: 1;
        color: #6b6b6b;
        margin-bottom: 0;
        padding: 0 1;
    }

    /* —— caption stage: generous, calm, no chunky frame —— */
    #caption-stage {
        height: 12;
        margin: 1 0;
        padding: 2 4;
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

    /* subtle separator between caption and log */
    #stage-rule, .rule {
        height: 1;
        color: #2a2a2a;
        background: #0a0a0a;
        border-top: solid #2a2a2a;
        margin: 0 1 1 1;
    }

    /* —— transcript log —— */
    #log-scroll {
        height: 1fr;
        border: none;
        background: #0a0a0a;
        padding: 0 1;
    }
    /* Caption (#partial) and transcript log (#transcript).
       Textual CSS has no font-family or font-size. Declaring
       font-family is an error ("Invalid CSS property 'font-family'")
       and the TUI will not start. These widgets inherit Terminal.app's
       font. Preferred stack for 繁體中文（香港）, in order:
       PingFang HK（蘋方-港）, Noto Sans HK, Noto Sans TC.
       Set that face in Terminal.app — README, "Hong Kong CJK font". */
    #transcript {
        height: auto;
        min-height: 100%;
        background: #0a0a0a;
        color: #e8e8e8;
        text-wrap: wrap;
        scrollbar-background: #0a0a0a;
        scrollbar-color: #2a2a2a;
        scrollbar-color-hover: #3a3a3a;
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
        self.push_screen(ListeningScreen(event.code))


def run_tui() -> None:
    AudioLiveApp().run()


if __name__ == "__main__":
    run_tui()
