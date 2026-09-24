"""Interactive TUI for Audio Live Transcription (Textual)."""
from __future__ import annotations

import threading
from dataclasses import dataclass

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import Screen
from rich.markup import escape
from textual.widgets import Footer, Header, Label, ListItem, ListView, RichLog, Static

# Patch tqdm before any engine load — Textual's FDs break multiprocessing locks.
try:
    from engines.whisper import _patch_tqdm_no_mp
    _patch_tqdm_no_mp()
except Exception:
    pass


@dataclass(frozen=True)
class EngineOption:
    code: str
    title: str
    blurb: str


ENGINES: list[EngineOption] = [
    EngineOption("iflytek", "iFlytek live ASR", "Cloud streaming 语音听写 — needs .env keys"),
    EngineOption("sensevoice", "SenseVoice Small", "Local FunAudioLLM — needs funasr + models/sensevoice-small"),
    EngineOption(
        "parakeet",
        "Parakeet Unified EN",
        "Local CoreML / Hex FluidAudio — models/parakeet-unified-en-0.6b-coreml",
    ),
    EngineOption(
        "whisper",
        "Whisper Large V3 Turbo",
        "Local openai/whisper-large-v3-turbo — models/whisper-large-v3-turbo",
    ),
]


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


class ModelPicked(Message):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__()


class PartialText(Message):
    def __init__(self, text: str) -> None:
        self.text = text
        super().__init__()


class FinalText(Message):
    def __init__(self, text: str) -> None:
        self.text = text
        super().__init__()


class SessionDone(Message):
    def __init__(self, summary: str, error: str | None = None) -> None:
        self.summary = summary
        self.error = error
        super().__init__()


class ModelPickerScreen(Screen):
    """OpenCode-style arrow-key model picker."""

    BINDINGS = [
        Binding("escape", "app.quit", "Quit", show=True),
        Binding("q", "app.quit", "Quit", show=False),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="picker-wrap"):
            yield Static("Select transcription model", id="picker-title")
            yield Static("↑↓ navigate · Enter confirm · Esc quit", id="picker-hint")
            items = [
                ListItem(
                    Label(f"{opt.title}\n  {opt.blurb}"),
                    id=f"eng-{opt.code}",
                )
                for opt in ENGINES
            ]
            yield ListView(*items, id="model-list")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#model-list", ListView).focus()

    @on(ListView.Selected)
    def on_selected(self, event: ListView.Selected) -> None:
        item_id = event.item.id or ""
        code = item_id.removeprefix("eng-")
        if code:
            self.post_message(ModelPicked(code))


class ListeningScreen(Screen):
    """Live transcription view."""

    BINDINGS = [
        Binding("m", "pick_model", "Models", show=True),
        Binding("escape", "pick_model", "Models", show=False),
        Binding("q", "app.quit", "Quit", show=True),
        Binding("ctrl+c", "stop_session", "Stop", show=True, priority=True),
    ]

    def __init__(self, engine_code: str) -> None:
        super().__init__()
        self.engine_code = engine_code
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._opt = next(o for o in ENGINES if o.code == engine_code)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="listen-wrap"):
            yield Static(f"Listening · {self._opt.title}", id="listen-title")
            yield Static("m models · Ctrl+C stop · q quit", id="listen-hint")
            yield Static("Status: starting…", id="status")
            with VerticalScroll(id="log-scroll"):
                yield RichLog(id="transcript", highlight=False, markup=True, wrap=True)
            yield Static("", id="partial")
        yield Footer()

    def on_mount(self) -> None:
        self._start_engine()

    def _start_engine(self) -> None:
        self._stop.clear()
        log = self.query_one("#transcript", RichLog)
        log.clear()
        self.query_one("#status", Static).update(f"Status: loading {self._opt.title}…")
        self.query_one("#partial", Static).update("")

        app = self.app

        def on_partial(text: str) -> None:
            if not self._stop.is_set():
                app.call_from_thread(self.post_message, PartialText(text))

        def on_final(text: str) -> None:
            if not self._stop.is_set():
                app.call_from_thread(self.post_message, FinalText(text))

        def worker() -> None:
            try:
                engine = _build_engine(self.engine_code)
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
        self.query_one("#status", Static).update("Status: listening…")

    def action_pick_model(self) -> None:
        self._request_stop()
        self.app.pop_screen()
        self.app.push_screen(ModelPickerScreen())

    def action_stop_session(self) -> None:
        self._request_stop()
        self.query_one("#status", Static).update("Status: stopping…")

    def _request_stop(self) -> None:
        self._stop.set()
        # Engines listen for KeyboardInterrupt in their own threads; best-effort flag.

    @on(PartialText)
    def show_partial(self, event: PartialText) -> None:
        self.query_one("#partial", Static).update(f"~ {event.text}")  # Static is plain text

    @on(FinalText)
    def show_final(self, event: FinalText) -> None:
        self.query_one("#partial", Static).update("")
        self.query_one("#transcript", RichLog).write(f"[bold]>[/] {escape(event.text)}")

    @on(SessionDone)
    def session_done(self, event: SessionDone) -> None:
        status = self.query_one("#status", Static)
        if event.error:
            status.update(f"Status: error — {event.error}")
            self.query_one("#transcript", RichLog).write(f"[red]{escape(event.error)}[/]")
        else:
            status.update("Status: stopped")
        self.query_one("#transcript", RichLog).write("")
        self.query_one("#transcript", RichLog).write(f"[dim]{escape(event.summary)}[/]")


class AudioLiveApp(App[None]):
    """Full-screen interactive transcription TUI."""

    TITLE = "Audio Live Transcription"
    CSS = """
    Screen {
        background: #0f0f0f;
        color: #e8e8e8;
    }
    #picker-wrap, #listen-wrap {
        padding: 1 2;
        height: 1fr;
    }
    #picker-title, #listen-title {
        text-style: bold;
        color: #ffffff;
        margin-bottom: 0;
    }
    #picker-hint, #listen-hint {
        color: #808080;
        margin-bottom: 1;
    }
    #model-list {
        height: 1fr;
        border: tall #333333;
        background: #141414;
    }
    ListItem {
        padding: 1 1;
    }
    ListItem > Label {
        color: #c0c0c0;
    }
    ListItem.--highlight {
        background: #2a2a2a;
    }
    ListItem.--highlight > Label {
        color: #ffffff;
        text-style: bold;
    }
    #status {
        color: #a0a0a0;
        margin-bottom: 1;
    }
    #log-scroll {
        height: 1fr;
        border: tall #333333;
        background: #141414;
    }
    #transcript {
        height: auto;
        min-height: 100%;
        background: #141414;
    }
    #partial {
        height: 3;
        color: #909090;
        margin-top: 1;
    }
    Header {
        background: #1a1a1a;
    }
    Footer {
        background: #1a1a1a;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
    ]

    def on_mount(self) -> None:
        self.push_screen(ModelPickerScreen())

    @on(ModelPicked)
    def start_listening(self, event: ModelPicked) -> None:
        # Replace picker with listening view
        if isinstance(self.screen, ModelPickerScreen):
            self.pop_screen()
        self.push_screen(ListeningScreen(event.code))


def run_tui() -> None:
    AudioLiveApp().run()


if __name__ == "__main__":
    run_tui()
