"""Headless mount checks for the Textual TUI.

TopBar used to store its label on ``_context``, which shadows
``MessagePump._context`` (a context manager). The pump then died with
``TypeError: 'str' object is not callable`` and the first paint stayed blank.
ListeningScreen._running had the same collision with the pump's running flag
(renamed to ``_session_running``).

Adapted for the current design: the listening screen's bottom chrome is the
clickable action bar (#action-bar + #bar-hint), not a keyhint row.
"""
from __future__ import annotations

import unittest
from unittest import mock

from textual.dom import DOMNode
from textual.widget import Widget

from tui_app import (
    ENGINES,
    ActionButton,
    AudioLiveApp,
    KeyHint,
    LevelMeter,
    ListeningScreen,
    ModelPickerScreen,
    ModelPicked,
    TopBar,
)


def _plain(widget: Widget) -> str:
    content = getattr(widget, "content", "")
    plain = getattr(content, "plain", None)
    if isinstance(plain, str):
        return plain
    return str(content)


def _visible_model_titles(screen) -> list[str]:
    """Titles whose row intersects the list viewport (not scrolled off-screen).

    The current-row default marker ("  · default") is stripped so the
    assertion is about the engine names themselves.
    """
    models = screen.query_one("#model-list")
    top = models.region.y
    bottom = models.region.y + models.region.height
    found: list[str] = []
    for item in models.children:
        if item.region.y < bottom and item.region.y + item.region.height > top:
            title = _plain(item.query_one(".model-title"))
            found.append(title.split("  ·")[0].strip())
    return found


def _shadowed_methods(node: DOMNode) -> list[str]:
    """Instance attributes that hide a callable defined on the class."""
    found: list[str] = []
    for child in node.walk_children(with_self=True):
        for key in child.__dict__:
            if key.startswith("__"):
                continue
            for cls in type(child).__mro__:
                attr = cls.__dict__.get(key)
                if callable(attr):
                    found.append(f"{type(child).__name__}.{key} shadows {cls.__name__}.{key}")
                    break
    return found


class TuiChromeTest(unittest.IsolatedAsyncioTestCase):
    async def test_picker_mounts_full_chrome_at_80x24(self) -> None:
        app = AudioLiveApp()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            screen = app.screen
            self.assertIsInstance(screen, ModelPickerScreen)
            self.assertEqual(_shadowed_methods(app), [])

            top = screen.query_one("#top-bar", TopBar)
            models = screen.query_one("#model-list")
            hint = screen.query_one("#keyhint", KeyHint)

            for widget, label in (
                (top, "top-bar"),
                (models, "model-list"),
                (hint, "keyhint"),
            ):
                self.assertGreater(widget.size.width, 0, label)
                self.assertGreater(widget.size.height, 0, label)

            self.assertIn("asr", _plain(top))
            self.assertIn("models", _plain(top))
            self.assertEqual(top._bar_context, "models")
            self.assertTrue(callable(top._context))
            self.assertEqual(len(models.children), len(ENGINES))
            self.assertEqual(_visible_model_titles(screen), [opt.title for opt in ENGINES])
            for item in models.children:
                self.assertLess(item.size.height, models.size.height)
                self.assertLessEqual(item.size.height, 6)
            self.assertIn("enter", _plain(hint))
            self.assertIn("quit", _plain(hint))

            top.set_context("renamed")
            await pilot.pause()
            self.assertEqual(top._bar_context, "renamed")
            self.assertIn("renamed", _plain(top))
            self.assertTrue(top.is_running)
            self.assertTrue(callable(top._context))

    async def test_listening_screen_mounts_full_chrome_at_80x24(self) -> None:
        app = AudioLiveApp()

        def _quiet_start(screen: ListeningScreen) -> None:
            screen._set_live_ui(True)
            screen._set_status_line("listening")
            screen._set_caption_indicator("listening")

        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            screen = ListeningScreen("parakeet")
            with mock.patch.object(ListeningScreen, "_start_engine", _quiet_start):
                await app.push_screen(screen)
            await pilot.pause()

            self.assertIs(app.screen, screen)
            self.assertEqual(_shadowed_methods(app), [])

            top = screen.query_one("#top-bar", TopBar)
            caption = screen.query_one("#caption")
            meter = screen.query_one("#meter", LevelMeter)
            status = screen.query_one("#status")
            bar = screen.query_one("#action-bar")
            btn_live = screen.query_one("#btn-live", ActionButton)
            btn_mp3 = screen.query_one("#btn-mp3", ActionButton)
            btn_mics = screen.query_one("#btn-mics", ActionButton)

            for widget, label in (
                (top, "top-bar"),
                (caption, "caption"),
                (meter, "meter"),
                (status, "status"),
                (bar, "action-bar"),
                (btn_live, "btn-live"),
                (btn_mp3, "btn-mp3"),
                (btn_mics, "btn-mics"),
            ):
                self.assertGreater(widget.size.width, 0, label)
                self.assertGreater(widget.size.height, 0, label)

            self.assertIn("asr", _plain(top))
            self.assertIn("Parakeet Unified EN", _plain(top))
            self.assertIn("Listening", _plain(caption))
            # The meter shows the idle dot until audio arrives, plus a dB slot.
            self.assertIn("○", _plain(meter))
            self.assertIn("dB", _plain(meter))
            # The clickable bar is the only bottom chrome (no keyhint text).
            self.assertIn("Stop", str(btn_live.label))  # session is live here
            self.assertIn("MP3", str(btn_mp3.label))
            self.assertIn("Mics", str(btn_mics.label))

            # Stopping the session must not clear the message-pump flag.
            self.assertTrue(screen.is_running)
            self.assertTrue(screen._session_running)
            screen._set_live_ui(False)
            self.assertFalse(screen._session_running)
            self.assertTrue(screen.is_running)
            self.assertTrue(callable(screen._context))
            self.assertIn("Live", str(btn_live.label))  # back to start state

    async def test_loading_and_live_layout_states(self) -> None:
        app = AudioLiveApp()

        async with app.run_test(size=(100, 28)) as pilot:
            await pilot.pause()
            screen = ListeningScreen("parakeet")
            with mock.patch.object(ListeningScreen, "_start_engine", lambda s: None):
                await app.push_screen(screen)
            await pilot.pause()

            panel = screen.query_one("#loading-panel")
            caption = screen.query_one("#caption-stage")
            # Default: live layout, caption filling the listen area.
            self.assertFalse(panel.display)
            self.assertTrue(caption.display)

            screen._show_loading(
                "Loading Parakeet (MLX GPU)…", "mic opens once the model is ready"
            )
            await pilot.pause()
            self.assertTrue(panel.display)
            self.assertFalse(caption.display)
            self.assertIn("Loading Parakeet", _plain(panel))
            self.assertIn("mic opens once the model is ready", _plain(panel))

            screen._show_live_layout()
            await pilot.pause()
            self.assertTrue(caption.display)
            self.assertFalse(panel.display)

    async def test_loading_spinner_animates(self) -> None:
        app = AudioLiveApp()

        async with app.run_test(size=(100, 28)) as pilot:
            await pilot.pause()
            screen = ListeningScreen("parakeet")
            with mock.patch.object(ListeningScreen, "_start_engine", lambda s: None):
                await app.push_screen(screen)
            await pilot.pause()
            panel = screen.query_one("#loading-panel")
            screen._show_loading("Loading Whisper Large V3 Turbo…")
            await pilot.pause()
            before = _plain(panel)
            await pilot.pause(0.25)
            self.assertNotEqual(before, _plain(panel), "spinner frame advances")

    async def test_model_rows_dim_until_highlighted(self) -> None:
        from textual.color import Color

        app = AudioLiveApp()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            screen = app.screen
            lv = screen.query_one("#model-list")
            lv.index = 1
            await pilot.pause()
            highlighted = None
            dimmed = None
            for item in lv.children:
                title = item.query_one(".model-title")
                if "-highlight" in item.classes:
                    highlighted = title
                elif dimmed is None:
                    dimmed = title
            self.assertIsNotNone(highlighted, "one row carries -highlight")
            self.assertEqual(dimmed.styles.color, Color(0x8A, 0x8A, 0x8A))
            self.assertEqual(highlighted.styles.color, Color(0xE8, 0xE8, 0xE8))

    async def _start_on_listening(self, app: AudioLiveApp, pilot, code: str = "parakeet"):
        """Reach the listening screen through the real startup flow.

        The startup picker is only popped inside start_listening, so pushing
        a listening screen manually would leave it buried on the stack.
        """
        with mock.patch.object(ListeningScreen, "_start_engine", lambda s: None):
            app.post_message(ModelPicked(code))
            await pilot.pause()
        screen = app.screen
        assert isinstance(screen, ListeningScreen)
        assert len(app.screen_stack) == 2
        return screen

    async def test_startup_picker_escape_is_inert(self) -> None:
        """Escape mashing on the startup picker must never quit the app."""
        app = AudioLiveApp()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            hint = app.screen.query_one("#keyhint", KeyHint)
            self.assertIn("quit", _plain(hint))
            self.assertNotIn("back", _plain(hint))

            await pilot.press("escape", "escape", "escape")
            await pilot.pause()
            self.assertIsInstance(app.screen, ModelPickerScreen)
            self.assertTrue(app.is_running)

    async def test_escape_from_listening_round_trips_picker(self) -> None:
        """esc: listening → picker → listening; mashing never quits or leaks."""
        app = AudioLiveApp()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            screen = await self._start_on_listening(app, pilot)

            await pilot.press("escape")
            await pilot.pause()
            picker = app.screen
            self.assertIsInstance(picker, ModelPickerScreen)
            self.assertTrue(picker._can_go_back)
            self.assertIn("back", _plain(picker.query_one("#keyhint", KeyHint)))

            await pilot.press("escape")
            await pilot.pause()
            self.assertIs(app.screen, screen)

            # Mash: picker and listening alternate, the app stays alive,
            # and no stale listening screens pile up in the stack.
            await pilot.press("escape", "escape", "escape", "escape", "escape", "escape")
            await pilot.pause()
            self.assertIs(app.screen, screen)
            self.assertEqual(len(app.screen_stack), 2)
            self.assertTrue(app.is_running)

    async def test_same_model_pick_keeps_listening_screen(self) -> None:
        app = AudioLiveApp()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            screen = await self._start_on_listening(app, pilot, "parakeet")
            await pilot.press("escape")
            await pilot.pause()

            app.post_message(ModelPicked("parakeet"))
            await pilot.pause()
            self.assertIs(app.screen, screen)
            self.assertEqual(len(app.screen_stack), 2)
            self.assertTrue(app.is_running)

    async def test_other_model_pick_replaces_listening_screen(self) -> None:
        app = AudioLiveApp()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await self._start_on_listening(app, pilot, "parakeet")
            await pilot.press("escape")
            await pilot.pause()

            with mock.patch.object(ListeningScreen, "_start_engine", lambda s: None):
                app.post_message(ModelPicked("sensevoice"))
                await pilot.pause()
            self.assertIsInstance(app.screen, ListeningScreen)
            self.assertEqual(app.screen.engine_code, "sensevoice")
            # Replaced, not stacked: no abandoned listening screen underneath.
            self.assertEqual(len(app.screen_stack), 2)
            self.assertTrue(app.is_running)


if __name__ == "__main__":
    unittest.main()
