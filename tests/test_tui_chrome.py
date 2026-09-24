"""Headless mount checks for the Textual TUI.

TopBar used to store its label on ``_context``, which shadows
``MessagePump._context`` (a context manager). The pump then died with
``TypeError: 'str' object is not callable`` and the first paint stayed blank.
"""
from __future__ import annotations

import unittest
from unittest import mock

from textual.dom import DOMNode
from textual.widget import Widget

from tui_app import (
    ENGINES,
    AudioLiveApp,
    KeyHint,
    ListeningScreen,
    ModelPickerScreen,
    TopBar,
)


def _plain(widget: Widget) -> str:
    content = getattr(widget, "content", "")
    plain = getattr(content, "plain", None)
    if isinstance(plain, str):
        return plain
    return str(content)


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
            self.assertEqual(top._context_label, "models")
            self.assertTrue(callable(top._context))
            self.assertEqual(len(models.children), len(ENGINES))
            titles = [_plain(w) for w in screen.query(".model-title")]
            self.assertIn("Parakeet Unified EN", titles)
            self.assertIn("enter", _plain(hint))
            self.assertIn("quit", _plain(hint))

            top.set_context("renamed")
            await pilot.pause()
            self.assertEqual(top._context_label, "renamed")
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
            hint = screen.query_one("#keyhint", KeyHint)
            status = screen.query_one("#status")

            for widget, label in (
                (top, "top-bar"),
                (caption, "caption"),
                (hint, "keyhint"),
                (status, "status"),
            ):
                self.assertGreater(widget.size.width, 0, label)
                self.assertGreater(widget.size.height, 0, label)

            self.assertIn("asr", _plain(top))
            self.assertIn("Parakeet Unified EN", _plain(top))
            self.assertIn("Listening", _plain(caption))
            self.assertIn("space", _plain(hint))
            self.assertIn("export", _plain(hint))

            # Stopping the session must not clear the message-pump flag.
            self.assertTrue(screen.is_running)
            self.assertTrue(screen._session_running)
            screen._set_live_ui(False)
            self.assertFalse(screen._session_running)
            self.assertTrue(screen.is_running)
            self.assertTrue(callable(screen._context))


if __name__ == "__main__":
    unittest.main()
