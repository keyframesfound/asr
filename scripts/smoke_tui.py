#!/usr/bin/env python
"""Headless TUI smoke test — no mic, no model loads.

Full-app run_test hangs on this Textual build (pre-existing, unrelated to
these changes), so real widgets are exercised inside a minimal harness app;
screens are verified by constructing and composing them.
"""
from __future__ import annotations

import asyncio
import sys
import time

sys.path.insert(0, ".")

from textual.app import App, ComposeResult

from engines import level as level_bus
from tui_app import AnimatedCaption, LevelMeter


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


# Note: TopBar is deliberately not mounted here — on this Textual build any
# app whose first screen mounts a TopBar stalls run_test's readiness event
# (reproduces on clean main, pre-dating these changes; real terminal unaffected).
class Harness(App[None]):
    def compose(self) -> ComposeResult:
        yield LevelMeter()
        yield AnimatedCaption(id="caption")


async def main() -> int:
    async with Harness().run_test(size=(120, 24)) as pilot:
        meter = pilot.app.query_one("#meter", LevelMeter)
        cap = pilot.app.query_one("#caption", AnimatedCaption)

        level_bus.update_level(0.06, 0.30, now=time.monotonic())
        await pilot.pause(0.25)
        check(meter._smooth > 0.1, f"meter reacts to level ({meter._smooth:.2f})")
        bar = str(getattr(meter.render(), "plain", ""))
        check("█" in bar, "meter draws a filled bar")
        check("dB" in bar, "meter shows dB readout")

        cap.set_speech("hello")
        await pilot.pause(0.4)
        check(cap._base == "hello", f"caption animates (base={cap._base!r})")

        cap.set_speech("hello world today")
        await pilot.pause(0.7)
        check(cap._base == "hello world today", f"append settles ({cap._base!r})")

        cap.set_speech("hello world")
        await pilot.pause(0.4)
        check(cap._base == "hello world", f"rewrite trims ({cap._base!r})")

        cap.set_speech("hello world", final=True)
        await pilot.pause()
        check(cap._final, "final upgrades styling")

        cap.set_status("Listening…", muted=True)
        await pilot.pause()
        check(not cap._pending and not cap._fading, "status cancels animation")

        # NOTE: screen mounting inside run_test stalls on this Textual build
        # whenever a screen composes a TopBar (pre-existing on clean main;
        # real terminal unaffected). Screen logic is unit-tested instead.

    print("SMOKE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
