#!/usr/bin/env python3
"""Audio Live Transcription — interactive terminal UI (default) or legacy CLI."""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Live microphone transcription — interactive TUI by default."
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Use the old non-TUI menu/argparse CLI instead of the full-screen TUI.",
    )
    parser.add_argument(
        "-e",
        "--engine",
        choices=["iflytek", "sensevoice", "parakeet", "whisper"],
        help="With --cli: start this engine without a menu.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List engines and exit.",
    )
    args, rest = parser.parse_known_args(argv)

    if args.list:
        from tui_app import ENGINES

        for i, opt in enumerate(ENGINES, 1):
            print(f"{i}  {opt.code:12}  {opt.title}")
        return 0

    if args.cli or args.engine:
        return _legacy_cli(args.engine)

    from tui_app import run_tui

    try:
        from engines.whisper import _patch_tqdm_no_mp
        _patch_tqdm_no_mp()
    except Exception:
        pass
    run_tui()
    return 0


def _legacy_cli(engine: str | None) -> int:
    """Kept for scripting; prefer the TUI."""
    from tui_app import ENGINES, _build_engine

    code = engine
    if not code:
        print("Audio Live Transcription (legacy CLI)")
        for i, opt in enumerate(ENGINES, 1):
            print(f"  {i}) {opt.title}")
        print("  q) Quit")
        while True:
            choice = input("Choose engine [1-4]: ").strip().lower()
            if choice in ("q", "quit", "exit"):
                return 0
            if choice.isdigit() and 1 <= int(choice) <= len(ENGINES):
                code = ENGINES[int(choice) - 1].code
                break
            print("Invalid choice.")

    eng = _build_engine(code)
    print(f"\nStarting: {eng.name}")
    print("Speak into the mic. Press Ctrl+C to stop.\n")

    def on_partial(text: str) -> None:
        sys.stdout.write(f"\r\033[K~ {text}")
        sys.stdout.flush()

    def on_final(text: str) -> None:
        sys.stdout.write(f"\r\033[K> {text}\n")
        sys.stdout.flush()

    try:
        summary = eng.run(on_partial, on_final)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0
    print("\n--- Summary ---")
    print(summary.short_text())
    return 1 if summary.error else 0


if __name__ == "__main__":
    raise SystemExit(main())
