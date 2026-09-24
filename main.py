#!/usr/bin/env python3
"""Audio Live Transcription — terminal CLI for iFlytek + local ASR engines."""
from __future__ import annotations

import argparse
import sys


ENGINES = {
    "1": ("iflytek", "iFlytek live ASR (cloud)"),
    "2": ("sensevoice", "SenseVoice Small (local)"),
    "3": ("parakeet", "Parakeet Unified EN / CoreML (local)"),
    "4": ("whisper", "Whisper Large V3 Turbo (local)"),
}

ALIASES = {
    "iflytek": "iflytek",
    "iat": "iflytek",
    "sensevoice": "sensevoice",
    "sense": "sensevoice",
    "parakeet": "parakeet",
    "pikaret": "parakeet",
    "whisper": "whisper",
    "turbo": "whisper",
}


def _print_menu() -> str:
    print()
    print("Audio Live Transcription")
    print("------------------------")
    for key, (_code, label) in ENGINES.items():
        print(f"  {key}) {label}")
    print("  q) Quit")
    print()
    while True:
        choice = input("Choose engine [1-4]: ").strip().lower()
        if choice in ("q", "quit", "exit"):
            sys.exit(0)
        if choice in ENGINES:
            return ENGINES[choice][0]
        if choice in ALIASES:
            return ALIASES[choice]
        print("Invalid choice. Enter 1–4 or q.")


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
    raise SystemExit(f"Unknown engine: {code}")


def _print_partial(text: str) -> None:
    sys.stdout.write(f"\r\033[K~ {text}")
    sys.stdout.flush()


def _print_final(text: str) -> None:
    sys.stdout.write(f"\r\033[K> {text}\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Live microphone transcription (terminal only)."
    )
    parser.add_argument(
        "-e",
        "--engine",
        choices=sorted(set(ALIASES.values())),
        help="Skip the menu and start this engine.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List engines and exit.",
    )
    args = parser.parse_args(argv)

    if args.list:
        for key, (code, label) in ENGINES.items():
            print(f"{key}  {code:12}  {label}")
        return 0

    code = args.engine or _print_menu()
    engine = _build_engine(code)
    print(f"\nStarting: {engine.name}")
    print("Speak into the mic. Press Ctrl+C to stop.\n")

    try:
        summary = engine.run(_print_partial, _print_final)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0

    print("\n--- Summary ---")
    print(summary.short_text())
    return 1 if summary.error else 0


if __name__ == "__main__":
    raise SystemExit(main())
