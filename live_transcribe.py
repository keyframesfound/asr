"""Backward-compatible shim → terminal CLI (iFlytek by default)."""
from main import main

if __name__ == "__main__":
    raise SystemExit(main(["--engine", "iflytek"]))
