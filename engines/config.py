"""Load user-tunable settings from config.json next to the app root."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.json"

_DEFAULTS: dict[str, Any] = {
    "default_engine": "parakeet",
    "parakeet_feed_sec": 0.4,
    "post_final_cooldown_sec": 1.25,
    "mic_blocksize": 4096,
    "vad_min_rms": 0.02,
    "min_final_chars": 8,
}

_FLOAT_KEYS = (
    "parakeet_feed_sec",
    "post_final_cooldown_sec",
    "vad_min_rms",
)
_INT_KEYS = ("mic_blocksize", "min_final_chars")


def load_config() -> dict[str, Any]:
    """Return config values (comment keys starting with // are ignored)."""
    data = dict(_DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                for key, value in raw.items():
                    if str(key).startswith("//"):
                        continue
                    data[key] = value
        except Exception:
            pass

    for key in _FLOAT_KEYS:
        try:
            data[key] = float(data[key])
        except Exception:
            data[key] = _DEFAULTS[key]

    for key in _INT_KEYS:
        try:
            data[key] = int(data[key])
        except Exception:
            data[key] = _DEFAULTS[key]

    eng = str(data.get("default_engine") or "parakeet").strip().lower()
    if eng not in ("parakeet", "whisper", "sensevoice", "iflytek"):
        eng = "parakeet"
    data["default_engine"] = eng

    # Clamp sensible ranges
    data["parakeet_feed_sec"] = max(0.2, min(1.0, float(data["parakeet_feed_sec"])))
    data["post_final_cooldown_sec"] = max(0.0, min(5.0, float(data["post_final_cooldown_sec"])))
    data["mic_blocksize"] = max(512, min(16384, int(data["mic_blocksize"])))
    data["vad_min_rms"] = max(0.0, min(0.5, float(data["vad_min_rms"])))
    data["min_final_chars"] = max(1, min(64, int(data["min_final_chars"])))
    return data
