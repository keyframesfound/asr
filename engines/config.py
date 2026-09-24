"""Load user-tunable settings from config.json next to the app root.

``load_config`` is the dict API used by the TUI and live engines (cooldown
seconds, VAD gate, Parakeet feed). ``load_audio_config`` is the typed view
for mic selection and the shared RMS / cooldown helper. Keys starting with
``//`` are comments. A missing or invalid file falls back to the defaults.
No secrets belong in this file.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("engines.config")

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.json"

# Live-engine knobs (main). post_final_cooldown_sec is what SenseVoice,
# Whisper, and Parakeet read when they drop audio after a final.
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

# Shared echo-guard knobs. sample_rate stays 16000 even if the file disagrees.
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_COOLDOWN_MS = 1200
DEFAULT_MIN_RMS = 0.003
DEFAULT_TARGET_RMS = 0.025
# Zoom/Teams/Steam/EShare plus the virtual devices main already refused.
DEFAULT_DEVICE_BLOCKLIST: tuple[str, ...] = (
    "Zoom",
    "Teams",
    "Steam",
    "EShare",
    "BlackHole",
    "Loopback",
    "Soundflower",
    "Aggregate",
    "Multi-output",
)
DEFAULT_DEVICE_PREFER: tuple[str, ...] = (
    "AirPods",
    "Headset",
    "Built-in",
    "MacBook",
    "USB",
)

_MAX_COOLDOWN_MS = 5000


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = DEFAULT_SAMPLE_RATE
    cooldown_ms: int = DEFAULT_COOLDOWN_MS
    min_rms: float = DEFAULT_MIN_RMS
    target_rms: float = DEFAULT_TARGET_RMS
    device_blocklist: tuple[str, ...] = DEFAULT_DEVICE_BLOCKLIST
    device_prefer: tuple[str, ...] = DEFAULT_DEVICE_PREFER


def default_config_path() -> Path:
    return CONFIG_PATH


def load_config() -> dict[str, Any]:
    """Return config values (comment keys starting with // are ignored)."""
    data = dict(_DEFAULTS)
    raw = _read_object(CONFIG_PATH)
    for key, value in raw.items():
        if str(key).startswith("//"):
            continue
        data[key] = value

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

    data["parakeet_feed_sec"] = max(0.2, min(1.0, float(data["parakeet_feed_sec"])))
    data["post_final_cooldown_sec"] = max(0.0, min(5.0, float(data["post_final_cooldown_sec"])))
    data["mic_blocksize"] = max(512, min(16384, int(data["mic_blocksize"])))
    data["vad_min_rms"] = max(0.0, min(0.5, float(data["vad_min_rms"])))
    data["min_final_chars"] = max(1, min(64, int(data["min_final_chars"])))
    return data


_CACHE: dict[str, AudioConfig] = {}


def clear_audio_config_cache() -> None:
    _CACHE.clear()


def load_audio_config(path: Path | None = None, *, use_cache: bool = True) -> AudioConfig:
    """Typed echo-guard tunables. A missing file returns :class:`AudioConfig` defaults."""
    resolved = (path or default_config_path()).resolve()
    key = str(resolved)
    if use_cache and key in _CACHE:
        return _CACHE[key]
    cfg = _load_audio(resolved)
    if use_cache:
        _CACHE[key] = cfg
    return cfg


def _load_audio(path: Path) -> AudioConfig:
    data = _read_object(path)
    sample_rate = DEFAULT_SAMPLE_RATE
    if "sample_rate" in data and data["sample_rate"] != DEFAULT_SAMPLE_RATE:
        logger.warning(
            "sample_rate %r ignored; live engines stay at %s Hz mono float32",
            data["sample_rate"],
            DEFAULT_SAMPLE_RATE,
        )

    cooldown_ms = _cooldown(data.get("cooldown_ms")) if "cooldown_ms" in data else DEFAULT_COOLDOWN_MS
    min_rms = _unit_float(data.get("min_rms"), DEFAULT_MIN_RMS) if "min_rms" in data else DEFAULT_MIN_RMS
    target_rms = (
        _unit_float(data.get("target_rms"), DEFAULT_TARGET_RMS)
        if "target_rms" in data
        else DEFAULT_TARGET_RMS
    )
    if target_rms < min_rms:
        target_rms = min_rms

    blocklist = (
        _str_tuple(data.get("device_blocklist"))
        if "device_blocklist" in data
        else DEFAULT_DEVICE_BLOCKLIST
    )
    prefer = (
        _str_tuple(data.get("device_prefer")) if "device_prefer" in data else DEFAULT_DEVICE_PREFER
    )
    if blocklist is None:
        blocklist = DEFAULT_DEVICE_BLOCKLIST
    if prefer is None:
        prefer = DEFAULT_DEVICE_PREFER

    return AudioConfig(
        sample_rate=sample_rate,
        cooldown_ms=cooldown_ms,
        min_rms=min_rms,
        target_rms=target_rms,
        device_blocklist=blocklist,
        device_prefer=prefer,
    )


def _read_object(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not read %s (%s); using built-in audio defaults", path, exc)
        return {}
    if not text.strip():
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("Ignoring invalid %s (%s); using built-in audio defaults", path, exc)
        return {}
    if not isinstance(data, dict):
        logger.warning("Ignoring %s (expected a JSON object); using built-in audio defaults", path)
        return {}
    return data


def _cooldown(value: object) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return DEFAULT_COOLDOWN_MS
    if parsed < 0 or parsed > _MAX_COOLDOWN_MS:
        return DEFAULT_COOLDOWN_MS
    return parsed


def _unit_float(value: object, default: float) -> float:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if parsed <= 0.0 or parsed > 1.0:
        return default
    return parsed


def _str_tuple(value: object) -> tuple[str, ...] | None:
    """Return patterns, or None if the value is not a list (caller uses defaults).

    An empty list is kept so a config can explicitly disable the blocklist
    or the prefer hints.
    """
    if not isinstance(value, list):
        return None
    items: list[str] = []
    for item in value:
        if isinstance(item, str):
            text = item.strip()
            if text:
                items.append(text)
    return tuple(items)
