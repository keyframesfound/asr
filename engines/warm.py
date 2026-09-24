"""One warm model at a time — drop previous engine globals before loading another."""
from __future__ import annotations


def drop_all_models(*, keep: str | None = None) -> None:
    """Clear cached model globals for every engine except optional ``keep`` code."""
    keep = (keep or "").strip().lower() or None

    if keep != "parakeet":
        try:
            from . import parakeet as pk

            pk.unload()
        except Exception:
            pass

    if keep != "whisper":
        try:
            from . import whisper as wh

            wh.unload()
        except Exception:
            pass

    if keep != "sensevoice":
        try:
            from . import sensevoice as sv

            sv.unload()
        except Exception:
            pass
