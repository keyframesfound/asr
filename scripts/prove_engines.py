#!/usr/bin/env python3
"""Offline proof: Whisper / SenseVoice / Parakeet must transcribe a known WAV."""
from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WAV = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/asr-test/sample.wav")


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        ch = w.getnchannels()
        raw = w.readframes(n)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1)
    return audio, sr


def main() -> int:
    if not WAV.exists():
        print(f"FAIL missing wav: {WAV}")
        return 2
    audio, sr = load_wav(WAV)
    print(f"wav={WAV} sr={sr} sec={len(audio)/sr:.2f} rms={float(np.sqrt(np.mean(audio**2))):.4f}")
    results: dict[str, str] = {}

    # --- Whisper ---
    try:
        import engines.whisper as whisper
        import torch
        from engines.audio_util import prepare_chunk

        print("\n[Whisper] loading…")
        print(whisper.preload())
        chunk = prepare_chunk(audio)
        if chunk is None:
            chunk = audio.astype(np.float32)
        inputs = whisper._PROCESSOR(chunk, sampling_rate=sr, return_tensors="pt")
        feats = inputs.input_features.to(device=whisper._DEVICE, dtype=whisper._DTYPE)
        with torch.inference_mode():
            ids = whisper._MODEL.generate(feats, condition_on_prev_tokens=False)
        text = whisper._PROCESSOR.batch_decode(ids, skip_special_tokens=True)[0].strip()
        if whisper._is_junk(text):
            raise RuntimeError(f"junk-only output: {text!r}")
        results["Whisper"] = text
        print(f"[Whisper] OK: {text!r}")
    except Exception as exc:
        results["Whisper"] = f"ERROR: {exc}"
        print(f"[Whisper] FAIL: {exc}")

    # --- SenseVoice ---
    try:
        import engines.sensevoice as sensevoice
        from engines.audio_util import prepare_chunk

        print("\n[SenseVoice] loading…")
        print(sensevoice.preload())
        chunk = prepare_chunk(audio)
        if chunk is None:
            chunk = audio.astype(np.float32)
        out = sensevoice._MODEL.generate(
            input=chunk, cache={}, language="auto", use_itn=True
        )
        if isinstance(out, list) and out:
            text = str(
                out[0].get("text") if isinstance(out[0], dict) else out[0]
            ).strip()
        elif isinstance(out, dict):
            text = str(out.get("text") or "").strip()
        else:
            text = str(out or "").strip()
        if not text:
            raise RuntimeError("empty transcript")
        results["SenseVoice"] = text
        print(f"[SenseVoice] OK: {text!r}")
    except Exception as exc:
        results["SenseVoice"] = f"ERROR: {exc}"
        print(f"[SenseVoice] FAIL: {exc}")

    # --- Parakeet MLX ---
    try:
        import engines.parakeet as parakeet

        print("\n[Parakeet] loading…")
        print(parakeet.preload())
        aligned = parakeet._MODEL.transcribe(str(WAV))
        text = (aligned.text or "").strip()
        if not text:
            raise RuntimeError("empty transcript")
        results["Parakeet"] = text
        print(f"[Parakeet] OK ({parakeet.MODEL_ID}): {text!r}")
    except Exception as exc:
        results["Parakeet"] = f"ERROR: {exc}"
        print(f"[Parakeet] FAIL: {exc}")

    print("\n=== SUMMARY ===")
    ok = 0
    for name, val in results.items():
        good = not str(val).startswith("ERROR")
        ok += int(good)
        print(f"{'PASS' if good else 'FAIL'} {name}: {val!r}")
    print(f"{ok}/{len(results)} engines transcribed")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
