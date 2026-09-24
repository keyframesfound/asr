"""iFlytek live streaming ASR (语音听写流式版)."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import queue
import ssl
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time

import websocket
from dotenv import load_dotenv

from .base import LiveEngine, OnFinal, OnPartial, SessionSummary
from .mic import float32_to_pcm16, listening_label, open_input_stream

HOST = "iat-api.xfyun.cn"
PATH = "/v2/iat"
STATUS_FIRST, STATUS_CONT, STATUS_LAST = 0, 1, 2


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_env() -> None:
    load_dotenv(_project_root() / ".env")


def _auth_url(api_key: str, api_secret: str) -> str:
    now = datetime.now(timezone.utc)
    date = format_date_time(mktime(now.timetuple()))
    signature_origin = f"host: {HOST}\ndate: {date}\nGET {PATH} HTTP/1.1"
    digest = hmac.new(
        api_secret.encode("utf-8"),
        signature_origin.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).digest()
    signature = base64.b64encode(digest).decode("utf-8")
    authorization_origin = (
        f'api_key="{api_key}", algorithm="hmac-sha256", '
        f'headers="host date request-line", signature="{signature}"'
    )
    authorization = base64.b64encode(authorization_origin.encode("utf-8")).decode("utf-8")
    params = urlencode({"authorization": authorization, "date": date, "host": HOST})
    return f"wss://{HOST}{PATH}?{params}"


def _frame(app_id: str, audio_b64: str, status: int, language: str, accent: str) -> str:
    if status == STATUS_FIRST:
        return json.dumps(
            {
                "common": {"app_id": app_id},
                "business": {
                    "language": language,
                    "domain": "iat",
                    "accent": accent,
                    "vad_eos": 3000,
                    "dwa": "wpgs",
                },
                "data": {
                    "status": status,
                    "format": "audio/L16;rate=16000",
                    "encoding": "raw",
                    "audio": audio_b64,
                },
            }
        )
    return json.dumps(
        {
            "data": {
                "status": status,
                "format": "audio/L16;rate=16000",
                "encoding": "raw",
                "audio": audio_b64,
            }
        }
    )


class IflytekEngine(LiveEngine):
    name = "iFlytek live ASR"

    def run(
        self,
        on_partial: OnPartial,
        on_final: OnFinal,
        *,
        sample_rate: int = 16000,
        stop_event: threading.Event | None = None,
    ) -> SessionSummary:
        _load_env()
        app_id = os.getenv("XFYUN_APP_ID") or os.getenv("APPID") or ""
        api_key = os.getenv("XFYUN_API_KEY") or os.getenv("APIKey") or ""
        api_secret = os.getenv("XFYUN_API_SECRET") or os.getenv("APISecret") or ""
        language = os.getenv("XFYUN_LANGUAGE", "zh_cn")
        accent = os.getenv("XFYUN_ACCENT", "mandarin")

        summary = SessionSummary(engine=self.name)
        if not (app_id and api_key and api_secret):
            summary.error = (
                "Missing iFlytek credentials. Copy .env.example to .env and set "
                "XFYUN_APP_ID / XFYUN_API_KEY / XFYUN_API_SECRET."
            )
            return summary

        stop = stop_event or threading.Event()
        result_lock = threading.Lock()
        last_partial = ""

        def on_message(ws, message):  # noqa: ANN001
            nonlocal last_partial
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                return
            code = payload.get("code", -1)
            if code != 0:
                summary.error = f"iFlytek error {code}: {payload.get('message')}"
                stop.set()
                try:
                    ws.close()
                except Exception:
                    pass
                return
            data = payload.get("data") or {}
            result = data.get("result") or {}
            ws_list = result.get("ws") or []
            text = "".join(
                w.get("w", "") for item in ws_list for w in (item.get("cw") or [])
            )
            if not text:
                if data.get("status") == 2:
                    stop.set()
                return
            pgs = result.get("pgs")
            with result_lock:
                if pgs == "rpl":
                    last_partial = text
                    summary.partial_count += 1
                    on_partial(text)
                elif data.get("status") == 2:
                    summary.final_count += 1
                    summary.finals.append(text)
                    on_final(text)
                    last_partial = ""
                else:
                    if pgs == "apd":
                        last_partial = last_partial + text
                    else:
                        last_partial = text
                    summary.partial_count += 1
                    on_partial(last_partial)
            if data.get("status") == 2:
                stop.set()

        def on_error(_ws, error):  # noqa: ANN001
            summary.error = str(error)
            stop.set()

        def on_close(_ws, *_args):  # noqa: ANN001
            stop.set()

        def on_open(ws):
            def sender():
                stream, q = open_input_stream(sample_rate=sample_rate)
                stream.start()
                on_partial(listening_label())
                status = STATUS_FIRST
                try:
                    while not stop.is_set():
                        try:
                            block = q.get(timeout=0.2)
                        except queue.Empty:
                            continue
                        pcm = float32_to_pcm16(block)
                        b64 = base64.b64encode(pcm).decode("utf-8")
                        ws.send(_frame(app_id, b64, status, language, accent))
                        status = STATUS_CONT
                        time.sleep(0.04)
                except KeyboardInterrupt:
                    stop.set()
                finally:
                    try:
                        ws.send(_frame(app_id, "", STATUS_LAST, language, accent))
                    except Exception:
                        pass
                    stream.stop()
                    stream.close()
                    time.sleep(0.3)
                    try:
                        ws.close()
                    except Exception:
                        pass

            threading.Thread(target=sender, daemon=True).start()

        url = _auth_url(api_key, api_secret)
        ws_app = websocket.WebSocketApp(
            url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )
        try:
            ws_app.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
        except KeyboardInterrupt:
            stop.set()
            try:
                ws_app.close()
            except Exception:
                pass
        return summary
