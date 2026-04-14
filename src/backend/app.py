"""
FastAPI application for the browser-extension backend.

Flow:
    1. Extension opens a WebSocket to /ws
    2. Server sends ReadyMessage with session_id + model params
    3. Extension sends StartMessage (with sample_rate, tab info)
    4. Extension streams binary float32 PCM audio frames
    5. Server decodes frames into numpy, feeds the DetectionSession buffer
    6. When a full hop is ready, server runs inference and emits:
         - DetectionMessage on every window
         - AlertMessage only on debounced state changes
    7. Extension can send StopMessage / SetThresholdMessage / PingMessage
    8. On disconnect, session is torn down

Binary frames are expected to be little-endian float32 mono PCM.
The client is responsible for downmixing stereo tab audio before sending.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from src.backend.detector import AsyncDetector
from src.backend.protocol import (
    AlertMessage,
    DetectionMessage,
    ErrorMessage,
    PongMessage,
    ReadyMessage,
    SetThresholdMessage,
    StartedMessage,
    StartMessage,
    StopMessage,
    StoppedMessage,
    parse_client_message,
)
from src.backend.session import DetectionSession, _now_ms

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# App state
# ----------------------------------------------------------------------

class AppState:
    """Process-global state injected at startup time."""

    def __init__(self):
        self.detector: Optional[AsyncDetector] = None
        self.active_sessions: dict[str, DetectionSession] = {}
        # Config surfaced via /config endpoint
        self.window_seconds: float = 4.0
        self.hop_seconds: float = 1.0
        self.default_threshold: float = 0.7
        self.mock: bool = False


app_state = AppState()


def configure_app_state(
    detector: AsyncDetector,
    window_seconds: float = 4.0,
    hop_seconds: float = 1.0,
    default_threshold: float = 0.7,
    mock: bool = False,
):
    """Called by run.py before uvicorn starts serving."""
    app_state.detector = detector
    app_state.window_seconds = window_seconds
    app_state.hop_seconds = hop_seconds
    app_state.default_threshold = default_threshold
    app_state.mock = mock


@asynccontextmanager
async def lifespan(app: FastAPI):
    if app_state.detector is None:
        # Lazy default so `uvicorn src.backend.app:app` still works in dev.
        logger.warning("No detector configured; falling back to mock mode")
        app_state.detector = AsyncDetector(mock=True)
        app_state.mock = True
    logger.info("Backend starting up")
    yield
    logger.info("Backend shutting down")
    for sess in list(app_state.active_sessions.values()):
        sess.stop()
    app_state.active_sessions.clear()


# ----------------------------------------------------------------------
# FastAPI app
# ----------------------------------------------------------------------

app = FastAPI(
    title="AI Audio Detection Backend",
    description="Real-time streaming AI-generated audio detection for browser extensions.",
    version="0.1.0",
    lifespan=lifespan,
)

# Browser extensions have origins like chrome-extension://<id> or moz-extension://<id>.
# We allow those plus localhost for dev. Origins can also be narrowed by the
# deployer via the FastAPI middleware config at startup.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^(chrome-extension|moz-extension|safari-web-extension)://.*$|^https?://localhost(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------
# REST endpoints
# ----------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "active_sessions": len(app_state.active_sessions),
        "mock": app_state.mock,
    }


@app.get("/config")
async def config():
    detector = app_state.detector
    return {
        "window_seconds": app_state.window_seconds,
        "hop_seconds": app_state.hop_seconds,
        "model_sample_rate": detector.target_sample_rate if detector else None,
        "threshold": detector.confidence_threshold if detector else app_state.default_threshold,
        "mock": app_state.mock,
    }


@app.get("/stats")
async def stats():
    return {
        "active_sessions": len(app_state.active_sessions),
        "sessions": [
            {
                "id": s.id,
                "tab_url": s.tab_url,
                "tab_title": s.tab_title,
                "is_running": s.is_running,
                "windows_analyzed": s.stats.windows_analyzed,
                "ai_detections": s.stats.ai_detections,
                "alerts_emitted": s.stats.alerts_emitted,
                "started_at_ms": s.stats.started_at_ms,
            }
            for s in app_state.active_sessions.values()
        ],
    }


# ----------------------------------------------------------------------
# WebSocket endpoint
# ----------------------------------------------------------------------

async def _send(ws: WebSocket, msg) -> None:
    """Serialize a pydantic message and send it as JSON text."""
    await ws.send_text(msg.model_dump_json())


async def _send_error(ws: WebSocket, code: str, message: str) -> None:
    try:
        await _send(ws, ErrorMessage(code=code, message=message))
    except Exception:
        # Client may already be gone; swallow to avoid masking the original error
        pass


def _decode_audio_frame(raw: bytes) -> np.ndarray:
    """
    Decode a binary frame into a float32 mono numpy array.
    We expect little-endian float32 PCM from the extension's AudioWorklet.
    Invalid frames raise ValueError so the caller can reply with ErrorMessage.
    """
    if len(raw) == 0:
        raise ValueError("empty audio frame")
    if len(raw) % 4 != 0:
        raise ValueError(f"frame length {len(raw)} is not a multiple of 4 bytes")
    return np.frombuffer(raw, dtype=np.float32)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()

    detector = app_state.detector
    if detector is None:
        await _send_error(websocket, "server_not_ready", "Detector not initialized")
        await websocket.close()
        return

    session = DetectionSession(
        detector=detector,
        window_seconds=app_state.window_seconds,
        hop_seconds=app_state.hop_seconds,
        threshold_high=detector.confidence_threshold,
    )
    app_state.active_sessions[session.id] = session

    try:
        await _send(
            websocket,
            ReadyMessage(
                session_id=session.id,
                model_sample_rate=detector.target_sample_rate,
                window_seconds=app_state.window_seconds,
                hop_seconds=app_state.hop_seconds,
                threshold=session.threshold_high,
            ),
        )

        # Serialize inference for this session: we never want two concurrent
        # `run_inference` calls on the same buffer, even if audio frames
        # arrive faster than the model can process them.
        inference_lock = asyncio.Lock()

        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            # ----- Binary audio frame -----
            if (data := message.get("bytes")) is not None:
                if not session.is_running:
                    # Ignore stray frames before start / after stop
                    continue
                try:
                    pcm = _decode_audio_frame(data)
                except ValueError as e:
                    await _send_error(websocket, "bad_frame", str(e))
                    continue

                ready = session.append_audio(pcm)
                if ready and not inference_lock.locked():
                    # Fire-and-forget inference; lock prevents overlap.
                    asyncio.create_task(_run_and_emit(websocket, session, inference_lock))
                continue

            # ----- JSON control message -----
            if (text := message.get("text")) is not None:
                try:
                    raw = _safe_json_loads(text)
                    msg = parse_client_message(raw)
                except (ValueError, ValidationError) as e:
                    await _send_error(websocket, "bad_message", str(e))
                    continue

                if isinstance(msg, StartMessage):
                    session.start(
                        sample_rate=msg.sample_rate,
                        tab_url=msg.tab_url,
                        tab_title=msg.tab_title,
                    )
                    await _send(websocket, StartedMessage(timestamp_ms=_now_ms()))

                elif isinstance(msg, StopMessage):
                    session.stop()
                    await _send(
                        websocket,
                        StoppedMessage(
                            timestamp_ms=_now_ms(),
                            total_windows_analyzed=session.stats.windows_analyzed,
                            total_ai_detections=session.stats.ai_detections,
                        ),
                    )

                elif isinstance(msg, SetThresholdMessage):
                    session.set_threshold(msg.threshold)

                else:  # PingMessage
                    await _send(websocket, PongMessage())

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.exception(f"WebSocket error in session {session.id[:8]}: {e}")
        await _send_error(websocket, "internal_error", str(e))
    finally:
        if session.is_running:
            session.stop()
        app_state.active_sessions.pop(session.id, None)
        try:
            await websocket.close()
        except Exception:
            pass


async def _run_and_emit(
    websocket: WebSocket,
    session: DetectionSession,
    lock: asyncio.Lock,
) -> None:
    """Run inference + emit detection / alert messages, guarded by a lock."""
    async with lock:
        try:
            detection = await session.run_inference()
        except Exception as e:
            logger.exception(f"Inference failed in session {session.id[:8]}: {e}")
            await _send_error(websocket, "inference_error", str(e))
            return

        if detection is None:
            return

        try:
            await _send(websocket, DetectionMessage(**detection))
        except Exception:
            return  # client gone

        alert = session.update_alert_state(detection)
        if alert is not None:
            try:
                await _send(websocket, AlertMessage(**alert))
            except Exception:
                return


def _safe_json_loads(text: str) -> dict:
    import json
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e}")
    if not isinstance(parsed, dict):
        raise ValueError("message must be a JSON object")
    return parsed
