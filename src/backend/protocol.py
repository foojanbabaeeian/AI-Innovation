"""
WebSocket message protocol for the browser extension backend.

The extension and server exchange JSON control messages plus
binary audio frames. All types are pydantic models for validation
and auto-generated API docs.

Client (extension) -> Server messages:
    - StartMessage: user clicked "activate"; begin detection for this session
    - StopMessage: user clicked "deactivate"; pause detection
    - PingMessage: keepalive
    - SetThresholdMessage: user changed sensitivity slider
    - (binary) raw float32 PCM audio frame (handled separately)

Server -> Client messages:
    - ReadyMessage: connection accepted, session initialized
    - StartedMessage: detection is running
    - StoppedMessage: detection is paused
    - DetectionMessage: per-window classification result (sent on every inference)
    - AlertMessage: state change (real -> ai OR ai -> real); triggers notification
    - PongMessage: keepalive response
    - ErrorMessage: something went wrong
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ------------------------------------------------------------------
# Client -> Server
# ------------------------------------------------------------------

class StartMessage(BaseModel):
    type: Literal["start"] = "start"
    sample_rate: int = Field(..., description="Audio sample rate from the browser (typically 44100 or 48000)")
    tab_url: Optional[str] = Field(None, description="URL of the tab being monitored (for logging)")
    tab_title: Optional[str] = Field(None, description="Title of the tab")


class StopMessage(BaseModel):
    type: Literal["stop"] = "stop"


class PingMessage(BaseModel):
    type: Literal["ping"] = "ping"


class SetThresholdMessage(BaseModel):
    type: Literal["set_threshold"] = "set_threshold"
    threshold: float = Field(..., ge=0.0, le=1.0)


# ------------------------------------------------------------------
# Server -> Client
# ------------------------------------------------------------------

class ReadyMessage(BaseModel):
    type: Literal["ready"] = "ready"
    session_id: str
    model_sample_rate: int
    window_seconds: float
    hop_seconds: float
    threshold: float


class StartedMessage(BaseModel):
    type: Literal["started"] = "started"
    timestamp_ms: int


class StoppedMessage(BaseModel):
    type: Literal["stopped"] = "stopped"
    timestamp_ms: int
    total_windows_analyzed: int
    total_ai_detections: int


class DetectionMessage(BaseModel):
    """Per-window detection result. Sent on every inference (high frequency)."""
    type: Literal["detection"] = "detection"
    ai_score: float
    smoothed_score: float
    label: Literal["real", "mixed", "ai"]
    is_ai_detected: bool
    confidence: float
    class_probabilities: dict[str, float]
    timestamp_ms: int


class AlertMessage(BaseModel):
    """State transition. Sent on flip real<->ai. Triggers browser notification."""
    type: Literal["alert"] = "alert"
    level: Literal["ai_detected", "cleared"]
    smoothed_score: float
    duration_sec: float = Field(..., description="How long the new state has held")
    message: str
    timestamp_ms: int


class PongMessage(BaseModel):
    type: Literal["pong"] = "pong"


class ErrorMessage(BaseModel):
    type: Literal["error"] = "error"
    code: str
    message: str


# ------------------------------------------------------------------
# Helper: parse an incoming JSON message into the right type
# ------------------------------------------------------------------

CLIENT_MESSAGE_TYPES = {
    "start": StartMessage,
    "stop": StopMessage,
    "ping": PingMessage,
    "set_threshold": SetThresholdMessage,
}


def parse_client_message(raw: dict) -> BaseModel:
    """
    Parse a JSON payload from the extension into the correct message type.
    Raises ValueError on unknown or malformed messages.
    """
    msg_type = raw.get("type")
    if msg_type not in CLIENT_MESSAGE_TYPES:
        raise ValueError(f"Unknown message type: {msg_type}")
    return CLIENT_MESSAGE_TYPES[msg_type](**raw)
