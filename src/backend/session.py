"""
Per-WebSocket session state.

Each browser tab that activates the extension gets one Session. The
session owns:
    - The audio buffer (sliding window of recent audio)
    - The detection history (recent scores for smoothing)
    - The current alert state (so we only notify on state *changes*,
      not on every detection window)

Detection smoothing uses:
    1. Exponential moving average (EMA) of raw ai_score to damp noise
    2. Hysteresis thresholds: different thresholds for entering vs. leaving
       the "AI detected" state to prevent flicker near the boundary
    3. Minimum-duration constraint: must stay in new state for N seconds
       before we emit an alert
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

from src.backend.audio_buffer import AudioWindowBuffer
from src.backend.detector import AsyncDetector

logger = logging.getLogger(__name__)


@dataclass
class SessionStats:
    windows_analyzed: int = 0
    ai_detections: int = 0
    alerts_emitted: int = 0
    started_at_ms: Optional[int] = None
    stopped_at_ms: Optional[int] = None


@dataclass
class AlertState:
    """Tracks the stable detection state to decide when to fire alerts."""
    # "clear" = we believe audio is real; "ai" = we believe it is AI
    current: Literal["clear", "ai"] = "clear"
    # When did the current state start?
    started_at_ms: int = 0
    # How long must the new state hold before we fire an alert? (debounce)
    enter_debounce_ms: int = 1500
    clear_debounce_ms: int = 3000  # slower to clear than to detect
    # Has the most recent alert for THIS state already been emitted?
    alert_fired: bool = True


class DetectionSession:
    """
    Stateful session for one browser tab.

    Workflow:
        1. Client sends StartMessage with its sample_rate
        2. Session creates an AudioWindowBuffer sized for the model
        3. Client streams binary audio chunks
        4. append_audio() returns True when a new window is ready for inference
        5. run_inference() runs the model and returns a DetectionResult
        6. update_alert_state() returns an alert if the stable state changed
    """

    def __init__(
        self,
        detector: AsyncDetector,
        window_seconds: float = 4.0,
        hop_seconds: float = 1.0,
        # EMA smoothing weight: higher = snappier, lower = smoother
        ema_alpha: float = 0.4,
        # Hysteresis: require smoothed score > threshold_high to trigger AI;
        # require smoothed score < threshold_low to clear back to real.
        threshold_high: Optional[float] = None,
        threshold_low: float = 0.4,
    ):
        self.id = str(uuid.uuid4())
        self.detector = detector

        self.window_seconds = window_seconds
        self.hop_seconds = hop_seconds
        self.ema_alpha = ema_alpha
        self.threshold_high = threshold_high if threshold_high is not None else detector.confidence_threshold
        self.threshold_low = threshold_low

        self.is_running = False
        self.source_sample_rate: Optional[int] = None
        self.tab_url: Optional[str] = None
        self.tab_title: Optional[str] = None

        self._buffer: Optional[AudioWindowBuffer] = None
        self._smoothed_score: Optional[float] = None
        self._alert_state = AlertState()
        self.stats = SessionStats()

    # --------------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------------

    def start(self, sample_rate: int, tab_url: Optional[str] = None, tab_title: Optional[str] = None):
        self.source_sample_rate = sample_rate
        self.tab_url = tab_url
        self.tab_title = tab_title
        self._buffer = AudioWindowBuffer(
            source_sample_rate=sample_rate,
            target_sample_rate=self.detector.target_sample_rate,
            window_seconds=self.window_seconds,
            hop_seconds=self.hop_seconds,
        )
        self._smoothed_score = None
        self._alert_state = AlertState(
            current="clear",
            started_at_ms=_now_ms(),
            alert_fired=True,
        )
        self.stats = SessionStats(started_at_ms=_now_ms())
        self.is_running = True
        logger.info(f"Session {self.id[:8]} started (src_sr={sample_rate}, tab={tab_title!r})")

    def stop(self):
        self.is_running = False
        self.stats.stopped_at_ms = _now_ms()
        logger.info(
            f"Session {self.id[:8]} stopped "
            f"(windows={self.stats.windows_analyzed}, "
            f"ai_detections={self.stats.ai_detections}, "
            f"alerts={self.stats.alerts_emitted})"
        )

    def set_threshold(self, threshold: float):
        """Adjust the upper threshold only; the lower bound stays fixed for stability."""
        self.threshold_high = threshold
        logger.debug(f"Session {self.id[:8]} threshold_high set to {threshold}")

    # --------------------------------------------------------------
    # Audio ingestion
    # --------------------------------------------------------------

    def append_audio(self, pcm_float32: np.ndarray) -> bool:
        """
        Add audio samples to the session buffer.
        Returns True when enough new audio has accumulated to run inference.
        """
        if not self.is_running or self._buffer is None:
            return False
        self._buffer.append_chunk(pcm_float32)
        return self._buffer.ready_for_inference()

    # --------------------------------------------------------------
    # Inference + smoothing
    # --------------------------------------------------------------

    async def run_inference(self) -> Optional[dict]:
        """
        Run inference on the latest window and update smoothing state.
        Returns a detection dict with:
            ai_score (raw), smoothed_score, label, is_ai_detected,
            confidence, class_probabilities, timestamp_ms
        """
        if not self.is_running or self._buffer is None:
            return None

        window = self._buffer.extract_window()
        result = await self.detector.predict(window, sample_rate=self.detector.target_sample_rate)

        raw_score = result["ai_score"]
        self._smoothed_score = self._update_ema(raw_score)

        self.stats.windows_analyzed += 1
        if result["is_ai_detected"]:
            self.stats.ai_detections += 1

        return {
            "ai_score": raw_score,
            "smoothed_score": round(self._smoothed_score, 4),
            "label": result["class"],
            "is_ai_detected": self._smoothed_score >= self.threshold_high,
            "confidence": result["confidence"],
            "class_probabilities": result["class_probabilities"],
            "timestamp_ms": _now_ms(),
        }

    def _update_ema(self, raw_score: float) -> float:
        if self._smoothed_score is None:
            self._smoothed_score = raw_score
        else:
            a = self.ema_alpha
            self._smoothed_score = a * raw_score + (1 - a) * self._smoothed_score
        return self._smoothed_score

    # --------------------------------------------------------------
    # Alert state machine
    # --------------------------------------------------------------

    def update_alert_state(self, detection: dict) -> Optional[dict]:
        """
        Update the debounced alert state based on the latest detection.
        Returns an alert dict (to be sent to the extension) if the state
        just *flipped* in a way that requires notifying the user.
        Otherwise returns None.
        """
        smoothed = detection["smoothed_score"]
        now = detection["timestamp_ms"]

        # Determine what the current raw (instantaneous, hysteresis-aware)
        # signal says. Hysteresis: only flip from clear->ai if above high,
        # only flip from ai->clear if below low.
        if self._alert_state.current == "clear":
            wants_flip = smoothed >= self.threshold_high
            target = "ai"
            debounce = self._alert_state.enter_debounce_ms
        else:
            wants_flip = smoothed < self.threshold_low
            target = "clear"
            debounce = self._alert_state.clear_debounce_ms

        if not wants_flip:
            # If we're in a fresh state that we haven't alerted on yet,
            # check whether the debounce window has elapsed.
            if not self._alert_state.alert_fired:
                held_for = now - self._alert_state.started_at_ms
                # Note: this branch is unused in current flow but kept
                # for clarity -- alerts fire on the `wants_flip` path.
                pass
            return None

        # The smoothed score is pointing toward a new state.
        # If this is the first window in the new state, start timing it.
        if target != self._alert_state.current:
            self._alert_state = AlertState(
                current=target,
                started_at_ms=now,
                enter_debounce_ms=self._alert_state.enter_debounce_ms,
                clear_debounce_ms=self._alert_state.clear_debounce_ms,
                alert_fired=False,
            )
            return None

        # Already in the new state; has it held long enough to fire?
        held_for = now - self._alert_state.started_at_ms
        if self._alert_state.alert_fired or held_for < debounce:
            return None

        self._alert_state.alert_fired = True
        self.stats.alerts_emitted += 1

        if target == "ai":
            return {
                "level": "ai_detected",
                "smoothed_score": smoothed,
                "duration_sec": round(held_for / 1000.0, 2),
                "message": "AI-generated audio detected on this page.",
                "timestamp_ms": now,
            }
        else:
            return {
                "level": "cleared",
                "smoothed_score": smoothed,
                "duration_sec": round(held_for / 1000.0, 2),
                "message": "Audio now appears to be authentic.",
                "timestamp_ms": now,
            }


def _now_ms() -> int:
    return int(time.time() * 1000)
