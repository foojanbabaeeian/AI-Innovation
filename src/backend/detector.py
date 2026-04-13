"""
Async wrapper around the AudioPredictor.

Model inference is CPU-bound and blocking. We run it in a thread pool
so the FastAPI event loop stays responsive and we can handle many
concurrent WebSocket sessions from different browser tabs.

Also supports a --mock mode for local development when no trained
checkpoint is available yet.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class MockPredictor:
    """
    Returns randomized but plausible detection results.
    Use when no trained checkpoint exists yet so the backend + extension
    can be wired up end-to-end before the model is ready.
    """

    CLASS_NAMES = ["real", "mixed", "ai"]

    def __init__(self, confidence_threshold: float = 0.7, bias_ai: float = 0.3):
        self.confidence_threshold = confidence_threshold
        self._bias_ai = bias_ai  # probability that a given window returns "ai"
        self.target_sr = 16000
        self.segment_length = 64000

    def predict(self, waveform: np.ndarray, sample_rate: int = 44100) -> dict:
        is_ai = random.random() < self._bias_ai
        if is_ai:
            ai_score = random.uniform(0.65, 0.95)
            class_probs = [0.05, 0.15, 0.80]
            predicted_class = "ai"
        else:
            ai_score = random.uniform(0.05, 0.35)
            class_probs = [0.80, 0.15, 0.05]
            predicted_class = "real"

        return {
            "ai_score": round(ai_score, 4),
            "class": predicted_class,
            "class_probabilities": dict(zip(self.CLASS_NAMES, [round(p, 4) for p in class_probs])),
            "is_ai_detected": ai_score >= self.confidence_threshold,
            "confidence": round(max(class_probs), 4),
        }


class AsyncDetector:
    """
    Thread-pool wrapper for the predictor so it doesn't block the event loop.

    All WebSocket sessions share a single predictor instance (the model is
    thread-safe for inference). Concurrency is bounded by the executor
    thread count to avoid swamping the CPU.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config_path: str = "configs/default.yaml",
        device: str = "cpu",
        confidence_threshold: float = 0.7,
        max_concurrent_inferences: int = 4,
        mock: bool = False,
    ):
        self.confidence_threshold = confidence_threshold
        self._semaphore = asyncio.Semaphore(max_concurrent_inferences)

        if mock or not checkpoint_path:
            logger.warning("Running in MOCK mode (no trained checkpoint loaded)")
            self._predictor = MockPredictor(confidence_threshold=confidence_threshold)
        else:
            # Import lazily so mock mode works without torch
            from src.inference.predictor import AudioPredictor
            self._predictor = AudioPredictor(
                checkpoint_path=checkpoint_path,
                config_path=config_path,
                device=device,
                confidence_threshold=confidence_threshold,
            )
            logger.info(f"Loaded model from {checkpoint_path} on device={device}")

    @property
    def target_sample_rate(self) -> int:
        return self._predictor.target_sr

    @property
    def segment_length(self) -> int:
        return self._predictor.segment_length

    def set_threshold(self, threshold: float):
        self.confidence_threshold = threshold
        self._predictor.confidence_threshold = threshold

    async def predict(self, waveform: np.ndarray, sample_rate: int) -> dict:
        """
        Run inference asynchronously. Bounded by semaphore so we never run
        more than N concurrent predictions even under many connections.
        """
        async with self._semaphore:
            return await asyncio.to_thread(self._predictor.predict, waveform, sample_rate)
