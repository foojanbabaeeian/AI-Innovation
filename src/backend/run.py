"""
CLI entry point for the backend server.

Usage:
    python -m src.backend.run \
        --checkpoint outputs/best.pt \
        --config configs/default.yaml \
        --host 0.0.0.0 \
        --port 8765 \
        --threshold 0.7

    # Dev without a trained model:
    python -m src.backend.run --mock
"""

from __future__ import annotations

import argparse
import logging

import uvicorn

from src.backend.app import app, configure_app_state
from src.backend.detector import AsyncDetector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI audio detection backend")
    p.add_argument("--checkpoint", type=str, default=None, help="Path to trained model checkpoint (.pt)")
    p.add_argument("--config", type=str, default="configs/default.yaml", help="Model config YAML")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--threshold", type=float, default=0.7, help="AI-detection upper threshold (hysteresis high)")
    p.add_argument("--window-seconds", type=float, default=4.0)
    p.add_argument("--hop-seconds", type=float, default=1.0)
    p.add_argument("--max-concurrent-inferences", type=int, default=4)
    p.add_argument("--mock", action="store_true", help="Skip model loading; return randomized results")
    p.add_argument("--log-level", type=str, default="info", choices=["debug", "info", "warning", "error"])
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    detector = AsyncDetector(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
        confidence_threshold=args.threshold,
        max_concurrent_inferences=args.max_concurrent_inferences,
        mock=args.mock,
    )

    configure_app_state(
        detector=detector,
        window_seconds=args.window_seconds,
        hop_seconds=args.hop_seconds,
        default_threshold=args.threshold,
        mock=args.mock or args.checkpoint is None,
    )

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        ws_ping_interval=20,
        ws_ping_timeout=20,
    )


if __name__ == "__main__":
    main()
