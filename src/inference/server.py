"""
WebSocket inference server for the browser extension.

The browser extension captures tab audio, buffers 3-5 second chunks,
and sends them over WebSocket as raw float32 PCM. The server runs
inference and returns the detection result.

Usage:
    python -m src.inference.server \
        --checkpoint outputs/ai_audio_detection/checkpoint_best.pt \
        --config configs/default.yaml \
        --port 8765
"""

import argparse
import asyncio
import json
import struct

import numpy as np

try:
    import websockets
except ImportError:
    websockets = None

from src.inference.predictor import AudioPredictor


class InferenceServer:
    def __init__(self, predictor: AudioPredictor, host: str = "0.0.0.0", port: int = 8765):
        self.predictor = predictor
        self.host = host
        self.port = port

    async def handle_connection(self, websocket):
        """Handle a single browser extension connection."""
        print(f"Client connected: {websocket.remote_address}")
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    result = self._process_audio_bytes(message)
                elif isinstance(message, str):
                    result = self._process_json_message(message)
                else:
                    result = {"error": "Unsupported message type"}

                await websocket.send(json.dumps(result))
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            print(f"Client disconnected: {websocket.remote_address}")

    def _process_audio_bytes(self, data: bytes) -> dict:
        """
        Process raw audio bytes from browser extension.

        Expected format:
            - First 4 bytes: sample rate as uint32 (little-endian)
            - Remaining bytes: float32 PCM audio samples (little-endian)
        """
        if len(data) < 8:
            return {"error": "Audio data too short"}

        sample_rate = struct.unpack("<I", data[:4])[0]
        audio_data = np.frombuffer(data[4:], dtype=np.float32)

        result = self.predictor.predict(audio_data, sample_rate=sample_rate)
        return result

    def _process_json_message(self, message: str) -> dict:
        """Handle JSON control messages (config updates, health checks)."""
        try:
            msg = json.loads(message)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON"}

        if msg.get("type") == "ping":
            return {"type": "pong"}

        if msg.get("type") == "set_threshold":
            self.predictor.confidence_threshold = float(msg["threshold"])
            return {"type": "threshold_updated", "threshold": self.predictor.confidence_threshold}

        return {"error": f"Unknown message type: {msg.get('type')}"}

    async def start(self):
        if websockets is None:
            raise ImportError("Install websockets: pip install websockets")
        async with websockets.serve(self.handle_connection, self.host, self.port):
            print(f"Inference server running on ws://{self.host}:{self.port}")
            await asyncio.Future()  # run forever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--threshold", type=float, default=0.7)
    args = parser.parse_args()

    predictor = AudioPredictor(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
        confidence_threshold=args.threshold,
    )

    server = InferenceServer(predictor, host=args.host, port=args.port)
    asyncio.run(server.start())


if __name__ == "__main__":
    main()
