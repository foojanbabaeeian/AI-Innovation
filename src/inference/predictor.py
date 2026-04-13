"""
Inference predictor for real-time audio classification.

Supports two modes:
1. PyTorch inference (server-side, full model)
2. ONNX Runtime inference (client-side browser extension or edge)

The browser extension sends audio chunks via WebSocket to the prediction
server, which returns real-time AI detection scores.
"""

import numpy as np
import torch
import torchaudio.transforms as T

from src.models.fusion_model import MultiBranchFusionModel
from src.utils.config import Config


class AudioPredictor:
    """
    Real-time inference wrapper.
    Loads a trained checkpoint and classifies audio segments.
    """

    CLASS_NAMES = ["real", "mixed", "ai"]

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = "configs/default.yaml",
        device: str = "cpu",
        confidence_threshold: float = 0.7,
    ):
        self.config = Config.from_yaml(config_path)
        self.device = torch.device(device)
        self.confidence_threshold = confidence_threshold
        self.target_sr = self.config.data.target_sr
        self.segment_length = self.config.data.segment_length

        # Load model
        self.model = MultiBranchFusionModel(
            sample_rate=self.config.data.target_sr,
            embed_dim=self.config.model.embed_dim,
            num_attention_heads=self.config.model.num_attention_heads,
            num_attention_layers=self.config.model.num_attention_layers,
            num_classes=self.config.model.num_classes,
            ssl_model_name=self.config.model.ssl_model_name,
        ).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, waveform: np.ndarray, sample_rate: int = 44100) -> dict:
        """
        Classify a single audio segment.

        Args:
            waveform: 1D numpy array of audio samples (float32, [-1, 1])
            sample_rate: sample rate of input audio

        Returns:
            {
                "ai_score": float,         # 0.0 (real) to 1.0 (AI)
                "class": str,              # "real", "mixed", or "ai"
                "class_probabilities": {str: float},
                "is_ai_detected": bool,    # ai_score >= threshold
                "confidence": float,
            }
        """
        # Convert to tensor
        audio = torch.from_numpy(waveform).float()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # (1, samples)

        # Resample if needed
        if sample_rate != self.target_sr:
            resampler = T.Resample(sample_rate, self.target_sr)
            audio = resampler(audio)

        # Pad or trim to expected length
        if audio.shape[-1] < self.segment_length:
            pad = self.segment_length - audio.shape[-1]
            audio = torch.nn.functional.pad(audio, (0, pad))
        else:
            audio = audio[..., :self.segment_length]

        # Add batch dim: (1, 1, segment_length)
        audio = audio.unsqueeze(0).to(self.device)

        outputs = self.model(audio)

        ai_score = outputs["regression_score"].item()
        class_probs = torch.softmax(outputs["class_logits"], dim=-1).squeeze().cpu().numpy()

        predicted_class_idx = int(np.argmax(class_probs))
        predicted_class = self.CLASS_NAMES[predicted_class_idx]

        return {
            "ai_score": round(ai_score, 4),
            "class": predicted_class,
            "class_probabilities": {
                name: round(float(prob), 4)
                for name, prob in zip(self.CLASS_NAMES, class_probs)
            },
            "is_ai_detected": ai_score >= self.confidence_threshold,
            "confidence": round(float(class_probs[predicted_class_idx]), 4),
        }

    @torch.no_grad()
    def predict_streaming(self, audio_chunks: list[np.ndarray], sample_rate: int = 44100) -> dict:
        """
        Classify a sequence of audio chunks (sliding window).
        Aggregates predictions across chunks for a more robust result.

        Used by the browser extension which sends overlapping 3-5s chunks.
        """
        predictions = []
        for chunk in audio_chunks:
            pred = self.predict(chunk, sample_rate)
            predictions.append(pred)

        if not predictions:
            return {"ai_score": 0.0, "class": "real", "is_ai_detected": False}

        # Aggregate: weighted average of recent chunks (more weight to newer audio)
        weights = np.linspace(0.5, 1.0, len(predictions))
        weights /= weights.sum()

        avg_score = sum(p["ai_score"] * w for p, w in zip(predictions, weights))
        avg_probs = {
            name: sum(p["class_probabilities"][name] * w for p, w in zip(predictions, weights))
            for name in self.CLASS_NAMES
        }

        predicted_class = max(avg_probs, key=avg_probs.get)

        return {
            "ai_score": round(float(avg_score), 4),
            "class": predicted_class,
            "class_probabilities": {k: round(v, 4) for k, v in avg_probs.items()},
            "is_ai_detected": avg_score >= self.confidence_threshold,
            "confidence": round(float(avg_probs[predicted_class]), 4),
            "num_chunks_analyzed": len(predictions),
        }
