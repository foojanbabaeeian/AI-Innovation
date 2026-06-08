"""
Export trained model to ONNX format for browser-side inference
via ONNX Runtime Web, or for optimized server-side deployment.

Usage:
    python -m src.inference.export_onnx \
        --checkpoint outputs/ai_audio_detection/checkpoint_best.pt \
        --output model.onnx \
        --config configs/default.yaml
"""

import argparse

import torch
import onnx

from src.models.fusion_model import MultiBranchFusionModel
from src.utils.config import Config


def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    config_path: str = "configs/default.yaml",
    opset_version: int = 17,
):
    config = Config.from_yaml(config_path)

    model = MultiBranchFusionModel(
        sample_rate=config.data.target_sr,
        embed_dim=config.model.embed_dim,
        num_attention_heads=config.model.num_attention_heads,
        num_attention_layers=config.model.num_attention_layers,
        num_classes=config.model.num_classes,
        ssl_model_name=config.model.ssl_model_name,
        disable_branches=list(config.model.disable_branches),
        fusion_method=config.model.fusion_method,
        ssl_layer_mode=config.model.ssl_layer_mode,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Dummy input: (batch=1, channels=1, samples=64000 for 4s@16kHz)
    dummy_input = torch.randn(1, 1, config.data.segment_length)

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=["waveform"],
        output_names=["regression_score", "class_logits"],
        dynamic_axes={
            "waveform": {0: "batch_size"},
            "regression_score": {0: "batch_size"},
            "class_logits": {0: "batch_size"},
        },
    )

    # Validate
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model exported and validated: {output_path}")

    # Print model size
    import os
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Model size: {size_mb:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="model.onnx")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    export_to_onnx(args.checkpoint, args.output, args.config)
