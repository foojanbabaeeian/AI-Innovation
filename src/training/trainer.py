"""
Training loop with CPU parallel processing.

Supports:
- Multi-core CPU parallelism via DataLoader workers and PyTorch thread pool
- Gradient accumulation for effective larger batch sizes
- Cosine LR scheduling with warmup
- Early stopping on validation EER
- Checkpoint saving/resuming
- MPS (Apple Silicon) or CUDA if available, otherwise CPU
"""

import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.models.fusion_model import MultiBranchFusionModel, DualHeadLoss
from src.data.dataset import build_dataloader
from src.utils.config import Config
from src.utils.metrics import compute_metrics, format_metrics_table


class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = self._setup_device()

        # Model
        self.model = MultiBranchFusionModel(
            sample_rate=config.data.target_sr,
            embed_dim=config.model.embed_dim,
            num_attention_heads=config.model.num_attention_heads,
            num_attention_layers=config.model.num_attention_layers,
            num_classes=config.model.num_classes,
            ssl_model_name=config.model.ssl_model_name,
            freeze_ssl_feature_extractor=config.model.freeze_ssl_feature_extractor,
            dropout=config.model.dropout,
        ).to(self.device)

        # Loss
        self.criterion = DualHeadLoss(
            alpha=config.training.loss_alpha,
            threshold_low=config.data.threshold_low,
            threshold_high=config.data.threshold_high,
        ).to(self.device)

        # Optimizer: different LR for SSL backbone vs. rest
        ssl_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "ssl_branch.ssl_model" in name:
                ssl_params.append(param)
            else:
                other_params.append(param)

        self.optimizer = torch.optim.AdamW([
            {"params": other_params, "lr": config.training.learning_rate},
            {"params": ssl_params, "lr": config.training.learning_rate * 0.1},
        ], weight_decay=config.training.weight_decay)

        # Scheduler
        self.scheduler = self._build_scheduler()

        # Data -- parallel loading via num_workers
        self.train_loader = build_dataloader(
            config.data.manifest_path, config.data.data_root, "train",
            batch_size=config.data.batch_size, num_workers=config.data.num_workers,
            target_sr=config.data.target_sr, segment_length=config.data.segment_length,
        )
        self.val_loader = build_dataloader(
            config.data.manifest_path, config.data.data_root, "val",
            batch_size=config.data.batch_size, num_workers=config.data.num_workers,
            target_sr=config.data.target_sr, segment_length=config.data.segment_length,
        )

        # State
        self.current_epoch = 0
        self.global_step = 0
        self.best_eer = float("inf")
        self.patience_counter = 0

        # Output
        self.output_dir = Path(config.logging.output_dir) / config.logging.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_csv_path = self.output_dir / "metrics.csv"

    def _setup_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _build_scheduler(self):
        cfg = self.config.training
        total_steps = cfg.epochs * len(self.train_loader) if hasattr(self, "train_loader") else cfg.epochs * 1000

        if cfg.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=total_steps, T_mult=1
            )
        elif cfg.scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=5
            )
        else:
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)

    def train_epoch(self) -> dict:
        self.model.train()
        epoch_losses = {"total": [], "regression": [], "classification": []}
        accum_steps = self.config.training.gradient_accumulation_steps

        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            waveform = batch["waveform"].to(self.device)
            ai_ratio = batch["ai_ratio"].to(self.device)
            class_label = batch["class_label"].to(self.device)

            outputs = self.model(waveform)
            losses = self.criterion(
                outputs["regression_score"],
                outputs["class_logits"],
                ai_ratio,
                class_label,
            )
            loss = losses["total_loss"] / accum_steps
            loss.backward()

            if (step + 1) % accum_steps == 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.grad_clip_norm
                )
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.config.training.scheduler == "cosine":
                    self.scheduler.step()

                self.global_step += 1

            epoch_losses["total"].append(losses["total_loss"].item())
            epoch_losses["regression"].append(losses["regression_loss"].item())
            epoch_losses["classification"].append(losses["classification_loss"].item())

            if (step + 1) % self.config.logging.log_every_n_steps == 0:
                avg_loss = np.mean(epoch_losses["total"][-self.config.logging.log_every_n_steps:])
                print(
                    f"  Epoch {self.current_epoch} | Step {step + 1}/{len(self.train_loader)} | "
                    f"Loss: {avg_loss:.4f} | LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                )

        return {k: float(np.mean(v)) for k, v in epoch_losses.items()}

    @torch.no_grad()
    def evaluate(self) -> dict:
        self.model.eval()
        all_reg_scores = []
        all_class_logits = []
        all_ai_ratios = []
        all_class_labels = []
        val_losses = {"total": [], "regression": [], "classification": []}

        for batch in self.val_loader:
            waveform = batch["waveform"].to(self.device)
            ai_ratio = batch["ai_ratio"].to(self.device)
            class_label = batch["class_label"].to(self.device)

            outputs = self.model(waveform)
            losses = self.criterion(
                outputs["regression_score"],
                outputs["class_logits"],
                ai_ratio,
                class_label,
            )
            val_losses["total"].append(losses["total_loss"].item())
            val_losses["regression"].append(losses["regression_loss"].item())
            val_losses["classification"].append(losses["classification_loss"].item())

            all_reg_scores.append(outputs["regression_score"].cpu().numpy())
            all_class_logits.append(outputs["class_logits"].cpu().numpy())
            all_ai_ratios.append(batch["ai_ratio"].numpy())
            all_class_labels.append(batch["class_label"].numpy())

        reg_scores = np.concatenate(all_reg_scores)
        class_logits = np.concatenate(all_class_logits)
        ai_ratios = np.concatenate(all_ai_ratios)
        class_labels = np.concatenate(all_class_labels)

        metrics = compute_metrics(reg_scores, class_logits, ai_ratios, class_labels)
        metrics["val_loss/total"] = float(np.mean(val_losses["total"]))
        metrics["val_loss/regression"] = float(np.mean(val_losses["regression"]))
        metrics["val_loss/classification"] = float(np.mean(val_losses["classification"]))
        return metrics

    def save_checkpoint(self, tag: str = "latest"):
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_eer": self.best_eer,
            "config": self.config,
        }
        path = self.output_dir / f"checkpoint_{tag}.pth"
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

    def _append_metrics_row(self, row: dict):
        write_header = not self.metrics_csv_path.exists()
        with open(self.metrics_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint["global_step"]
        self.best_eer = checkpoint["best_eer"]
        print(f"Resumed from epoch {checkpoint['epoch']}")

    def fit(self):
        """Main training loop."""
        print(f"Training on: {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"DataLoader workers: {self.config.data.num_workers}")
        print(f"Gradient accumulation steps: {self.config.training.gradient_accumulation_steps}")
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Parameters: {total_params:,} total, {trainable:,} trainable")

        for epoch in range(self.current_epoch, self.config.training.epochs):
            self.current_epoch = epoch
            t0 = time.time()

            # Train
            train_losses = self.train_epoch()

            # Evaluate
            if (epoch + 1) % self.config.logging.eval_every_n_epochs == 0:
                val_metrics = self.evaluate()
                elapsed = time.time() - t0

                val_loss = val_metrics.get("val_loss/total", float("nan"))
                val_acc = val_metrics.get("classification/accuracy", float("nan"))
                eer = val_metrics.get("binary/eer", float("inf"))

                # Check EER for early stopping and best model
                is_best = eer < self.best_eer
                if is_best:
                    self.best_eer = eer
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                patience_limit = self.config.training.early_stopping_patience
                best_marker = "  [best]" if is_best else ""
                print(
                    f"\nEpoch {epoch:>3} ({elapsed:>5.1f}s) | "
                    f"Train Loss: {train_losses['total']:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_acc:.4f} | "
                    f"Val EER: {eer:.4f} | "
                    f"Patience: {self.patience_counter}/{patience_limit}"
                    f"{best_marker}"
                )
                print(format_metrics_table(val_metrics))

                # Append per-epoch row to metrics.csv
                self._append_metrics_row({
                    "epoch": epoch,
                    "elapsed_sec": round(elapsed, 2),
                    "train_loss_total": round(train_losses["total"], 6),
                    "train_loss_regression": round(train_losses["regression"], 6),
                    "train_loss_classification": round(train_losses["classification"], 6),
                    "val_loss_total": round(val_metrics.get("val_loss/total", float("nan")), 6),
                    "val_loss_regression": round(val_metrics.get("val_loss/regression", float("nan")), 6),
                    "val_loss_classification": round(val_metrics.get("val_loss/classification", float("nan")), 6),
                    "val_accuracy": round(val_acc, 6),
                    "val_eer": round(eer, 6) if eer != float("inf") else "",
                    "val_auc_roc": round(val_metrics.get("binary/auc_roc", float("nan")), 6),
                    "val_mae": round(val_metrics.get("regression/mae", float("nan")), 6),
                    "patience_counter": self.patience_counter,
                    "is_best": int(is_best),
                    "lr": self.optimizer.param_groups[0]["lr"],
                })

                if is_best:
                    self.save_checkpoint("best")
                self.save_checkpoint("latest")

                if self.patience_counter >= patience_limit:
                    print(f"\nEarly stopping at epoch {epoch} (best EER: {self.best_eer:.4f})")
                    break

                if self.config.training.scheduler == "plateau":
                    self.scheduler.step(eer)

        print(f"\nTraining complete. Best EER: {self.best_eer:.4f}")
        self.save_checkpoint("final")
