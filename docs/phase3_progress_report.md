# Phase 3 — Prototype Progress Report

**Team 1** · Fozhan Babaeiyan Ghamsari, Anh Le, Sofia Tejada Sarria, Camille Wong
**CECS 458: AI-Powered Business Innovation** · April 17, 2026

## 1. Architecture: Three-Branch Late-Fusion Detector

The detector consumes mono 16 kHz waveforms of shape `(batch, 1, 64000)` — 4 s/clip per `configs/gpu_local.yaml:15-16` — and emits a scalar `P(AI) ∈ [0,1]` plus 3-way logits over `{real, mixed, AI}`.

**Branch 1 — Spectral (`src/models/spectral_branch.py`)**: `torchaudio.transforms.MelSpectrogram(n_fft=2048, hop=512, n_mels=128)` → `AmplitudeToDB` → 4 conv stages (32→64→128→256) → `Linear(256,128)`. Captures formant artifacts, phase inconsistencies, vocoder residuals.

**Branch 2 — SSL (`src/models/ssl_branch.py`)**: frozen `microsoft/wavlm-base-plus` via `transformers.WavLMModel`, learned softmax weights over its 13 hidden-layer outputs → mean-pool → `LayerNorm → Linear → GELU → Dropout → Linear(128)`. Captures long-range prosodic/temporal irregularities from 94 kHr of pretraining.

**Branch 3 — Raw Waveform (`src/models/rawnet_branch.py`)**: 70 learnable SincNet bandpass filters (mel-initialized, kernel 251) → 3× `Conv1d + ResBlock1D + MaxPool1d` → `Linear(256,128)`. Captures phase artifacts and codec-compression patterns directly from time-domain.

**Fusion (`src/models/fusion_model.py`)**: 2-layer, 4-head `nn.MultiheadAttention` over the three 128-d embeddings, feeding a `DualHeadLoss`:
```
L = MSE(score, ai_ratio) + 0.5 · CE(logits, class_label)
  where class_label = {0 if ai_ratio<0.2, 2 if >0.8, else 1}
```
Attention fusion is our primary technical novelty over single-branch baselines; it allows per-sample down-weighting of unreliable branches (e.g., heavily-compressed audio).

## 2. Data Pipeline

Raw sources are registered in `master_manifest.csv`; `scripts/preprocess_segments.py` resamples to 16/44.1 kHz, loudness-normalizes (EBU R128 via `pyloudnorm`), and segments into 4 s windows with 2 s hop (50 % overlap) to `master_manifest_segmented.csv`. Splits are assigned deterministically by `md5(sample_id) % 100` (70/15/15) so segments from one source stay in one split (prevents speaker leakage).

After balancing (`scripts/balance_manifest.py`) and multi-generator ingestion, the corpus spans **~180K segments across 3 domains and 8 source datasets**:

| Domain | Real sources | Fake sources | Real / Fake |
|---|---|---|---|
| Voice | LJSpeech, ASVspoof 2019/21 bonafide | ASVspoof 2019 (19 attacks, A01–A19), ASVspoof 2021 | 32.8 K / 43.1 K |
| Music | MusicCaps, FMA-small | **MusicGen** (Meta), **Suno** (chirp-v2-xxl/v3/v3.5), **Udio** (30s/120s) | ~40 K / ~25 K |
| Non-human | ESC-50, FSD50K eval | AudioGen, AudioLDM2 | ~36 K / ~4 K |

**Differentiator:** we include 5 commercial AI-music model variants (Suno, Udio) at the scale of ~3,000 stratified clips — no prior detection study has evaluated on these. ASVspoof 2019 fake attacks are subsampled per attack-ID (`scripts/balance_manifest.py:57-80`) to preserve all 19 TTS/VC methods at 1:1 real/fake ratio at the source-file level.

## 3. Training Infrastructure (`src/training/trainer.py`)

Four stacked GPU techniques:

**(a) bf16 mixed precision** via `torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)` — halves activation memory, doubles tensor-core throughput. We probe `torch.cuda.is_bf16_supported()` and fall back to fp16 + `GradScaler` on pre-Ampere GPUs. bf16's 8-bit exponent matches fp32 range, eliminating the epoch-0 NaN we observed with fp16 overflow in the fusion attention softmax.

**(b) Selective fp32 islands** wrap two precision-sensitive layers in `autocast(enabled=False)`: `SincConv.forward` (`rawnet_branch.py:64-90`) where the `1e-8` numerical guard rounds to zero under fp16, and the frozen WavLM forward (`ssl_branch.py:61-69`) whose attention logits exceed half-precision range. WavLM in fp32 costs 0 extra VRAM (no grad/optimizer state).

**(c) Gradient accumulation** preserves effective batch 64 while the forward uses micro-batch 16 (`gradient_accumulation_steps=4`, `configs/gpu_local.yaml:50`): loss is divided by `accum_steps`; optimizer steps every 4th micro-batch.

**(d) Fragmentation mitigation** via `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — the caching allocator grows blocks instead of reserving fixed pools, recovering ~1.5 GB of fragmented VRAM post-WavLM-load. `DataLoader(num_workers=4)` parallelizes WAV decoding via `soundfile`.

Training runs 50 epochs with `CosineAnnealingWarmRestarts` (base LR 1e-4, WavLM group 1e-5, 1000-step warmup), early-stopping on validation EER with patience 7. Checkpoints (`checkpoint_{latest,best}.pth`) round-trip model + optimizer + scheduler + scaler state via `torch.save`/`torch.load`; training auto-resumes from `checkpoint_latest.pth`. Per-epoch metrics (train/val loss, accuracy, EER via `scipy.optimize.brentq` on `sklearn.metrics.roc_curve`, AUC-ROC, MAE, LR, patience) are appended to `metrics.csv`.

## 4. PoC → MVP Status

**PoC (complete ✓):** finite forward/backward on `cuda:0`, real loss (~0.08 at init), `scale → unscale_ → clip_grad_norm_ → step` runs without underflow, checkpoint round-trip verified, metrics CSV logged.

**MVP success criteria:** val EER < 0.15, 3-class accuracy > 0.75, per-class recall > 0.70.

**Initial training run** (Camille Wong, 27 K-segment subset): train loss 0.0026, val EER 0.0076, val accuracy 0.988 at epoch 18. We identified this result as **overfitting on an imbalanced 78 %-real subset** — not a reportable number — and rebuilt the 180 K corpus with stratified class balance before the definitive training run, currently in progress.

## 5. Supporting Infrastructure

- **Containerization**: `Dockerfile` + 4 `k8s/` manifests (`training-job.yaml`, `inference-deployment.yaml`, `preprocessing-job.yaml`, `storage.yaml`) for multi-GPU scale-out.
- **Data augmentation** (`src/data/augmentation.py`): `CodecAugmentor` applies MP3 (32/64/128 kbps), AAC (32/64 kbps), Opus (6/12/24 kbps), and simulated VoIP (8 kHz µ-law) during training — supports the paper's codec-robustness claim (pending wire-up to `dataset.py`).
- **Reproducibility**: all splits deterministic via MD5 hash; all model checkpoints + all preprocessing scripts version-controlled on the `preprocessing` branch.

## 6. Remaining Work (Weeks 10–12)

| Deliverable | Owner | Status |
|---|---|---|
| Full training run on 180 K corpus | Camille | in progress |
| Single-branch ablations (spectral / SSL / raw) | Camille + Fozhan | code flags pending |
| Fusion-method ablation (attention vs. concat vs. avg) | Camille | pending |
| Per-domain + per-generator EER eval harness | Fozhan | `scripts/evaluate.py` pending |
| Cross-dataset eval (train 2019 → test 2021) | Camille | 1 config change |
| Browser extension (Chrome MV3) with live inference | Sofia | spec complete, dev this week |
| Paper: Related Work, Methods, Experiments | Anh | Related Work draft this week |

**Command to reproduce current training**:
```bash
python -m src.training.train --config configs/gpu_local.yaml
# → outputs/ai_audio_detection/checkpoint_best.pth
```

Code + data manifests: `github.com/foojanbabaeeian/AI-Innovation` (branch `preprocessing`).
