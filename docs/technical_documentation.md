# Technical Documentation — AI Audio Detector

**Team 1**: Fozhan Babaeiyan Ghamsari, Anh Le, Sofia Tejada Sarria, Camille Wong
**CECS 458 — Phase 3 (Apr 17, 2026)**

This document explains the architecture and code of the AI-audio-detection prototype. It accompanies the 2-page Progress Report. The system has three parts: a **data pipeline** that ingests eight heterogeneous source datasets into a unified manifest, a **three-branch late-fusion neural network** that classifies 4-second waveforms as real vs. AI-generated, and a **deployment layer** (FastAPI backend + Chrome extension) that streams tab audio from a browser to live inference.

---

## 1. System Overview

```
 raw audio (WAV/MP3)        register_generated.py                preprocess_segments.py
      │                             │                                     │
      ▼                             ▼                                     ▼
 raw/{domain}/{real|fake}/  → master_manifest.csv (source files) → Processed/{domain}/{...}/*.wav
                                    │                                     │
                                    └──────── build_segmented_manifest.py ┘
                                                         │
                                                         ▼
                                       master_manifest_segmented.csv (196K rows)
                                                         │
                                        ┌────────────────┼────────────────┐
                                        ▼                ▼                ▼
                                    train.py         evaluate.py       gradcam.py
                                        │                │
                               checkpoint_best.pt       LaTeX tables
                                        │
                                        ▼
                                   FastAPI backend  ← WebSocket →  Chrome extension
```

All code lives under `src/` (library) and `scripts/` (entry points and ingestion utilities). Configurations are YAML under `configs/`. Containerization is via `Dockerfile` + four Kubernetes manifests in `k8s/` for multi-GPU scale-out.

---

## 2. Data Pipeline

The pipeline's core abstraction is a **two-stage manifest system**. Stage 1 lists every *source file* (one row per original audio clip); Stage 2 lists every *4-second segment* (one row per training example). Both are plain CSVs, version-controlled so any teammate with Drive access can reproduce the dataset state at any commit.

### 2.1 Source registration (`scripts/register_generated.py`)

Each dataset is registered with a single command that appends rows to `data/metadata/master_manifest.csv`:

```bash
python scripts/register_generated.py \
    --dir {data_root}/raw/music/fake/suno \
    --domain music --source suno --generator suno \
    --data-root {data_root} --pattern "*.mp3" --recursive
```

The script walks the directory (recursively if `--recursive`), constructs one row per audio file with columns `(sample_id, file_path, label, domain, source_dataset, generator, split)`, and appends them to the manifest. Paths are stored **relative** to `data_root` so the manifest is portable across machines (Colab, Windows laptops, Kubernetes pods). Duplicates are detected via `sample_id` and silently skipped, making the script idempotent — safe to rerun after a partial failure. Splits (`train`/`val`/`test`) are assigned deterministically at this stage via `md5(sample_id) % 100`, with ratio 70/15/15 (see §2.4).

### 2.2 Preprocessing (`scripts/preprocess_segments.py`)

Once registered, each source file is converted into fixed-length training segments. The preprocessing worker (`_process_row`, running in a `ProcessPoolExecutor` with 4 workers by default) performs, for each source file:

1. **Load** via `torchaudio.load`; falls back to returning a `LOAD_ERROR` row rather than crashing if the file is corrupt.
2. **Mono-mix** (mean across channels).
3. **Resample** to the domain's target rate: 16 kHz for voice, 44.1 kHz for music and non-human sounds. Domain-specific rates preserve the harmonic content relevant to each class.
4. **Loudness-normalize** to −23 LUFS using EBU R128 (`pyloudnorm`); falls back to peak-normalize if `pyloudnorm` is not installed.
5. **Segment** into 4-second windows with a 2-second hop (50% overlap). Segments shorter than 1 second are discarded; partial final segments are zero-padded to the fixed length.
6. **Write** each segment as a 16-bit WAV to `Processed/{domain}/{real|fake}/{source}/{stem}_seg{NNNN}.wav`.

Drive Streaming in Colab occasionally returns transient `OSError(Errno 5)` during these writes. We wrap the entire worker in `try/except OSError` (`preprocess_segments.py:76-88`) so a single bad file produces an `IO_ERROR` row instead of killing the pool, and we retry `torchaudio.save` once on failure (`preprocess_segments.py:166-178`). The resulting scripts are **resumable**: skipping segments whose output file already exists means a crashed or disconnected session can be rerun and only processes what's missing.

### 2.3 Segmented manifest (`scripts/build_segmented_manifest.py`)

After preprocessing, a third script scans `Processed/` and produces `data/metadata/master_manifest_segmented.csv`. Each row represents one training example. Source, domain, and label are inferred from the directory layout (`{domain}/{real|fake}/{source}/...`), and the split is joined in from the source manifest so all segments belonging to one source file stay in the same split (preventing speaker/track leakage). This segmented manifest is the single file consumed by the training `Dataset` class.

### 2.4 Splits (`scripts/balance_manifest.py`, `scripts/rebalance_splits.py`)

Splits are computed by `assign_split(sample_id) = hash(sample_id) % 100`, mapping 0–69 → `train`, 70–84 → `val`, 85–99 → `test`. MD5 is used for the hash so splits are machine-independent and git-trackable; any change in the source manifest reproduces identical splits without coordination. Two balancing passes handle dataset-specific irregularities:

- `rebalance_splits.py`: originally ASVspoof 2019 shipped with an official protocol that placed 52% of segments in `test`, starving train. We unfroze that split and stratified 70/15/15 across all sources.
- `balance_manifest.py`: ASVspoof 2019's **fake** subset (108,978 utterances across 19 attack types A01–A19) is stratified-subsampled to ~24,000 (~1,260 per attack), yielding a 1:1 real-to-fake ratio at the source-file level while preserving coverage of all 19 spoofing methods.

### 2.5 Long-form music segment capping

Suno and Udio source clips are 120–240 seconds long. At 4s windows / 2s hop, one clip yields ~100 segments — a 1,500-clip subsample would then produce 150K+ segments and dominate the corpus. After preprocessing, a post-processing step caps each long-form source at 8 segments per source clip:

```python
MAX_PER_CLIP = {'suno': 8, 'udio': 8, 'fma_small': 6}
for src, sub in df.groupby('source_dataset'):
    sub.groupby('orig_stem').head(MAX_PER_CLIP.get(src) or len(sub))
```

This takes the total corpus from 418K to 196K segments while preserving clip diversity.

### 2.6 Codec augmentation (`src/data/augmentation.py`)

Trained with a `CodecAugmentor` (wired into `ManifestAudioDataset.__getitem__` in `src/data/dataset.py:85-91`, train split only), each segment has a 50% probability of being encoded through one of: MP3 (32/64/128 kbps), AAC (32/64 kbps), Opus (6/12/24 kbps), or simulated VoIP (8 kHz μ-law). Encoding is piped through `ffmpeg` via `stdin/stdout` to avoid temp files. This directly models the compression the detector will encounter at deployment — streaming audio over WebSocket, YouTube ingestion, phone-call audio — and supports the paper's codec-robustness claim.

### 2.7 Final corpus

After ingestion, preprocessing, balancing, and capping, the training manifest contains **~196K segments** across 11 sources and 3 domains:

| Domain | Real (sources, segments) | Fake (sources, segments) |
|---|---|---|
| Voice | LJSpeech, ASVspoof 19+21 bonafide (~33K) | ASVspoof 19 (A01–A19), ASVspoof 21 (~43K) |
| Music | MusicCaps, FMA-small (~20K) | MusicGen, Suno (chirp-v2-xxl/v3/v3.5), Udio (30s/120s) (~34K) |
| Non-human | ESC-50, FSD50K eval (~36K) | AudioGen, AudioLDM2 (~30K) |

Three commercial AI music generators (MusicGen + Suno + Udio) and two AI environmental-sound generators (AudioGen + AudioLDM2), evaluated in a single unified model.

---

## 3. Model Architecture

### 3.1 Three branches (`src/models/`)

Each branch maps a 4-second waveform `x ∈ ℝ^64000` to a 128-dimensional embedding:

- **Branch 1 — Spectral** (`spectral_branch.py`): `MelSpectrogram(n_fft=2048, hop=512, n_mels=128)` → `AmplitudeToDB` → 4-stage ResNet (32→64→128→256) → global average pool → `Linear(256, 128)`. Captures vocoder residuals, formant discontinuities, phase artifacts in the frequency domain.
- **Branch 2 — SSL** (`ssl_branch.py`): frozen `microsoft/wavlm-base-plus` (13 layers × 768 dims, pretrained on 94 khr of speech) with learned softmax-normalized weights over layer outputs → mean-pool → `LayerNorm → Linear → GELU → Dropout → Linear(256, 128)`. Captures long-range prosodic irregularities and phonetic anomalies.
- **Branch 3 — Raw waveform** (`rawnet_branch.py`): 70 learnable SincNet bandpass filters (kernel 251, mel-initialized) → 3× `Conv1d + ResBlock1D + MaxPool1d` (128→128→256) → `Linear(256, 128)`. Captures codec-compression signatures and phase-level artifacts that spectral features lose.

### 3.2 Attention fusion (`src/models/fusion_model.py`)

The three embeddings are stacked as 3 tokens `z ∈ ℝ^(3×128)` and passed through a 2-layer, 4-head `nn.MultiheadAttention`. Attention lets the model per-sample re-weight branches — e.g., heavily codec-degraded audio relies on the raw-waveform branch while clean audio relies more on SSL. Output is flattened and fed to two heads:

```python
fused ∈ ℝ^384
score   = σ(W_s · fused)     ∈ [0, 1]   # P(AI)
logits  = W_ℓ · fused         ∈ ℝ^3     # {real, mixed, AI}
```

The auxiliary 3-way classification head acts as a regularizer and provides a clean interface for mixed-content audio in future work.

### 3.3 Dual-head loss (`src/models/fusion_model.py:DualHeadLoss`)

```
L = MSE(score, ai_ratio) + 0.5 · CE(logits, class_label)

class_label = 0 if ai_ratio < 0.2    (real)
              2 if ai_ratio > 0.8    (AI)
              1 otherwise            (mixed)
```

Training on both heads jointly improves calibration of the continuous P(AI) while the categorical head anchors the decision boundary away from 0.5.

### 3.4 Ablation surface

The model constructor accepts two knobs (`src/utils/config.py:ModelConfig`):

- `disable_branches ⊆ {spectral, ssl, rawnet}` — disabled branches are not instantiated (saves memory + compute) and removed from the fusion stack. A single-branch model bypasses attention fusion entirely.
- `fusion_method ∈ {attention, concat, average}` — switches how the branch embeddings are combined. Head input dimension auto-adjusts.

These are exposed as CLI flags on `train.py` (§4.6), so ablation runs differ by flag only — no per-experiment config file.

---

## 4. Training Infrastructure

### 4.1 Trainer (`src/training/trainer.py`)

The `Trainer` class owns the model, optimizer, scheduler, and both dataloaders. Construction:

1. **Device selection** — `cuda → mps → cpu` auto-fallback.
2. **Optimizer** — `AdamW` with two parameter groups at different LRs: WavLM's trainable parameters at `1e-5`, everything else at `1e-4`. This prevents the large SSL encoder from overwhelming gradient flow through the smaller spectral and raw branches.
3. **Scheduler** — `CosineAnnealingWarmRestarts` with a 1000-step linear warmup, T_0=10 epochs, restart on epoch boundary.
4. **Dataloaders** — `build_dataloader(...)` returns a `DataLoader` wrapping `ManifestAudioDataset`. Train uses `augmentor=CodecAugmentor(...)`; val/test pass `augmentor=None` for deterministic evaluation.

### 4.2 Mixed precision (bf16 with fp16 fallback)

Four stacked techniques reduce VRAM and accelerate training:

- **bf16 autocast** (`trainer.py`): `torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)` on Ampere+ GPUs halves activation memory and doubles tensor-core throughput. Because bf16 has an 8-bit exponent (same range as fp32), no `GradScaler` is needed — this fixes an epoch-0 NaN we hit with fp16 due to overflow in the fusion attention softmax.
- **fp16 fallback** with `torch.amp.GradScaler` on pre-Ampere cards (detected via `torch.cuda.is_bf16_supported()`).
- **Selective fp32 islands** wrap two precision-sensitive layers in `autocast(enabled=False)`:
  - `SincConv.forward` (`rawnet_branch.py`): its `1e-8` zero-division guard rounds to 0 in fp16, producing NaN gradients.
  - Frozen WavLM forward (`ssl_branch.py`): attention logits can exceed fp16's 65K range on noisy inputs.
  Both cost ~0 extra VRAM because frozen submodules contribute no gradient or optimizer state.

### 4.3 Gradient accumulation

`gradient_accumulation_steps = 4` keeps effective batch 64 while forward passes use micro-batch 16. The loss is divided by `accum_steps` before backward; the optimizer steps every 4th micro-batch. This lets us fit the full model in 8 GB VRAM on a consumer GPU without sacrificing batch statistics.

### 4.4 Memory-fragmentation mitigation

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` lets PyTorch's caching allocator grow blocks instead of reserving fixed pools, reclaiming ~1.5 GB of fragmented VRAM after WavLM loads. `DataLoader(num_workers=4, pin_memory=True)` overlaps WAV decoding with GPU compute.

### 4.5 Checkpointing and resume (`trainer.py:save_checkpoint` / `load_checkpoint`)

Each checkpoint is a `torch.save` dict containing `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `epoch`, `global_step`, and `best_eer`. Two files are maintained: `checkpoint_latest.pt` (every epoch) and `checkpoint_best.pt` (only when val EER improves). Resuming from either restores exact training state — useful for Colab sessions that disconnect.

### 4.6 Ablation and cross-dataset CLI flags (`src/training/train.py`)

Composable with any YAML config; no per-experiment file needed:

```
--disable-branch {spectral, ssl, rawnet}   (repeatable)
--fusion-method {attention, concat, average}
--no-augmentation                          # disables CodecAugmentor
--sources <src1> <src2> ...                # whitelist source_datasets
--experiment-name <str>                    # isolates output directory
```

For example, reproducing the paper's single-branch baselines:

```bash
python -m src.training.train --config configs/gpu_local.yaml \
    --disable-branch ssl --disable-branch rawnet \
    --experiment-name ablation_spectral_only
```

Cross-dataset evaluation (train 2019, test 2021):

```bash
python -m src.training.train --config configs/gpu_local.yaml \
    --sources asvspoof2019 --experiment-name cross_ds_train19
python scripts/evaluate.py --config configs/gpu_local.yaml \
    --checkpoint outputs/cross_ds_train19/checkpoint_best.pt \
    --sources asvspoof2021 --label "Train 19 → Test 21"
```

### 4.7 Metrics and logging (`src/utils/metrics.py`)

Per-epoch metrics (`train_loss`, `val_loss`, `accuracy`, `EER`, `AUC-ROC`, `MAE`, learning rate, early-stopping patience counter) are appended to `outputs/{experiment}/metrics.csv`. EER is computed via `scipy.optimize.brentq` on `sklearn.metrics.roc_curve`, with a fallback to `argmin(|fpr − fnr|)` if the optimizer doesn't converge on degenerate ROC curves (`metrics.py:31-38`). Early stopping watches `val/binary/eer` with patience 7.

### 4.8 Multi-GPU and containerization

`Dockerfile` builds a CUDA-enabled image with all Python dependencies baked in. `k8s/training-job.yaml` provisions a multi-GPU pod that mounts the data volume and runs `python -m src.training.train`. This has been smoke-tested on Colab (single GPU); multi-node scaling hasn't been exercised yet because the 8-GB consumer GPU fits the full model.

---

## 5. Evaluation (`scripts/evaluate.py`)

A single command loads a checkpoint, runs inference over the test split, and emits paper-ready artifacts:

```bash
python scripts/evaluate.py \
    --config configs/gpu_local.yaml \
    --checkpoint outputs/{experiment}/checkpoint_best.pt \
    --output-dir outputs/eval/{experiment} \
    --label "Ours (full)"
```

Outputs:
1. **`metrics.json`** — overall, per-domain, and per-source EER / AUC / per-class recall.
2. **`main_row.tex`** — one LaTeX row ready to paste into the paper's Table 2: `"Ours (full) & 2.34 & 5.67 & 8.90 & 5.21 \\"`.
3. **`per_generator_rows.tex`** — paper Table 3 rows grouped by domain (per-generator breakdown).
4. **`predictions.npz`** — raw scores + logits for downstream analysis (ensembling, calibration studies).

Running `evaluate.py` on each trained ablation checkpoint produces five rows that concatenate into the main-results table; no manual number transcription between experiments and paper.

Interpretability scripts `scripts/gradcam.py` (time-frequency saliency maps on the spectral branch) and `scripts/attention_viz.py` (per-domain mean attention weights across the 3 branches) produce the paper's qualitative figures.

---

## 6. Deployment

### 6.1 Backend (`src/backend/`)

A FastAPI application hosts a single WebSocket endpoint at `/ws`. Each connection owns a `DetectionSession` with its own ring buffer (~4 seconds of recent audio), threshold, and smoothed score. The protocol is defined in `src/backend/protocol.py` (Pydantic models for `StartMessage`, `StopMessage`, `SetThresholdMessage`, `PingMessage`, and `DetectionMessage` / `AlertMessage` responses). Incoming binary frames (little-endian float32 mono PCM) are decoded with `numpy.frombuffer`, resampled to 16 kHz if needed, and pushed into the buffer. When a full hop is ready, `AsyncDetector.predict(...)` runs the PyTorch model on a background thread (keeping the WebSocket loop responsive). Score smoothing (exponential moving average) plus hysteresis on label changes (requires N consecutive windows above/below threshold before flipping) prevent the popup from flickering on transient noise.

### 6.2 Chrome extension (`extension/`)

A Manifest V3 extension with three components: a **popup** (single-click start/stop + live P(AI) display), a **background service worker** (handles chrome.tabCapture → MediaStream → offscreen doc), and an **offscreen document** (Web Audio + AudioWorklet + WebSocket client). The offscreen doc is required because MV3 service workers cannot host long-lived Web Audio graphs. Audio flows: tab → MediaStream → AudioContext → AudioWorkletNode (downmix to mono float32, pack into 4-second frames) → WebSocket → backend. Responses flow back through the service worker into popup state (`chrome.storage.session`) and optionally trigger `chrome.notifications` when a label flips.

### 6.3 Inference entry points

Three alternative deployment shapes exist beyond the WebSocket streaming path:

- **`src/inference/predictor.py`**: load checkpoint, predict on a single file. Used by tests and notebooks.
- **`src/inference/server.py`**: simple HTTP server accepting a WAV file upload, for scripted batch inference.
- **`src/inference/export_onnx.py`**: traces the model to ONNX for CPU-only deployment scenarios (e.g., a future offline mode for the extension).

---

## 7. Reproducibility

Every ingestion and preprocessing step is deterministic (MD5-hashed splits, seeded random subsampling). The full corpus can be rebuilt from raw data with three commands:

```bash
# 1. Register all sources into master_manifest.csv
for src in asvspoof2019 asvspoof2021 ljspeech esc50 musiccaps ... ; do
    python scripts/register_generated.py --dir raw/... --source $src ...
done

# 2. Preprocess all registered source files into segments
for src in asvspoof2019 asvspoof2021 ljspeech ... ; do
    python scripts/preprocess_segments.py --source $src --workers 4 ...
done

# 3. Build the unified segmented manifest
python scripts/build_segmented_manifest.py --processed-root ... --out ...
```

Training a full model from the segmented manifest is a single command:

```bash
python -m src.training.train --config configs/gpu_local.yaml
```

All outputs land in `outputs/{experiment_name}/`: checkpoints, `metrics.csv`, any evaluation results written subsequently. Configs (`configs/*.yaml`) serialize every hyperparameter; the CLI flags in §4.6 layer on top for ablation runs.

**Repository**: `github.com/foojanbabaeeian/AI-Innovation`, branch `preprocessing`. All scripts, configs, Dockerfile, Kubernetes manifests, and data manifests are version-controlled. The trained model checkpoint will be uploaded to the repo's releases page after the final training run.

---

## 8. Role & Contribution (anticipated, for Phase 4)

- **Fozhan**: Data pipeline (ingestion, registration, preprocessing, balancing, segment cap, codec augmentation wiring), training infrastructure extensions (ablation flags, `--sources`, cross-dataset), evaluation harness, GradCAM and attention-weight visualizations.
- **Camille**: Three-branch model architecture, attention fusion, GPU optimization (bf16/fp16/fp32-islands, gradient accumulation, fragmentation mitigation), trainer + scheduler + checkpointing, initial training runs, Kubernetes and Docker setup.
- **Anh**: Data curation strategy, Suno/Udio benchmark integration, related-work survey, paper writing (Introduction, Related Work, Methods), results tables.
- **Sofia**: Chrome extension (Manifest V3, popup UI, offscreen audio worklet, WebSocket client), deployment UX, integration testing with the backend.

---

## Code Inventory

| Path | Purpose |
|---|---|
| `src/models/` | Model code: `spectral_branch.py`, `ssl_branch.py`, `rawnet_branch.py`, `fusion_model.py` |
| `src/data/dataset.py` | `ManifestAudioDataset`, `build_dataloader` (with optional `augmentor`, `sources`) |
| `src/data/augmentation.py` | `CodecAugmentor` (MP3/AAC/Opus/µ-law) |
| `src/training/trainer.py` | `Trainer` class (train loop, mixed precision, checkpointing) |
| `src/training/train.py` | CLI entry point with ablation flags |
| `src/utils/config.py` | Dataclass-based YAML config loader |
| `src/utils/metrics.py` | EER, AUC-ROC, per-class metrics |
| `src/backend/` | FastAPI + WebSocket server (DetectionSession, AsyncDetector) |
| `src/inference/` | Offline predictor, HTTP server, ONNX export |
| `scripts/register_generated.py` | Append source files to master manifest |
| `scripts/preprocess_segments.py` | Resample + normalize + segment → Processed/ |
| `scripts/build_segmented_manifest.py` | Scan Processed/ → segmented manifest |
| `scripts/balance_manifest.py` | ASVspoof per-attack subsample |
| `scripts/rebalance_splits.py` | Unfreeze ASVspoof split, stratify 70/15/15 |
| `scripts/evaluate.py` | Per-domain + per-source EER + LaTeX table output |
| `scripts/gradcam.py` | Time-frequency saliency maps (paper §7.2) |
| `scripts/attention_viz.py` | Per-domain attention weights (paper §7.1) |
| `scripts/generate_musicgen.py` | MusicGen synthetic music generation (Colab) |
| `scripts/generate_audiogen.py` | AudioLDM2 synthetic SFX generation (Colab) |
| `configs/default.yaml` + `configs/gpu_local.yaml` | Training configs |
| `Dockerfile` + `k8s/*.yaml` | Containerization + multi-GPU pod definitions |

~2,300 lines of Python across `src/`; ~1,000 lines across `scripts/`.
