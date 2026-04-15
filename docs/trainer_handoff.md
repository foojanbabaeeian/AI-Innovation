# Trainer Handoff — AI Audio Detector

**Project:** CECS 458 AI-Powered Business Innovation  
**Team:** Fozhan Babaeiyan Ghamsari, Anh Le, Sofia Tejada Sarria, Camille Wong  
**Prepared by:** Fozhan Babaeiyan Ghamsari  
**Data location:** `C:\Users\fooja\Google Drive Streaming\My Drive\AI-Innovation-Data`  
**Repo:** `C:\Users\fooja\Documents\GitHub\AI-Innovation` (branch: `AI_sounds`)

---

## 1. What We Are Building

A **multi-branch deep learning classifier** that detects AI-generated audio (voice, music, and environmental sounds) for use in a browser extension. Three parallel branches fuse their representations via attention:

| Branch | Input | Detects |
|--------|-------|---------|
| Branch 1 — Spectral CNN | Mel-spectrograms, MFCCs, CQT | Vocoder artifacts, phase inconsistencies |
| Branch 2 — SSL (wav2vec 2.0 / WavLM) | Pretrained transformer embeddings | Prosodic irregularities |
| Branch 3 — RawNet2/SincNet | Raw waveform | Phase artifacts, codec patterns |

Output: binary classification `0 = real`, `1 = AI-generated`.

---

## 2. Dataset Status (as of April 2026)

### What Is on Drive and Registered in the Manifest

| Dataset | Count | Label | Split |
|---------|-------|-------|-------|
| ASVspoof 2019 LA | 121,461 | real + fake voice | official train/val/test |
| LJSpeech | 13,100 | real voice | 70/15/15 |
| ASVspoof 2021 (Anh's clips) | 5,535 | real + fake voice | 70/15/15 |
| ESC-50 | 2,000 | real non-human | 70/15/15 |
| MusicCaps | 931 | real music | 70/15/15 |
| AudioGen | 174 | fake non-human | 70/15/15 |
| **TOTAL** | **143,201** | | |

**Manifest:** `data/metadata/master_manifest.csv` (in the Git repo, committed)

### Split and Label Distribution

```
dataset                   total     train       val      test
asvspoof2019             121461    25380 (21%)  24844 (20%)  71237 (59%)  ← official protocol
asvspoof2021               5535     3875 (70%)    856 (15%)    804 (15%)
audiogen                    174      123 (71%)     26 (15%)     25 (14%)
esc50                      2000     1400 (70%)    297 (15%)    303 (15%)
ljspeech                  13100     9193 (70%)   1945 (15%)   1962 (15%)
musiccaps                   931      644 (69%)    150 (16%)    137 (15%)
TOTAL                    143201    40615 (28%)  28118 (20%)  74468 (52%)

Label balance per split:
  train:  35% real  /  65% fake   ← use class_weight or WeightedRandomSampler
  val:    18% real  /  82% fake
  test:   13% real  /  87% fake
```

**Note on ASVspoof 2019 skew:** The official eval set (71K files, 59% of all data)
is mapped to `test` per the ASVspoof benchmark convention. The train/val proportions
look low (28/20%) because of this — it is correct and intentional. Use the
`max_samples` parameter during early training runs to avoid long epochs.

### What Is Still Missing

These datasets were in the feasibility study but are NOT yet on Drive:

| Dataset | Size | How to Get | Priority |
|---------|------|-----------|----------|
| SpeechT5 generated clips | 1,000 clips | `python scripts/fetch_dataset.py speecht5` (GPU, ~45 min) | **HIGH** |
| AudioGen remaining clips | 26 clips | `python scripts/fetch_dataset.py audiogen` (GPU, ~5 min) | **HIGH** |
| WaveFake | ~117K clips | Download from https://github.com/RUB-SysSec/WaveFake | Medium |
| ASVspoof 5 | ~100K+ | Download from https://www.asvspoof.org/ (requires registration) | Medium |
| ASVspoof 2021 LA full set | ~190K | Download from https://www.asvspoof.org/ | Low (we have 5K already) |
| Suno AI music | 500–1K clips | Self-collect from suno.com, save to `raw/music/fake/suno/` | Medium |
| Udio AI music | 500–1K clips | Self-collect from udio.com, save to `raw/music/fake/udio/` | Medium |
| FakeAVCeleb | ~20K | Download from https://github.com/ahaliassos/FakeAVCeleb | Low |

After downloading, register new datasets in the manifest using the appropriate script (or `fetch_dataset.py`).

---

## 3. Where the Data Lives on Google Drive

```
AI-Innovation-Data/
├── raw/
│   ├── voice/
│   │   ├── real/LJSpeech-1.1/wavs/         ← 13,100 .wav
│   │   ├── LA/                              ← ASVspoof 2019 LA (122K .flac)
│   │   │   ├── ASVspoof2019_LA_train/flac/
│   │   │   ├── ASVspoof2019_LA_dev/flac/
│   │   │   └── ASVspoof2019_LA_eval/flac/
│   │   └── fake/speecht5/                   ← EMPTY — needs generation
│   ├── music/
│   │   ├── real/musiccaps/                  ← 931 .wav
│   │   └── fake/                            ← EMPTY — needs Suno/Udio
│   └── non_human/
│       ├── real/esc50_processed/            ← 2,000 .wav
│       └── fake/audiogen/                   ← 174/200 .wav
│
├── Processed/Voice/Anh-ASVspoof 2021 voice clips/AI voice/
│   └── *.wav                                ← 5,535 clips (real + fake voice)
│       (NOTE: despite the folder name, this also contains real speech;
│        the manifest correctly labels each file as real=0.0 or fake=1.0)
│
└── processed/                               ← TO BE CREATED by preprocess_segments.py
    ├── voice/real/ and fake/
    ├── music/real/ and fake/
    └── non_human/real/ and fake/
```

**Important:** The manifest `file_path` column stores **absolute Windows paths**. When
training on a different machine or on Colab, you must remap these paths. See Step 4.

---

## 4. Preprocessing (Must Do Before Training)

The raw files need to be segmented into 4-second windows before training. This
is handled by `scripts/preprocess_segments.py`. Running it on Colab is recommended
because it keeps all data on Google Drive without filling local disk.

### Option A — Colab (Recommended)

Open `notebooks/train_colab.ipynb` or create a new Colab notebook:

```python
# ── Cell 1: Mount Drive and clone repo ──────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/YOUR_ORG/AI-Innovation.git /content/AI-Innovation
%cd /content/AI-Innovation
!git checkout AI_sounds

# ── Cell 2: Install dependencies ────────────────────────────────────────────
!pip install -q torchaudio soundfile pyloudnorm pandas

# ── Cell 3: Remap absolute paths in manifest to Colab Drive paths ───────────
import csv, re
from pathlib import Path

MANIFEST_IN  = 'data/metadata/master_manifest.csv'
MANIFEST_OUT = 'data/metadata/master_manifest_colab.csv'

# The Windows paths in the manifest start with this prefix:
WINDOWS_PREFIX = 'C:/Users/fooja/Google Drive Streaming/My Drive/AI-Innovation-Data'
COLAB_PREFIX   = '/content/drive/MyDrive/AI-Innovation-Data'

with open(MANIFEST_IN, newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys()) if rows else []

for row in rows:
    row['file_path'] = row['file_path'].replace(WINDOWS_PREFIX, COLAB_PREFIX)

with open(MANIFEST_OUT, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f'Remapped {len(rows)} paths → {MANIFEST_OUT}')

# ── Cell 4: Smoke test preprocessing (5 files) ──────────────────────────────
DATA_ROOT = '/content/drive/MyDrive/AI-Innovation-Data'
!python scripts/preprocess_segments.py \
    --manifest data/metadata/master_manifest_colab.csv \
    --processed-root {DATA_ROOT}/processed \
    --dry-run \
    --limit 10

# ── Cell 5: Run full preprocessing (3–6 hours on A100) ──────────────────────
# Run each domain separately so you can resume if Colab disconnects:

# Voice only (largest batch):
!python scripts/preprocess_segments.py \
    --manifest data/metadata/master_manifest_colab.csv \
    --processed-root {DATA_ROOT}/processed \
    --source ljspeech \
    --workers 4

!python scripts/preprocess_segments.py \
    --manifest data/metadata/master_manifest_colab.csv \
    --processed-root {DATA_ROOT}/processed \
    --source asvspoof2019 \
    --workers 4

!python scripts/preprocess_segments.py \
    --manifest data/metadata/master_manifest_colab.csv \
    --processed-root {DATA_ROOT}/processed \
    --source asvspoof2021 \
    --workers 4

# Music and non-human:
!python scripts/preprocess_segments.py \
    --manifest data/metadata/master_manifest_colab.csv \
    --processed-root {DATA_ROOT}/processed \
    --source musiccaps \
    --workers 4

!python scripts/preprocess_segments.py \
    --manifest data/metadata/master_manifest_colab.csv \
    --processed-root {DATA_ROOT}/processed \
    --source esc50 \
    --workers 4

!python scripts/preprocess_segments.py \
    --manifest data/metadata/master_manifest_colab.csv \
    --processed-root {DATA_ROOT}/processed \
    --source audiogen \
    --workers 4

# After all sources: create a combined segmented manifest
!python scripts/preprocess_segments.py \
    --manifest data/metadata/master_manifest_colab.csv \
    --processed-root {DATA_ROOT}/processed \
    --out-manifest data/metadata/master_manifest_segmented.csv \
    --workers 4
```

### Option B — Local (in tf-gpu-210 conda env)

```bash
# Activate environment
conda activate tf-gpu-210
cd C:/Users/fooja/Documents/GitHub/AI-Innovation

# Run per-source (resumable — skips existing output files)
python scripts/preprocess_segments.py \
    --manifest data/metadata/master_manifest.csv \
    --processed-root "C:/Users/fooja/Google Drive Streaming/My Drive/AI-Innovation-Data/processed" \
    --source audiogen \
    --workers 2

# etc. for each source_dataset
```

---

## 5. Completing Missing Data (Before Full Training)

### Generate SpeechT5 fake voice (needs GPU, ~45 min)

```bash
conda activate tf-gpu-210
cd C:/Users/fooja/Documents/GitHub/AI-Innovation
python scripts/fetch_dataset.py speecht5
```

This generates 1,000 AI voice clips to `raw/voice/fake/speecht5/` and
automatically registers them in the manifest. No further steps needed.

### Finish AudioGen (26 clips remaining)

```bash
conda activate tf-gpu-210
cd C:/Users/fooja/Documents/GitHub/AI-Innovation
python scripts/fetch_dataset.py audiogen
```

Picks up where it left off (174/200). Adds new entries to the manifest.

### After generation: rebalance splits

```bash
python scripts/rebalance_splits.py --apply
```

This redistributes newly added SpeechT5/AudioGen clips to 70/15/15.

---

## 6. Training

### Prerequisites

1. Conda env `tf-gpu-210` with torch 2.5.1+cu121, torchaudio, transformers, datasets
2. Google Drive mounted (or all data downloaded locally)
3. `data/metadata/master_manifest.csv` present (committed in Git)
4. Optional but recommended: run preprocessing first (Section 4)

### Config files

| Config | Purpose |
|--------|---------|
| `configs/default.yaml` | Base training config |
| `configs/gpu_local.yaml` | Local GPU overrides (batch size, paths) |
| `configs/k8s_multi_gpu.yaml` | Multi-GPU Kubernetes config |

Key fields to review in `configs/gpu_local.yaml`:
- `data.manifest_path` — update to your manifest location
- `data.data_root` — set to empty string `""` if manifest uses absolute paths, otherwise set to Drive root
- `training.batch_size` — reduce to 16 or 8 if OOM
- `training.max_samples` — set to 1000 for a smoke test run

### Run training

```bash
conda activate tf-gpu-210
cd C:/Users/fooja/Documents/GitHub/AI-Innovation

# Smoke test (1000 samples, 2 epochs)
python src/training/train.py --config configs/gpu_local.yaml \
    --override training.max_samples=1000 training.epochs=2

# Full training
python src/training/train.py --config configs/gpu_local.yaml
```

### On Colab

Use `notebooks/train_colab.ipynb`. Set runtime to A100 GPU (Colab Pro recommended
for 143K+ samples). Training checkpoints are saved to
`AI-Innovation-Data/checkpoints/` on Drive automatically.

### Handling class imbalance

The training set is ~35% real / 65% fake (due to ASVspoof 2019). Use one of:

**Option 1 — Weighted loss (in `configs/default.yaml`):**
```yaml
training:
  class_weights: [1.86, 1.0]   # [weight_real, weight_fake] = (65/35, 1)
```

**Option 2 — WeightedRandomSampler** (in `src/training/trainer.py`): set
`use_weighted_sampler: true` in config to oversample real examples.

---

## 7. Model Architecture Quick Reference

```
src/models/
├── fusion_model.py      ← Main entry point: FusionModel(config)
├── spectral_branch.py   ← Branch 1: ResNet-34 on mel-specs
├── ssl_branch.py        ← Branch 2: wav2vec 2.0 / WavLM fine-tuning
└── rawnet_branch.py     ← Branch 3: RawNet2/SincNet on raw waveform

src/training/
├── train.py             ← CLI entry point
└── trainer.py           ← Training loop, logging, checkpointing

src/data/
├── dataset.py           ← ManifestAudioDataset + build_dataloader
├── preprocessing.py     ← AudioPreprocessor (used at runtime)
└── augmentation.py      ← On-the-fly augmentations
```

---

## 8. Expected Results / Baselines

Per the feasibility study, target metrics to beat (published baselines on ASVspoof 2019 LA eval):

| Model | EER (lower=better) | Notes |
|-------|-------------------|-------|
| LCNN baseline | ~5.1% | Standard single-branch baseline |
| RawNet2 | ~4.0% | Our Branch 3 baseline |
| wav2vec 2.0 fine-tuned | ~1.8% | Our Branch 2 baseline |
| **Ours (3-branch fusion)** | **<2%** | Target |

Report Equal Error Rate (EER) and min-tDCF on the ASVspoof 2019 LA eval set.
The test split in our manifest IS the ASVspoof 2019 LA eval set — keep it untouched
until final reporting.

---

## 9. Scripts Reference

| Script | What it does | Command |
|--------|-------------|---------|
| `fetch_dataset.py` | Download/generate datasets | `python scripts/fetch_dataset.py [ljspeech\|speecht5\|audiogen\|esc50\|status]` |
| `register_asvspoof2019.py` | Register ASVspoof 2019 in manifest | `python scripts/register_asvspoof2019.py --root <path>` |
| `rebalance_splits.py` | Fix train/val/test distribution | `python scripts/rebalance_splits.py --apply` |
| `preprocess_segments.py` | Segment raw audio → processed/ | `python scripts/preprocess_segments.py --processed-root <path>` |
| `fetch_musiccaps.py` | Download MusicCaps audio | `python scripts/fetch_musiccaps.py` |
| `merge_anh_metadata.py` | Register ASVspoof 2021 clips | `python scripts/merge_anh_metadata.py` |

---

## 10. Questions / Contact

Contact Fozhan Babaeeian Ghamsari for questions about the data pipeline or manifest.
The full data documentation is in `docs/data_format_specification.md`.
