# Data Format Specification for AI Audio Detection Model

## Overview

This document defines the data format, directory structure, labeling schema, and preprocessing pipeline required to train the multi-branch deep learning model for detecting AI-generated audio. The system must handle three distinct audio domains: **voice**, **music**, and **non-human sounds** (e.g., sound effects, environmental audio).

### Deployment Context

The trained model will be deployed as a **browser extension** that captures audio playing on the user's current page in real time (via the Web Audio API / `AudioContext`) and classifies it as real or AI-generated. Users receive an in-browser notification when AI-generated audio is detected. This real-time inference constraint has direct implications for the data format:

- **Training data must reflect browser-quality audio.** Audio captured through a browser undergoes codec compression (AAC, Opus, MP3), resampling, and potential quality degradation from streaming. The training pipeline must simulate these conditions.
- **Segment length must support low-latency inference.** The model must produce a classification within 1-3 seconds of audio playback, so training segments should match the inference window (3-5 seconds).
- **Model must be lightweight enough for client-side or edge inference.** While full multi-branch training uses all three branches, the deployed model may use a distilled or pruned version. Training data format should support both full and distilled pipelines.

### Publication Context

This work targets a peer-reviewed research publication. All data handling follows reproducibility standards: deterministic splits, versioned manifests, documented preprocessing, and publicly available benchmark datasets with clear provenance.

---

## 1. Raw Data Ingestion Format

### 1.1 Accepted Audio Formats

| Format | Extension | Use Case |
|--------|-----------|----------|
| WAV (PCM) | `.wav` | Preferred lossless format for all training data |
| FLAC | `.flac` | Acceptable lossless alternative |
| MP3 | `.mp3` | Accepted for in-the-wild samples; will be decoded to WAV |
| AAC/M4A | `.aac`, `.m4a` | Accepted for in-the-wild samples |
| OGG/Opus | `.ogg`, `.opus` | Accepted for in-the-wild samples |

All compressed formats are decoded to WAV during preprocessing. The original compressed files are retained separately for codec-robustness augmentation.

### 1.2 Audio Requirements

| Parameter | Voice | Music | Non-Human Sound |
|-----------|-------|-------|-----------------|
| Sample rate | 16 kHz | 44.1 kHz | 44.1 kHz |
| Bit depth | 16-bit PCM | 16-bit PCM | 16-bit PCM |
| Channels | Mono | Mono (downmixed) | Mono (downmixed) |
| Loudness normalization | EBU R128 (-23 LUFS) | EBU R128 (-23 LUFS) | EBU R128 (-23 LUFS) |
| Min duration (raw clip) | 1 second | 3 seconds | 1 second |
| Max duration (raw clip) | 300 seconds | 300 seconds | 300 seconds |

**Note on dual sample rates:** Voice data uses 16 kHz to match pretrained SSL models (wav2vec 2.0, WavLM). Music and non-human audio use 44.1 kHz to preserve high-frequency content critical for detecting vocoder artifacts. Branch 1 (spectral) and Branch 3 (raw waveform) will handle resampling internally per domain.

---

## 2. Directory Structure

```
data/
├── raw/                          # Original unprocessed files
│   ├── voice/
│   │   ├── real/
│   │   │   ├── asvspoof2019/
│   │   │   ├── asvspoof2021/
│   │   │   └── asvspoof5/
│   │   └── fake/
│   │       ├── asvspoof2019/
│   │       ├── asvspoof2021/
│   │       ├── asvspoof5/
│   │       ├── wavefake/
│   │       └── fakeavceleb/
│   ├── music/
│   │   ├── real/
│   │   │   └── musiccaps/
│   │   └── fake/
│   │       ├── suno/
│   │       └── udio/
│   └── non_human/
│       ├── real/
│       │   └── audioset/
│       └── fake/
│           ├── elevenlabs_sfx/
│           └── bark_sfx/
│
├── processed/                    # Preprocessed, normalized, segmented
│   ├── voice/
│   │   ├── real/
│   │   └── fake/
│   ├── music/
│   │   ├── real/
│   │   └── fake/
│   └── non_human/
│       ├── real/
│       └── fake/
│
├── features/                     # Extracted features (per branch)
│   ├── spectrograms/             # Branch 1: Mel-specs, MFCCs, CQT
│   ├── ssl_embeddings/           # Branch 2: wav2vec 2.0 / WavLM
│   └── raw_waveforms/            # Branch 3: Preprocessed segments
│
├── metadata/
│   ├── master_manifest.csv       # Single source of truth for all samples
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
│
└── augmented/                    # Augmented copies (generated on-the-fly or cached)
```

---

## 3. Labeling Schema

### 3.1 Master Manifest (`master_manifest.csv`)

Every audio sample gets one row in this CSV. This is the single source of truth that the data loader reads.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `sample_id` | string | Unique identifier | `voice_real_asvspoof2019_LA_E_0001` |
| `file_path` | string | Relative path from `data/processed/` | `voice/real/asvspoof2019/LA_E_0001.wav` |
| `label` | int | Binary class: `0` = real, `1` = AI-generated | `0` |
| `domain` | string | Audio domain | `voice`, `music`, `non_human` |
| `source_dataset` | string | Origin dataset name | `asvspoof2019` |
| `generator` | string | AI model that produced it (if fake) | `tacotron2`, `wavenet`, `suno_v3`, `N/A` |
| `attack_type` | string | ASVspoof attack ID or generation method | `A01`, `A17`, `tts`, `vc`, `N/A` |
| `duration_sec` | float | Duration in seconds after preprocessing | `4.2` |
| `sample_rate` | int | Sample rate in Hz | `16000` |
| `split` | string | Dataset partition | `train`, `val`, `test` |
| `num_segments` | int | Number of windowed segments produced | `3` |
| `language` | string | ISO 639-1 code (if applicable) | `en`, `zh`, `N/A` |
| `codec_original` | string | Original codec before decoding | `pcm`, `mp3`, `aac` |
| `notes` | string | Free-text field for edge cases | `background noise present` |

### 3.2 Label Definitions

```
0 = Real (bonafide)
    - Recorded by a human with a physical microphone
    - No AI synthesis, voice conversion, or AI editing applied
    - May contain standard production effects (EQ, compression, reverb)

1 = Fake (AI-generated or AI-manipulated)
    - Fully synthesized by TTS, voice conversion, or neural vocoder
    - AI-generated music (Suno, Udio, MusicGen, etc.)
    - AI-generated sound effects (ElevenLabs SFX, Bark, AudioLDM, etc.)
    - Partially AI-edited audio where AI tools modified the acoustic content
      (e.g., AI vocal replacement in an otherwise real track)
```

### 3.3 Domain-Specific Label Notes

**Voice:** Follow ASVspoof protocol labels where available. For ASVspoof 2019 LA, attack types A01-A06 are training, A07-A19 are evaluation. Maintain this separation to test generalization to unseen attacks.

**Music:** "Real" = human-performed recordings from MusicCaps. "Fake" = AI-generated tracks from Suno/Udio. Tracks that mix real instrumentals with AI vocals should be labeled `1` (fake) with a note in the `notes` field.

**Non-Human Sound:** "Real" = AudioSet clips verified as natural recordings. "Fake" = AI-generated sound effects. Exclude ambiguous clips where generation method is unknown.

---

## 4. Segmentation (Windowing)

Raw clips are segmented into fixed-length windows for training. Each window becomes one training sample.

| Parameter | Value |
|-----------|-------|
| Window length | 4 seconds |
| Hop size (overlap) | 2 seconds (50% overlap) |
| Minimum usable segment | 2 seconds (zero-pad shorter tails) |
| Discard threshold | Segments < 1 second are discarded |

**Segment naming convention:**
```
{sample_id}_seg{NNN}.wav

Example: voice_real_asvspoof2019_LA_E_0001_seg000.wav
         voice_real_asvspoof2019_LA_E_0001_seg001.wav
         voice_real_asvspoof2019_LA_E_0001_seg002.wav
```

**Important:** All segments from a single source clip must be in the **same split** (train/val/test) to prevent data leakage.

---

## 5. Feature Extraction Format (Per Branch)

### 5.1 Branch 1 -- Spectral Features (CNN/ResNet)

Stored as `.npy` files (NumPy arrays) or `.pt` files (PyTorch tensors).

| Feature | Shape | Parameters |
|---------|-------|------------|
| Mel-spectrogram | `(n_mels, time_frames)` = `(128, T)` | n_fft=2048, hop=512, n_mels=128 |
| MFCC | `(n_mfcc, time_frames)` = `(40, T)` | 40 coefficients + delta + delta-delta |
| CQT | `(n_bins, time_frames)` = `(84, T)` | 84 bins, 7 octaves, 12 bins/octave |

All spectral features are computed in log scale (log-mel, log-CQT). Time frames `T` depends on segment duration and hop size.

**File naming:**
```
features/spectrograms/{sample_id}_seg{NNN}_mel.npy
features/spectrograms/{sample_id}_seg{NNN}_mfcc.npy
features/spectrograms/{sample_id}_seg{NNN}_cqt.npy
```

### 5.2 Branch 2 -- SSL Embeddings (wav2vec 2.0 / WavLM)

Stored as `.pt` files (PyTorch tensors).

| Feature | Shape | Source Model |
|---------|-------|-------------|
| wav2vec 2.0 embedding | `(time_frames, 768)` | `facebook/wav2vec2-base-960h` |
| WavLM embedding | `(time_frames, 768)` | `microsoft/wavlm-base-plus` |

Embeddings are extracted from the last hidden layer. For fine-tuning, raw audio is fed directly to the model during training (no pre-extraction needed).

**For pre-extraction (frozen encoder):**
```
features/ssl_embeddings/{sample_id}_seg{NNN}_wav2vec2.pt
features/ssl_embeddings/{sample_id}_seg{NNN}_wavlm.pt
```

### 5.3 Branch 3 -- Raw Waveform (RawNet2 / SincNet)

No feature extraction needed. The processed `.wav` segments in `data/processed/` are used directly. The model's learnable sinc filters operate on raw samples.

**Expected tensor shape at model input:** `(1, num_samples)`
- Voice (16 kHz, 4 sec): `(1, 64000)`
- Music/Non-human (44.1 kHz, 4 sec): `(1, 176400)`

---

## 6. Data Splits

Follow a stratified split ensuring balanced representation across domains, generators, and attack types.

| Split | Percentage | Purpose |
|-------|-----------|---------|
| Train | 70% | Model training |
| Validation | 15% | Hyperparameter tuning, early stopping |
| Test | 15% | Final evaluation (untouched until reporting) |

### Split Constraints

1. **No leakage:** All segments from a single source clip go into the same split.
2. **Speaker disjoint (voice):** No speaker appears in both train and test. Follow ASVspoof official speaker partitions where available.
3. **Generator holdout:** For generalization testing, consider holding out at least one AI generator entirely from training (e.g., train on A01-A06, test on A07-A19 per ASVspoof protocol).
4. **Domain balance:** Each split should contain proportional representation of voice, music, and non-human samples.

---

## 7. Augmentation Pipeline

Augmentations are applied on-the-fly during training (preferred) or pre-computed and cached in `data/augmented/`.

| Augmentation | Parameters | Purpose |
|--------------|-----------|---------|
| Room Impulse Response (RIR) | Convolution with real/simulated RIRs | Simulate reverberant environments |
| Codec simulation | MP3 (64-320 kbps), AAC (64-256 kbps), Opus (32-128 kbps) | Robustness to compression |
| Additive noise | SNR 5-30 dB; noise types: white, babble, ambient | Real-world noise conditions |
| Telephone band filter | 300-3400 Hz bandpass | Simulate phone call conditions |
| Time stretch | 0.9x - 1.1x | Tempo variation robustness |
| Gain variation | +/- 6 dB | Volume robustness |
| Resampling | Downsample then upsample (e.g., 16k -> 8k -> 16k) | Simulate quality degradation |

**Augmentation metadata** should be logged so that augmented samples can be traced back to their source.

---

## 8. PyTorch Dataset Example

Below is a reference for how the data loader should consume this format:

```python
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset

class AIAudioDetectionDataset(Dataset):
    """
    Reads segments from processed/ directory using the manifest CSV.
    Returns raw waveform + metadata; feature extraction happens in the model
    or in a collate function.
    """

    def __init__(self, manifest_path: str, data_root: str, split: str = "train",
                 target_sr: int = 16000, segment_length: int = 64000,
                 augment: bool = False):
        self.df = pd.read_csv(manifest_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.data_root = data_root
        self.target_sr = target_sr
        self.segment_length = segment_length
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = f"{self.data_root}/processed/{row['file_path']}"

        waveform, sr = torchaudio.load(filepath)

        # Resample if needed
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)

        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Pad or trim to fixed length
        if waveform.shape[1] < self.segment_length:
            pad = self.segment_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        else:
            waveform = waveform[:, :self.segment_length]

        label = torch.tensor(row["label"], dtype=torch.long)

        return {
            "waveform": waveform,        # (1, segment_length)
            "label": label,              # 0 or 1
            "domain": row["domain"],
            "sample_id": row["sample_id"],
        }
```

---

## 9. Estimated Storage Requirements

| Component | Estimated Size |
|-----------|---------------|
| Raw voice datasets (ASVspoof + WaveFake + FakeAVCeleb) | ~30 GB |
| Raw music (MusicCaps + self-collected AI music) | ~5 GB |
| Raw non-human (AudioSet subset + AI SFX) | ~15 GB |
| Processed segments (all domains) | ~40 GB |
| Spectral features (Mel + MFCC + CQT, .npy) | ~20 GB |
| SSL embeddings (pre-extracted, .pt) | ~15 GB |
| Augmented cache (if pre-computed) | ~30 GB |
| **Total (with augmentation cache)** | **~155 GB** |
| **Total (on-the-fly augmentation)** | **~125 GB** |

---

## 10. Browser Extension Inference Data Format

This section specifies the data format at inference time when the browser extension captures live audio.

### 10.1 Audio Capture Pipeline (Browser Side)

```
Browser Tab Audio
    │
    ├─► chrome.tabCapture / AudioContext.createMediaStreamSource()
    │
    ├─► Web Audio API: AnalyserNode or ScriptProcessorNode / AudioWorklet
    │       - Sample rate: 44100 Hz (browser default) or 48000 Hz
    │       - Channels: Stereo → downmix to mono
    │       - Buffer size: 4096 or 8192 samples per callback
    │
    ├─► Sliding window buffer (ring buffer)
    │       - Accumulate 3-5 seconds of audio
    │       - Overlap: 50% (new inference every 1.5-2.5 seconds)
    │
    ├─► Preprocessing (in JS or WASM)
    │       - Resample to 16 kHz (for model input)
    │       - Normalize amplitude to [-1.0, 1.0]
    │       - Convert Float32Array → model input tensor
    │
    └─► Inference
            - Option A: ONNX Runtime Web (client-side, ~100-300ms)
            - Option B: WebSocket to backend API (~200-500ms + network)
            - Output: { score: 0.0-1.0, label: "real"|"ai", confidence: float }
```

### 10.2 Inference Input Tensor Format

| Parameter | Value |
|-----------|-------|
| Shape | `(1, 1, 48000)` for 3s @ 16kHz or `(1, 1, 80000)` for 5s @ 16kHz |
| Dtype | `float32` |
| Range | `[-1.0, 1.0]` (peak-normalized) |
| Format | ONNX tensor (client-side) or raw PCM float32 (API) |

### 10.3 Training Data Augmentation for Browser Conditions

To ensure the model generalizes to browser-captured audio, training data must include augmentations that simulate the browser audio pipeline:

| Condition | Simulation |
|-----------|-----------|
| YouTube streaming codec | AAC-LC @ 128-256 kbps, Opus @ 48-160 kbps |
| Browser resampling | 44.1kHz/48kHz → 16kHz via `torchaudio.transforms.Resample` |
| Tab capture artifacts | Double encoding simulation (source codec → PCM → model) |
| Variable bitrate streaming | VBR MP3/AAC with bitrate drops (simulate buffering) |
| Page background audio mixing | Mix target audio with UI sounds at high SNR (30-50 dB) |

### 10.4 Confidence Threshold and Notification Logic

The extension uses a configurable confidence threshold (default: 0.7) before triggering a notification. Training evaluation should report metrics at multiple thresholds (0.5, 0.6, 0.7, 0.8, 0.9) to help select the optimal operating point that balances false positive rate against detection sensitivity.

---

## 11. Data Collection Checklist

- [ ] Download ASVspoof 2019 LA from [official site](https://www.asvspoof.org/)
- [ ] Download ASVspoof 2021 LA
- [ ] Download ASVspoof 5
- [ ] Download WaveFake from [GitHub](https://github.com/RUB-SysSec/WaveFake)
- [ ] Download FakeAVCeleb
- [ ] Download MusicCaps audio (via YouTube)
- [ ] Download AudioSet subset (via YouTube, select relevant ontology classes)
- [ ] Generate AI music samples from Suno (target: 500+ clips)
- [ ] Generate AI music samples from Udio (target: 500+ clips)
- [ ] Generate AI SFX from ElevenLabs
- [ ] Generate AI SFX from Bark
- [ ] Build `master_manifest.csv` with all metadata columns populated
- [ ] Run preprocessing pipeline (normalize, resample, segment)
- [ ] Validate no data leakage across splits
- [ ] Verify class balance per split and domain
