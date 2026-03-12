"""
Smoke tests for the data loading and preprocessing pipeline.

These tests use synthetic audio data (random tensors saved as WAV files)
so they can run locally without any real datasets downloaded. They verify
that every component of the pipeline produces correctly shaped outputs.

Run with: pytest tests/test_data_pipeline.py -v
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from src.config import AudioConfig, Config, MelConfig, load_config
from src.data.augmentation import CodecAugmentor
from src.data.preprocessing import AudioPreprocessor
from src.evaluation.metrics import compute_all_metrics, compute_eer, compute_min_dcf


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def audio_config():
    """Standard audio config for testing."""
    return AudioConfig(
        sample_rate=16000,
        duration_sec=4.0,
        n_samples=64000,
        mel=MelConfig(
            n_mels=128,
            n_fft=1024,
            hop_length=256,
            f_min=0,
            f_max=8000,
            power=2.0,
            normalize=True,
        ),
    )


@pytest.fixture
def preprocessor(audio_config):
    """AudioPreprocessor instance."""
    return AudioPreprocessor(audio_config)


@pytest.fixture
def synthetic_wav(tmp_path):
    """Create a synthetic WAV file for testing.

    Returns path to a 4-second mono WAV file at 16kHz.
    """
    sr = 16000
    duration = 4.0
    n_samples = int(sr * duration)

    # Generate a simple tone (440Hz) + noise
    t = torch.linspace(0, duration, n_samples)
    tone = 0.5 * torch.sin(2 * np.pi * 440 * t)
    noise = 0.1 * torch.randn(n_samples)
    waveform = (tone + noise).numpy()

    path = tmp_path / "test_audio.wav"
    sf.write(str(path), waveform, sr)
    return str(path)


@pytest.fixture
def synthetic_stereo_wav(tmp_path):
    """Create a stereo WAV file at a non-standard sample rate."""
    sr = 44100
    duration = 2.0
    n_samples = int(sr * duration)

    waveform = 0.3 * torch.randn(2, n_samples).numpy().T  # (N, 2) for soundfile
    path = tmp_path / "test_stereo.wav"
    sf.write(str(path), waveform, sr)
    return str(path)


@pytest.fixture
def short_wav(tmp_path):
    """Create a WAV file shorter than 4 seconds (needs padding)."""
    sr = 16000
    duration = 1.0
    n_samples = int(sr * duration)

    waveform = 0.3 * torch.randn(n_samples).numpy()
    path = tmp_path / "test_short.wav"
    sf.write(str(path), waveform, sr)
    return str(path)


@pytest.fixture
def fake_asvspoof_dataset(tmp_path):
    """Create a minimal fake ASVspoof-like dataset for testing.

    Creates protocol file + audio files so the Dataset class can be tested
    end-to-end without the real dataset.
    """
    root = tmp_path / "ASVspoof2019_LA"
    proto_dir = root / "ASVspoof2019_LA_cm_protocols"
    audio_dir = root / "ASVspoof2019_LA_train" / "flac"
    proto_dir.mkdir(parents=True)
    audio_dir.mkdir(parents=True)

    sr = 16000
    protocol_lines = []

    for i in range(10):
        utt_id = f"LA_T_{i:07d}"
        speaker = f"LA_{i:04d}"
        label = "bonafide" if i < 5 else "spoof"
        system = "-" if i < 5 else f"A{i:02d}"
        protocol_lines.append(f"{speaker} {utt_id} - {system} {label}")

        # Create a small FLAC file
        waveform = 0.3 * torch.randn(1, sr * 4)
        sf.write(str(audio_dir / f"{utt_id}.flac"), waveform.squeeze(0).numpy(), sr)

    proto_file = proto_dir / "ASVspoof2019.LA.cm.train.trn.txt"
    proto_file.write_text("\n".join(protocol_lines) + "\n")

    return str(root)


@pytest.fixture
def fake_wavefake_dataset(tmp_path):
    """Create a minimal fake WaveFake-like dataset for testing."""
    root = tmp_path / "WaveFake"
    real_dir = root / "real"
    fake_dir = root / "ljspeech_melgan"
    real_dir.mkdir(parents=True)
    fake_dir.mkdir(parents=True)

    sr = 16000
    for i in range(5):
        waveform = 0.3 * torch.randn(1, sr * 4)
        sf.write(str(real_dir / f"real_{i}.wav"), waveform.squeeze(0).numpy(), sr)
        sf.write(str(fake_dir / f"fake_{i}.wav"), waveform.squeeze(0).numpy(), sr)

    return str(root)


@pytest.fixture
def fake_music_dataset(tmp_path):
    """Create a minimal fake music dataset for testing."""
    real_dir = tmp_path / "MusicCaps" / "audio"
    ai_dir = tmp_path / "AI_Music"
    real_dir.mkdir(parents=True)
    ai_dir.mkdir(parents=True)

    sr = 16000
    for i in range(5):
        waveform = 0.3 * torch.randn(1, sr * 4)
        sf.write(str(real_dir / f"real_music_{i}.wav"), waveform.squeeze(0).numpy(), sr)
        sf.write(str(ai_dir / f"ai_music_{i}.wav"), waveform.squeeze(0).numpy(), sr)

    return str(real_dir), str(ai_dir)


# =============================================================================
# Config tests
# =============================================================================


class TestConfig:
    """Tests for configuration loading."""

    def test_default_config_loads(self):
        """Default config file loads without errors."""
        config = load_config()
        assert config.audio.sample_rate == 16000
        assert config.audio.n_samples == 64000
        assert config.model.projection_dim == 256

    def test_config_override(self):
        """Config overrides work correctly."""
        config = load_config(overrides={
            "audio": {"sample_rate": 8000, "duration_sec": 2.0},
        })
        assert config.audio.sample_rate == 8000

    def test_smoke_test_config(self):
        """Smoke test config has expected defaults."""
        config = load_config()
        assert config.smoke_test.num_samples == 10
        assert config.smoke_test.num_batches == 2


# =============================================================================
# Preprocessing tests
# =============================================================================


class TestPreprocessing:
    """Tests for AudioPreprocessor."""

    def test_load_audio_shape(self, preprocessor, synthetic_wav):
        """Loading audio produces correct shape (1, 64000)."""
        waveform = preprocessor.load_audio(synthetic_wav)
        assert waveform.shape == (1, 64000)

    def test_load_stereo_audio(self, preprocessor, synthetic_stereo_wav):
        """Stereo audio is converted to mono and resampled."""
        waveform = preprocessor.load_audio(synthetic_stereo_wav)
        assert waveform.shape == (1, 64000)

    def test_short_audio_padded(self, preprocessor, short_wav):
        """Short audio is zero-padded to target length."""
        waveform = preprocessor.load_audio(short_wav)
        assert waveform.shape == (1, 64000)

    def test_mel_spectrogram_shape(self, preprocessor, synthetic_wav):
        """Mel spectrogram has shape (1, 128, T)."""
        waveform = preprocessor.load_audio(synthetic_wav)
        mel = preprocessor.compute_mel_spectrogram(waveform)
        assert mel.shape[0] == 1
        assert mel.shape[1] == 128
        assert mel.shape[2] > 0  # T depends on hop_length

    def test_mel_spectrogram_normalized(self, preprocessor, synthetic_wav):
        """Normalized mel spectrogram has approximately zero mean."""
        waveform = preprocessor.load_audio(synthetic_wav)
        mel = preprocessor.compute_mel_spectrogram(waveform)
        assert abs(mel.mean().item()) < 0.1  # Roughly zero-centered

    def test_wavlm_output_shape(self, preprocessor, synthetic_wav):
        """WavLM input is 1D tensor of length n_samples."""
        waveform = preprocessor.load_audio(synthetic_wav)
        wavlm_input = preprocessor.get_raw_waveform(waveform)
        assert wavlm_input.shape == (64000,)

    def test_rawnet2_output_shape(self, preprocessor, synthetic_wav):
        """RawNet2 input is (1, n_samples)."""
        waveform = preprocessor.load_audio(synthetic_wav)
        rawnet2_input = preprocessor.get_rawnet2_input(waveform)
        assert rawnet2_input.shape == (1, 64000)

    def test_full_process(self, preprocessor, synthetic_wav):
        """Full process() returns all expected keys with correct shapes."""
        result = preprocessor.process(synthetic_wav)

        assert "mel_spectrogram" in result
        assert "waveform_wavlm" in result
        assert "waveform_rawnet2" in result
        assert "sample_rate" in result

        assert result["mel_spectrogram"].shape[1] == 128
        assert result["waveform_wavlm"].shape == (64000,)
        assert result["waveform_rawnet2"].shape == (1, 64000)
        assert result["sample_rate"] == 16000


# =============================================================================
# Augmentation tests
# =============================================================================


class TestAugmentation:
    """Tests for CodecAugmentor."""

    def test_augmentor_disabled(self):
        """Disabled augmentor returns waveform unchanged."""
        aug = CodecAugmentor(enabled=False)
        waveform = torch.randn(1, 64000)
        result = aug(waveform)
        assert torch.equal(result, waveform)

    def test_augmentor_output_shape(self):
        """Augmented waveform has same shape as input."""
        aug = CodecAugmentor(probability=1.0, enabled=True, sample_rate=16000)
        waveform = torch.randn(1, 64000)
        result = aug(waveform)
        assert result.shape == (1, 64000)

    def test_voip_augmentation(self):
        """VoIP augmentation produces output of correct shape."""
        aug = CodecAugmentor(enabled=True, sample_rate=16000)
        waveform = 0.5 * torch.randn(1, 64000)
        result = aug.apply_voip(waveform)
        # VoIP may change length slightly due to resampling, but __call__ fixes it
        assert result.shape[0] == 1
        assert result.shape[1] > 0

    def test_voip_modifies_audio(self):
        """VoIP augmentation actually changes the audio (not identity)."""
        aug = CodecAugmentor(enabled=True, sample_rate=16000)
        waveform = 0.5 * torch.randn(1, 64000)
        result = aug.apply_voip(waveform)
        # Due to mu-law quantization, the output should differ
        result_trimmed = result[:, :64000]
        assert not torch.allclose(waveform, result_trimmed, atol=1e-3)

    def test_augmentor_probability_zero(self):
        """With probability=0, augmentor never modifies audio."""
        aug = CodecAugmentor(probability=0.0, enabled=True, sample_rate=16000)
        waveform = torch.randn(1, 64000)
        result = aug(waveform)
        assert torch.equal(result, waveform)


# =============================================================================
# Dataset tests
# =============================================================================


class TestDatasets:
    """Tests for Dataset classes."""

    def test_asvspoof_dataset(self, fake_asvspoof_dataset, preprocessor):
        """ASVspoofDataset loads and returns correct format."""
        from src.data.datasets import ASVspoofDataset

        ds = ASVspoofDataset(
            root_dir=fake_asvspoof_dataset,
            partition="train",
            preprocessor=preprocessor,
        )
        assert len(ds) == 10

        sample = ds[0]
        assert "mel_spectrogram" in sample
        assert "waveform_wavlm" in sample
        assert "waveform_rawnet2" in sample
        assert "label" in sample
        assert "metadata" in sample
        assert sample["label"] in (0, 1)

    def test_asvspoof_labels_correct(self, fake_asvspoof_dataset, preprocessor):
        """ASVspoof labels match protocol: first 5 bonafide, last 5 spoof."""
        from src.data.datasets import ASVspoofDataset

        ds = ASVspoofDataset(
            root_dir=fake_asvspoof_dataset,
            partition="train",
            preprocessor=preprocessor,
        )
        labels = [ds[i]["label"] for i in range(10)]
        assert labels[:5] == [0, 0, 0, 0, 0]
        assert labels[5:] == [1, 1, 1, 1, 1]

    def test_wavefake_dataset(self, fake_wavefake_dataset, preprocessor):
        """WaveFakeDataset discovers files and assigns labels."""
        from src.data.datasets import WaveFakeDataset

        ds = WaveFakeDataset(
            root_dir=fake_wavefake_dataset,
            preprocessor=preprocessor,
        )
        assert len(ds) == 10  # 5 real + 5 fake

        # Check that we have both labels
        labels = [ds[i]["label"] for i in range(len(ds))]
        assert 0 in labels
        assert 1 in labels

    def test_music_dataset(self, fake_music_dataset, preprocessor):
        """MusicAIDataset discovers files correctly."""
        from src.data.datasets import MusicAIDataset

        real_dir, ai_dir = fake_music_dataset
        ds = MusicAIDataset(
            real_music_dir=real_dir,
            ai_music_dir=ai_dir,
            preprocessor=preprocessor,
        )
        assert len(ds) == 10

        labels = [ds[i]["label"] for i in range(len(ds))]
        assert 0 in labels
        assert 1 in labels

    def test_max_samples_limit(self, fake_asvspoof_dataset, preprocessor):
        """max_samples correctly limits dataset size."""
        from src.data.datasets import ASVspoofDataset

        ds = ASVspoofDataset(
            root_dir=fake_asvspoof_dataset,
            partition="train",
            preprocessor=preprocessor,
            max_samples=3,
        )
        assert len(ds) == 3

    def test_unified_dataset(self, fake_asvspoof_dataset, fake_wavefake_dataset, preprocessor):
        """UnifiedAudioDataset combines multiple datasets."""
        from src.data.datasets import ASVspoofDataset, WaveFakeDataset, UnifiedAudioDataset

        ds1 = ASVspoofDataset(
            root_dir=fake_asvspoof_dataset,
            partition="train",
            preprocessor=preprocessor,
        )
        ds2 = WaveFakeDataset(
            root_dir=fake_wavefake_dataset,
            preprocessor=preprocessor,
        )
        unified = UnifiedAudioDataset([ds1, ds2])
        assert len(unified) == len(ds1) + len(ds2)


# =============================================================================
# Collate function tests
# =============================================================================


class TestCollate:
    """Tests for batch collation."""

    def test_collate_fn(self, fake_asvspoof_dataset, preprocessor):
        """Collate function produces correctly batched tensors."""
        from torch.utils.data import DataLoader
        from src.data.datasets import ASVspoofDataset

        ds = ASVspoofDataset(
            root_dir=fake_asvspoof_dataset,
            partition="train",
            preprocessor=preprocessor,
            max_samples=4,
        )

        def _collate_fn(batch):
            return {
                "mel_spectrogram": torch.stack([s["mel_spectrogram"] for s in batch]),
                "waveform_wavlm": torch.stack([s["waveform_wavlm"] for s in batch]),
                "waveform_rawnet2": torch.stack([s["waveform_rawnet2"] for s in batch]),
                "label": torch.tensor([s["label"] for s in batch], dtype=torch.long),
                "metadata": [s["metadata"] for s in batch],
            }

        loader = DataLoader(ds, batch_size=2, collate_fn=_collate_fn)
        batch = next(iter(loader))

        assert batch["mel_spectrogram"].shape[0] == 2
        assert batch["waveform_wavlm"].shape == (2, 64000)
        assert batch["waveform_rawnet2"].shape == (2, 1, 64000)
        assert batch["label"].shape == (2,)
        assert len(batch["metadata"]) == 2


# =============================================================================
# Metrics tests
# =============================================================================


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_eer_perfect_separation(self):
        """EER is 0 when scores perfectly separate classes."""
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.1, 0.15, 0.2, 0.9, 0.85, 0.95, 0.8, 0.9])
        eer, _ = compute_eer(labels, scores)
        assert eer < 0.05  # Should be near zero

    def test_eer_random_scores(self):
        """EER is ~0.5 for random scores."""
        np.random.seed(42)
        labels = np.array([0] * 500 + [1] * 500)
        scores = np.random.rand(1000)
        eer, _ = compute_eer(labels, scores)
        assert 0.3 < eer < 0.7  # Should be roughly 0.5

    def test_min_dcf_perfect(self):
        """min-DCF is near 0 for perfect separation."""
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.1, 0.15, 0.2, 0.9, 0.85, 0.95, 0.8, 0.9])
        min_dcf, _ = compute_min_dcf(labels, scores)
        assert min_dcf < 0.1

    def test_compute_all_metrics(self):
        """compute_all_metrics returns all expected keys."""
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.2, 0.3, 0.7, 0.8])
        result = compute_all_metrics(labels, scores)
        assert "eer" in result
        assert "min_dcf" in result
        assert "accuracy" in result
        assert "eer_threshold" in result
        assert "min_dcf_threshold" in result


# =============================================================================
# Integration: end-to-end pipeline test
# =============================================================================


class TestEndToEnd:
    """End-to-end pipeline integration tests."""

    def test_full_pipeline_smoke(self, fake_asvspoof_dataset, preprocessor):
        """Run the full pipeline: load → preprocess → augment → batch."""
        from torch.utils.data import DataLoader
        from src.data.datasets import ASVspoofDataset

        augmentor = CodecAugmentor(probability=1.0, enabled=True, sample_rate=16000)

        ds = ASVspoofDataset(
            root_dir=fake_asvspoof_dataset,
            partition="train",
            preprocessor=preprocessor,
            augmentor=augmentor,
            max_samples=4,
        )

        def _collate_fn(batch):
            return {
                "mel_spectrogram": torch.stack([s["mel_spectrogram"] for s in batch]),
                "waveform_wavlm": torch.stack([s["waveform_wavlm"] for s in batch]),
                "waveform_rawnet2": torch.stack([s["waveform_rawnet2"] for s in batch]),
                "label": torch.tensor([s["label"] for s in batch], dtype=torch.long),
                "metadata": [s["metadata"] for s in batch],
            }

        loader = DataLoader(ds, batch_size=2, collate_fn=_collate_fn)

        for batch in loader:
            # Verify all tensors are finite and correctly shaped
            assert torch.isfinite(batch["mel_spectrogram"]).all()
            assert torch.isfinite(batch["waveform_wavlm"]).all()
            assert torch.isfinite(batch["waveform_rawnet2"]).all()
            assert batch["label"].max() <= 1
            assert batch["label"].min() >= 0
            break  # Just one batch for smoke test
