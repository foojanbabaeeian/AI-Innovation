"""
WaveFake dataset ingestor.

WaveFake directory structure:
    WaveFake/
    ├── ljspeech_full_band_melgan/       ← fake (generator = full_band_melgan)
    ├── ljspeech_hifiGAN/                ← fake (generator = hifigan)
    ├── ljspeech_melgan/                 ← fake (generator = melgan)
    ├── ljspeech_melgan_large/           ← fake (generator = melgan_large)
    ├── ljspeech_multi_band_melgan/      ← fake (generator = multi_band_melgan)
    ├── ljspeech_parallel_wavegan/       ← fake (generator = parallel_wavegan)
    └── ljspeech_waveglow/               ← fake (generator = waveglow)

Real LJSpeech audio must be placed separately (usually under a "real" directory).
If a directory name contains "real" or "bonafide" it is treated as real.

After ingestion:
    data/raw/voice/real/wavefake/*.wav
    data/raw/voice/fake/wavefake/{generator}/*.wav
"""

import logging
import shutil
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Known WaveFake generator directory prefixes → canonical names
GENERATOR_ALIASES = {
    "full_band_melgan": "full_band_melgan",
    "hifigan": "hifigan",
    "hifi_gan": "hifigan",
    "melgan_large": "melgan_large",
    "multi_band_melgan": "multi_band_melgan",
    "parallel_wavegan": "parallel_wavegan",
    "waveglow": "waveglow",
    "melgan": "melgan",
}


def _link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        try:
            os.symlink(src.resolve(), dst)
        except (OSError, NotImplementedError):
            shutil.copy2(src, dst)


def _canonical_generator(dirname: str) -> str:
    """Map a WaveFake directory name to a canonical generator name."""
    lower = dirname.lower().replace("ljspeech_", "")
    for key, name in GENERATOR_ALIASES.items():
        if key in lower:
            return name
    return lower  # fallback: use the cleaned directory name


def ingest_wavefake(
    source_dir: str,
    output_dir: str = "data/raw/voice",
    copy: bool = False,
) -> int:
    """Ingest the WaveFake dataset.

    Walks source_dir subdirectories, determines real vs fake from the
    directory name, and symlinks (or copies) .wav files into output_dir.

    Args:
        source_dir: Root of the WaveFake dataset.
        output_dir: Destination root (files go into output_dir/real/ and
                    output_dir/fake/).
        copy: If True, copy files instead of symlinking.

    Returns:
        Number of files ingested.
    """
    src = Path(source_dir)
    out = Path(output_dir)
    count = 0

    if not src.exists():
        logger.error("WaveFake source directory not found: %s", src)
        return 0

    for subdir in sorted(src.iterdir()):
        if not subdir.is_dir():
            continue

        dirname_lower = subdir.name.lower()
        is_real = "real" in dirname_lower or "bonafide" in dirname_lower

        audio_files = list(subdir.glob("*.wav")) + list(subdir.glob("*.flac"))
        if not audio_files:
            continue

        if is_real:
            for f in audio_files:
                dst = out / "real" / "wavefake" / f.name
                _link_or_copy(f, dst, copy)
                count += 1
        else:
            generator = _canonical_generator(subdir.name)
            for f in audio_files:
                dst = out / "fake" / "wavefake" / generator / f.name
                _link_or_copy(f, dst, copy)
                count += 1

    logger.info("WaveFake: ingested %d files → %s", count, out)
    return count
