"""
General-purpose voice ingestors.

Handles:
  - LJSpeech (real TTS training corpus, label = 0)
  - Generic audio directory ingestor for any pre-organized folder

After ingestion:
    data/raw/voice/real/ljspeech/*.wav
    data/raw/voice/fake/{generator_name}/*.wav
"""

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg"}


def _link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy:
        shutil.copy2(src, dst)
        return
    # Try hard link first (no admin needed, works on same volume, Windows-safe)
    try:
        os.link(src.resolve(), dst)
        return
    except (OSError, NotImplementedError):
        pass
    # Try symlink (requires Developer Mode or admin on Windows)
    try:
        os.symlink(src.resolve(), dst)
        return
    except (OSError, NotImplementedError):
        pass
    # Last resort: copy
    shutil.copy2(src, dst)


def ingest_ljspeech(
    source_dir: str,
    output_dir: str = "data/raw/voice",
    copy: bool = False,
) -> int:
    """Ingest LJSpeech as real (label=0) voice data.

    LJSpeech directory structure:
        LJSpeech-1.1/
        ├── wavs/       ← 13,100 WAV files (16-bit PCM, 22050 Hz)
        └── metadata.csv

    Args:
        source_dir: Root of LJSpeech-1.1 (contains wavs/ subdirectory).
        output_dir: Destination root (files go into output_dir/real/ljspeech/).
        copy: If True, copy files instead of symlinking.

    Returns:
        Number of files ingested.
    """
    src = Path(source_dir)
    out = Path(output_dir) / "real" / "ljspeech"
    out.mkdir(parents=True, exist_ok=True)
    count = 0

    # LJSpeech stores wavs in a wavs/ subdirectory
    wavs_dir = src / "wavs"
    if not wavs_dir.exists():
        # Some extracted archives put wavs directly in root
        wavs_dir = src

    for f in sorted(wavs_dir.glob("*.wav")):
        dst = out / f.name
        _link_or_copy(f, dst, copy)
        count += 1

    logger.info("LJSpeech: ingested %d files → %s", count, out)
    return count


def ingest_generic_audio(
    source_dir: str,
    output_dir: str,
    label: int = 1,
    source_name: str = "generated",
    copy: bool = False,
) -> int:
    """Ingest any directory of audio files with a fixed label.

    Useful for AI-generated audio produced by notebooks (AudioGen,
    SpeechT5, Bark, etc.) that is already in a flat directory.

    Args:
        source_dir: Directory containing audio files.
        output_dir: Full destination path (e.g. "data/raw/voice/fake/speecht5").
        label: 0 for real, 1 for AI-generated (used only for logging).
        source_name: Name to use in log messages.
        copy: If True, copy files instead of symlinking.

    Returns:
        Number of files ingested.
    """
    src = Path(source_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    count = 0

    if not src.exists():
        logger.error("Source directory not found: %s", src)
        return 0

    for f in sorted(src.rglob("*")):
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS:
            dst = out / f.name
            _link_or_copy(f, dst, copy)
            count += 1

    label_str = "real" if label == 0 else "AI-generated"
    logger.info("%s (%s): ingested %d files → %s", source_name, label_str, count, out)
    return count
