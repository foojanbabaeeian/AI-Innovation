"""
Music dataset ingestors.

Handles:
  - MusicCaps (real music, label = 0): downloads from YouTube using yt-dlp.
  - AI-generated music from Suno / Udio (label = 1): ingests from a local
    directory that you have already collected.

After ingestion:
    data/raw/music/real/musiccaps/*.wav       (or .mp3)
    data/raw/music/fake/{generator_name}/*.mp3
"""

import logging
import shutil
import os
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


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


def _yt_dlp_available() -> bool:
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, timeout=10)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def download_musiccaps_audio(
    csv_path: str,
    output_dir: str = "data/raw/music/real/musiccaps",
    max_clips: int = 0,
) -> int:
    """Download MusicCaps audio from YouTube.

    Reads the public MusicCaps CSV (available from Google Research) and
    downloads each clip as a 30-second WAV using yt-dlp.

    Requires yt-dlp to be installed: pip install yt-dlp

    Args:
        csv_path: Path to musiccaps-public.csv.
        output_dir: Destination directory for WAV files.
        max_clips: Maximum number of clips to download (0 = all).

    Returns:
        Number of clips successfully downloaded.
    """
    import pandas as pd

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not _yt_dlp_available():
        logger.error(
            "yt-dlp not found. Install with: pip install yt-dlp\n"
            "Then re-run ingestion."
        )
        return 0

    df = pd.read_csv(csv_path)

    # MusicCaps CSV has columns: ytid, start_s, end_s, audioset_positive_labels, ...
    if "ytid" not in df.columns:
        logger.error("Expected 'ytid' column in %s", csv_path)
        return 0

    if max_clips > 0:
        df = df.iloc[:max_clips]

    count = 0
    for _, row in df.iterrows():
        ytid = row["ytid"]
        start_s = int(row.get("start_s", 0))
        end_s = int(row.get("end_s", start_s + 30))
        duration = end_s - start_s

        out_file = out / f"{ytid}.wav"
        if out_file.exists():
            count += 1
            continue

        url = f"https://www.youtube.com/watch?v={ytid}"
        cmd = [
            "yt-dlp",
            "--quiet",
            "--no-warnings",
            "-x",                           # extract audio
            "--audio-format", "wav",
            "--download-sections", f"*{start_s}-{end_s}",
            "--force-keyframes-at-cuts",
            "-o", str(out_file.with_suffix(".%(ext)s")),
            url,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode == 0 and out_file.exists():
            count += 1
        else:
            logger.debug("Failed to download ytid=%s", ytid)

    logger.info("MusicCaps: downloaded %d/%d clips → %s", count, len(df), out)
    return count


def ingest_ai_music(
    source_dir: str,
    output_dir: str = "data/raw/music/fake",
    generator_name: str = "unknown",
    copy: bool = False,
) -> int:
    """Ingest AI-generated music from a local directory.

    Recursively walks source_dir for audio files and symlinks (or copies)
    them into output_dir/{generator_name}/.

    Args:
        source_dir: Directory containing AI-generated music files.
        output_dir: Destination root (files go into output_dir/{generator_name}/).
        generator_name: Name to use for the generator subdirectory (e.g. "suno").
        copy: If True, copy files instead of symlinking.

    Returns:
        Number of files ingested.
    """
    src = Path(source_dir)
    out = Path(output_dir) / generator_name
    out.mkdir(parents=True, exist_ok=True)
    count = 0

    if not src.exists():
        logger.error("AI music source directory not found: %s", src)
        return 0

    for f in sorted(src.rglob("*")):
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS:
            dst = out / f.name
            _link_or_copy(f, dst, copy)
            count += 1

    logger.info("AI music (%s): ingested %d files → %s", generator_name, count, out)
    return count
