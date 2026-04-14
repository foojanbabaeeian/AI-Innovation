"""
Non-human sound ingestors.

Handles:
  - ESC-50 / AudioSet (real environmental sounds, label = 0)
  - AI-generated SFX from ElevenLabs or LAION-AI (label = 1)

ESC-50 is the recommended real non-human source.  Clone the repo:
    git clone https://github.com/karolpiczak/ESC-50 data/raw/non_human/real/esc50

Then run ingest_ai_sfx for the AI-generated counterpart.

After ingestion:
    data/raw/non_human/real/{source_name}/*.wav
    data/raw/non_human/fake/{generator_name}/*.wav
"""

import csv
import logging
import shutil
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aif", ".aiff"}

# ESC-50 categories that are non-human environmental / nature sounds
# (exclude human sounds: categories 0-9 in ESC-50 are human-related)
ESC50_NON_HUMAN_CATEGORIES = {
    "dog", "rooster", "pig", "cow", "frog", "cat", "hen",
    "insects", "sheep", "crow",                          # animals
    "rain", "sea_waves", "crackling_fire", "crickets",
    "chirping_birds", "water_drops", "wind", "pouring_water",
    "toilet_flush", "thunderstorm",                      # natural soundscapes
    "crying_baby",                                       # (optional — borderline)
    "sneezing", "clapping", "breathing", "coughing",     # human but non-speech
    "footsteps", "laughing", "brushing_teeth",
    "snoring", "drinking_sipping",                       # non-speech human
    "door_wood_knock", "mouse_click", "keyboard_typing",
    "door_wood_creaks", "can_opening", "washing_machine",
    "vacuum_cleaner", "clock_alarm", "clock_tick",
    "glass_breaking", "helicopter", "chainsaw",
    "siren", "car_horn", "engine", "train",
    "church_bells", "airplane", "fireworks", "hand_saw",
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


def _yt_dlp_available() -> bool:
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, timeout=10)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def ingest_esc50(
    source_dir: str,
    output_dir: str = "data/raw/non_human/real/esc50",
    copy: bool = False,
    non_human_only: bool = True,
) -> int:
    """Ingest ESC-50 environmental sounds dataset.

    ESC-50 structure:
        ESC-50/
        ├── audio/
        │   └── *.wav   (filename: {fold}-{clip_id}-{take}-{target}.wav)
        └── meta/
            └── esc50.csv

    Args:
        source_dir: Root of the ESC-50 repository/download.
        output_dir: Destination directory.
        copy: If True, copy files instead of symlinking.
        non_human_only: If True, skip human voice/speech categories.

    Returns:
        Number of files ingested.
    """
    src = Path(source_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    count = 0

    audio_dir = src / "audio"
    meta_csv = src / "meta" / "esc50.csv"

    if not audio_dir.exists():
        # Fallback: look for wav files directly under source_dir
        audio_dir = src

    if meta_csv.exists():
        # Use metadata to filter non-human categories
        with open(meta_csv, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                category = row.get("category", "").lower()
                if non_human_only and category not in ESC50_NON_HUMAN_CATEGORIES:
                    continue
                filename = row.get("filename", "")
                src_file = audio_dir / filename
                if src_file.exists():
                    dst = out / filename
                    _link_or_copy(src_file, dst, copy)
                    count += 1
    else:
        # No metadata — ingest all wav files
        for f in sorted(audio_dir.glob("*.wav")):
            dst = out / f.name
            _link_or_copy(f, dst, copy)
            count += 1

    logger.info("ESC-50: ingested %d files → %s", count, out)
    return count


def download_audioset_subset(
    segments_csv: str,
    output_dir: str = "data/raw/non_human/real/audioset",
    max_clips: int = 5000,
) -> int:
    """Download a subset of AudioSet non-human environmental sounds.

    Reads AudioSet balanced_train_segments.csv and downloads clips that
    belong to non-human sound categories using yt-dlp.

    NOTE: ESC-50 (ingest_esc50) is the recommended alternative — it requires
    no downloading and is already curated for environmental sounds.

    Args:
        segments_csv: Path to AudioSet balanced_train_segments.csv.
        output_dir: Destination directory for downloaded clips.
        max_clips: Maximum number of clips to download.

    Returns:
        Number of clips successfully downloaded.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not _yt_dlp_available():
        logger.error(
            "yt-dlp not found. Install with: pip install yt-dlp\n"
            "Or use ingest_esc50() instead (no download needed)."
        )
        return 0

    # AudioSet non-human environmental sound label IDs (subset)
    # Full ontology: https://research.google.com/audioset/ontology/
    NATURE_LABELS = {
        "/m/0jbk",   # animal
        "/m/01d380", # music
        "/m/07yv9",  # vehicle
        "/m/07rwj",  # rain
        "/m/0ytgt",  # thunder
        "/m/07c52", "/m/07phhsh", "/m/07plct2",  # wind variants
    }

    count = 0
    with open(segments_csv, newline="") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.strip().split(", ")
            if len(parts) < 4:
                continue
            ytid = parts[0].strip()
            start_s = float(parts[1])
            end_s = float(parts[2])
            labels = set(parts[3].replace('"', "").split(","))

            if not labels.intersection(NATURE_LABELS):
                continue

            out_file = out / f"{ytid}_{int(start_s)}.wav"
            if out_file.exists():
                count += 1
                if count >= max_clips:
                    break
                continue

            url = f"https://www.youtube.com/watch?v={ytid}"
            cmd = [
                "yt-dlp", "--quiet", "--no-warnings",
                "-x", "--audio-format", "wav",
                "--download-sections", f"*{start_s}-{end_s}",
                "--force-keyframes-at-cuts",
                "-o", str(out_file.with_suffix(".%(ext)s")),
                url,
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            if result.returncode == 0 and out_file.exists():
                count += 1
                logger.debug("Downloaded %s", ytid)
            if count >= max_clips:
                break

    logger.info("AudioSet: downloaded %d clips → %s", count, out)
    return count


def ingest_ai_sfx(
    source_dir: str,
    output_dir: str = "data/raw/non_human/fake",
    generator_name: str = "unknown",
    copy: bool = False,
) -> int:
    """Ingest AI-generated sound effects from a local directory.

    Recursively walks source_dir and symlinks (or copies) audio files
    into output_dir/{generator_name}/.

    Args:
        source_dir: Directory containing AI-generated SFX audio files.
        output_dir: Destination root.
        generator_name: Name for the generator subdirectory (e.g. "elevenlabs_sfx").
        copy: If True, copy files instead of symlinking.

    Returns:
        Number of files ingested.
    """
    src = Path(source_dir)
    out = Path(output_dir) / generator_name
    out.mkdir(parents=True, exist_ok=True)
    count = 0

    if not src.exists():
        logger.error("AI SFX source directory not found: %s", src)
        return 0

    for f in sorted(src.rglob("*")):
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS:
            dst = out / f.name
            _link_or_copy(f, dst, copy)
            count += 1

    logger.info("AI SFX (%s): ingested %d files → %s", generator_name, count, out)
    return count
