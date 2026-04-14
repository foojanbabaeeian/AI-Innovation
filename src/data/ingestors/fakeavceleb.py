"""
FakeAVCeleb dataset ingestor.

FakeAVCeleb contains real and fake audio-visual clips.  We use only the
audio track.  The directory structure is:

    FakeAVCeleb_v1.2/
    ├── RealVideo-RealAudio/    ← real audio (label = 0)
    ├── FakeVideo-FakeAudio/    ← fake audio (label = 1)
    └── RealVideo-FakeAudio/    ← fake audio, voice-converted (label = 1)

Audio is extracted from mp4 files using ffmpeg when only video files are
present.  WAV files are used directly if they already exist.

After ingestion:
    data/raw/voice/real/fakeavceleb/*.wav
    data/raw/voice/fake/fakeavceleb/{category}/*.wav
"""

import logging
import shutil
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


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


def _extract_audio(video_path: Path, wav_path: Path) -> bool:
    """Extract mono 16kHz WAV from a video file using ffmpeg.

    Args:
        video_path: Source mp4/mkv file.
        wav_path: Destination WAV file.

    Returns:
        True on success, False on failure.
    """
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(video_path),
                "-vn",                      # no video
                "-ac", "1",                 # mono
                "-ar", "16000",             # 16 kHz
                "-acodec", "pcm_s16le",
                str(wav_path),
            ],
            capture_output=True,
            timeout=120,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.warning("ffmpeg extraction failed for %s: %s", video_path, exc)
        return False


def ingest_fakeavceleb(
    source_dir: str,
    output_dir: str = "data/raw/voice",
    copy: bool = True,
) -> int:
    """Ingest the FakeAVCeleb dataset (audio track only).

    Args:
        source_dir: Root of FakeAVCeleb_v1.2.
        output_dir: Destination root.
        copy: If True, copy WAV files instead of symlinking.
                WAV files extracted from mp4 are always copied.

    Returns:
        Number of audio files ingested.
    """
    src = Path(source_dir)
    out = Path(output_dir)
    count = 0

    # Category → (is_real, category_slug)
    category_map = {
        "RealVideo-RealAudio": (True, "realvideo_realaudio"),
        "FakeVideo-FakeAudio": (False, "fakevideo_fakeaudio"),
        "RealVideo-FakeAudio": (False, "realvideo_fakeaudio"),
    }

    if not src.exists():
        logger.error("FakeAVCeleb source directory not found: %s", src)
        return 0

    for category_dir in sorted(src.iterdir()):
        if not category_dir.is_dir():
            continue

        info = category_map.get(category_dir.name)
        if info is None:
            # Try prefix match for nested layouts
            matched = None
            for key, val in category_map.items():
                if key.lower() in category_dir.name.lower():
                    matched = val
                    break
            if matched is None:
                continue
            info = matched

        is_real, slug = info

        for audio_file in sorted(category_dir.rglob("*.wav")):
            stem = audio_file.stem
            if is_real:
                dst = out / "real" / "fakeavceleb" / f"{stem}.wav"
            else:
                dst = out / "fake" / "fakeavceleb" / slug / f"{stem}.wav"
            _link_or_copy(audio_file, dst, copy)
            count += 1

        # Extract audio from mp4 files if no wav files found
        mp4_files = list(category_dir.rglob("*.mp4"))
        if mp4_files and count == 0:
            for mp4 in mp4_files:
                stem = mp4.stem
                if is_real:
                    dst = out / "real" / "fakeavceleb" / f"{stem}.wav"
                else:
                    dst = out / "fake" / "fakeavceleb" / slug / f"{stem}.wav"
                if not dst.exists():
                    if _extract_audio(mp4, dst):
                        count += 1

    logger.info("FakeAVCeleb: ingested %d files → %s", count, out)
    return count
