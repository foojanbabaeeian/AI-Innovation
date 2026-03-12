"""
Dataset download utilities and instructions.

Some datasets require manual registration/download. This module provides:
- Automated download for freely available datasets (WaveFake, MusicCaps metadata)
- Clear instructions for datasets requiring manual access (ASVspoof, FakeAVCeleb)
- Verification utilities to check dataset integrity
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm


def _download_file(url: str, dest: Path, desc: Optional[str] = None):
    """Download a file with a progress bar.

    Args:
        url: URL to download from.
        dest: Destination file path.
        desc: Description for the progress bar.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=desc or dest.name
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def download_wavefake(data_root: str = "data"):
    """Download the WaveFake dataset from Zenodo.

    WaveFake is MIT-licensed and freely available.
    Source: https://zenodo.org/record/5642694

    Args:
        data_root: Root directory for datasets.
    """
    dest_dir = Path(data_root) / "WaveFake"
    if dest_dir.exists() and any(dest_dir.iterdir()):
        print(f"[WaveFake] Already exists at {dest_dir}, skipping download.")
        return

    dest_dir.mkdir(parents=True, exist_ok=True)

    # WaveFake is hosted on Zenodo — multiple zip files
    # These are the main archive parts
    zenodo_record = "5642694"
    base_url = f"https://zenodo.org/record/{zenodo_record}/files"

    files = [
        "generated_audio.zip",
        "real_audio.zip",
    ]

    print("[WaveFake] Downloading from Zenodo...")
    print("[WaveFake] NOTE: This dataset is ~8GB total. Ensure sufficient disk space.")

    for fname in files:
        url = f"{base_url}/{fname}"
        dest = dest_dir / fname
        if dest.exists():
            print(f"  {fname} already downloaded, skipping.")
            continue
        try:
            _download_file(url, dest, desc=f"WaveFake/{fname}")
        except requests.HTTPError as e:
            print(f"  WARNING: Could not download {fname}: {e}")
            print(f"  Manual download: https://zenodo.org/record/{zenodo_record}")

    # Extract if unzip is available
    print("[WaveFake] Extracting archives...")
    for fname in files:
        archive = dest_dir / fname
        if archive.exists():
            try:
                subprocess.run(
                    ["unzip", "-o", "-q", str(archive), "-d", str(dest_dir)],
                    check=True,
                )
                print(f"  Extracted {fname}")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print(f"  Could not auto-extract {fname}. Please extract manually.")


def download_musiccaps_metadata(data_root: str = "data"):
    """Download MusicCaps metadata CSV from Google.

    The actual audio must be downloaded separately from YouTube using the video IDs.
    MusicCaps is CC BY-SA 4.0 licensed.

    Args:
        data_root: Root directory for datasets.
    """
    dest_dir = Path(data_root) / "MusicCaps"
    dest_dir.mkdir(parents=True, exist_ok=True)

    csv_path = dest_dir / "musiccaps-public.csv"
    if csv_path.exists():
        print(f"[MusicCaps] Metadata already exists at {csv_path}")
        return

    url = (
        "https://huggingface.co/datasets/google/MusicCaps/resolve/main/"
        "musiccaps-public.csv"
    )
    print("[MusicCaps] Downloading metadata CSV...")
    try:
        _download_file(url, csv_path, desc="MusicCaps metadata")
        print(f"[MusicCaps] Saved to {csv_path}")
    except requests.HTTPError:
        print("[MusicCaps] Could not download automatically.")
        print("  Manual: https://huggingface.co/datasets/google/MusicCaps")

    print("[MusicCaps] NOTE: You still need to download the actual audio files.")
    print("  Use yt-dlp to download audio for each video ID in the CSV.")
    print("  Example: yt-dlp -x --audio-format wav -o '%(id)s.%(ext)s' <VIDEO_ID>")


def print_manual_download_instructions():
    """Print instructions for datasets that require manual download/registration.

    These datasets have license agreements or require institutional access.
    """
    instructions = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MANUAL DATASET DOWNLOAD INSTRUCTIONS                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

━━━ 1. ASVspoof 2019 LA ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - Register at: https://www.asvspoof.org/
  - Download the LA partition (train/dev/eval)
  - Expected structure after extraction:
      data/ASVspoof2019_LA/
      ├── ASVspoof2019_LA_cm_protocols/
      │   ├── ASVspoof2019.LA.cm.train.trn.txt
      │   ├── ASVspoof2019.LA.cm.dev.trl.txt
      │   └── ASVspoof2019.LA.cm.eval.trl.txt
      ├── ASVspoof2019_LA_train/flac/
      ├── ASVspoof2019_LA_dev/flac/
      └── ASVspoof2019_LA_eval/flac/

━━━ 2. ASVspoof 2021 LA ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - Register at: https://www.asvspoof.org/
  - Download the LA evaluation set
  - Expected structure:
      data/ASVspoof2021_LA/
      ├── ASVspoof2021_LA_cm_protocols/
      │   └── ASVspoof2021.LA.cm.eval.trl.txt
      └── flac/

━━━ 3. FakeAVCeleb ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - Request access: https://github.com/DASH-Lab/FakeAVCeleb
  - Requires institutional email and agreement to terms
  - Extract audio tracks from videos using ffmpeg:
      ffmpeg -i video.mp4 -vn -acodec pcm_s16le -ar 16000 output.wav

━━━ 4. AudioSet ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - Ontology: https://research.google.com/audioset/
  - Download segments using the official download tool or yt-dlp
  - Focus on non-speech, non-music categories for environmental sounds
  - Alternatively, use the pre-downloaded subset from:
      https://huggingface.co/datasets/agkphysics/AudioSet

━━━ 5. AI-Generated Music (Self-Collected) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - Generate samples from:
    • Suno AI: https://suno.com/
    • Udio: https://www.udio.com/
  - Save as WAV files in: data/AI_Music/
  - Aim for at least 200-500 samples across different genres
  - Keep metadata (prompt used, model version, generation date)
  - Create a CSV: data/AI_Music/metadata.csv with columns:
      filename, source (suno/udio), genre, prompt, date_generated

━━━ 6. MusicCaps Audio ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - Metadata CSV can be auto-downloaded (run download_musiccaps_metadata())
  - Audio must be downloaded from YouTube using video IDs in the CSV
  - Install yt-dlp: pip install yt-dlp
  - Batch download script:
      import pandas as pd
      import subprocess
      df = pd.read_csv("data/MusicCaps/musiccaps-public.csv")
      for vid in df["ytid"]:
          subprocess.run([
              "yt-dlp", "-x", "--audio-format", "wav",
              "-o", f"data/MusicCaps/audio/%(id)s.%(ext)s",
              f"https://youtube.com/watch?v={vid}"
          ])

══════════════════════════════════════════════════════════════════════════════
After downloading, run verify_datasets() to check that all expected files
are in place.
══════════════════════════════════════════════════════════════════════════════
"""
    print(instructions)


def verify_datasets(data_root: str = "data") -> dict:
    """Check which datasets are present and report their status.

    Args:
        data_root: Root directory for datasets.

    Returns:
        Dictionary mapping dataset name to status dict with keys:
        'found' (bool), 'path' (str), 'num_files' (int), 'notes' (str).
    """
    root = Path(data_root)
    results = {}

    # ASVspoof 2019
    asvspoof19_dir = root / "ASVspoof2019_LA"
    protocols = asvspoof19_dir / "ASVspoof2019_LA_cm_protocols"
    num_flac = sum(
        1 for _ in asvspoof19_dir.rglob("*.flac")
    ) if asvspoof19_dir.exists() else 0
    results["ASVspoof2019_LA"] = {
        "found": asvspoof19_dir.exists() and num_flac > 0,
        "path": str(asvspoof19_dir),
        "num_files": num_flac,
        "notes": "Need ~150K+ FLAC files across train/dev/eval"
        if num_flac == 0 else f"Found {num_flac} FLAC files",
    }

    # ASVspoof 2021
    asvspoof21_dir = root / "ASVspoof2021_LA"
    num_flac_21 = sum(
        1 for _ in asvspoof21_dir.rglob("*.flac")
    ) if asvspoof21_dir.exists() else 0
    results["ASVspoof2021_LA"] = {
        "found": asvspoof21_dir.exists() and num_flac_21 > 0,
        "path": str(asvspoof21_dir),
        "num_files": num_flac_21,
        "notes": "Need eval partition FLAC files"
        if num_flac_21 == 0 else f"Found {num_flac_21} FLAC files",
    }

    # WaveFake
    wf_dir = root / "WaveFake"
    num_wav = sum(
        1 for _ in wf_dir.rglob("*.wav")
    ) if wf_dir.exists() else 0
    results["WaveFake"] = {
        "found": wf_dir.exists() and num_wav > 0,
        "path": str(wf_dir),
        "num_files": num_wav,
        "notes": "MIT license, auto-downloadable"
        if num_wav == 0 else f"Found {num_wav} WAV files",
    }

    # MusicCaps
    mc_dir = root / "MusicCaps"
    mc_csv = mc_dir / "musiccaps-public.csv"
    mc_audio_dir = mc_dir / "audio"
    num_mc_audio = sum(
        1 for _ in mc_audio_dir.rglob("*.wav")
    ) if mc_audio_dir.exists() else 0
    results["MusicCaps"] = {
        "found": mc_csv.exists(),
        "path": str(mc_dir),
        "num_files": num_mc_audio,
        "notes": f"CSV: {'✓' if mc_csv.exists() else '✗'}, Audio: {num_mc_audio} files",
    }

    # AI Music
    ai_dir = root / "AI_Music"
    num_ai = sum(
        1 for ext in ["*.wav", "*.mp3", "*.flac"]
        for _ in ai_dir.rglob(ext)
    ) if ai_dir.exists() else 0
    results["AI_Music"] = {
        "found": ai_dir.exists() and num_ai > 0,
        "path": str(ai_dir),
        "num_files": num_ai,
        "notes": "Self-collected from Suno/Udio"
        if num_ai == 0 else f"Found {num_ai} audio files",
    }

    # Print report
    print("\n" + "=" * 60)
    print("DATASET VERIFICATION REPORT")
    print("=" * 60)
    for name, info in results.items():
        status = "✓ FOUND" if info["found"] else "✗ MISSING"
        print(f"\n  {status}  {name}")
        print(f"         Path: {info['path']}")
        print(f"         Files: {info['num_files']}")
        print(f"         Notes: {info['notes']}")
    print("\n" + "=" * 60)

    return results


def download_all_available(data_root: str = "data"):
    """Download all freely available datasets and print instructions for the rest.

    Args:
        data_root: Root directory for datasets.
    """
    print("=" * 60)
    print("DOWNLOADING AVAILABLE DATASETS")
    print("=" * 60)

    download_wavefake(data_root)
    print()
    download_musiccaps_metadata(data_root)
    print()
    print_manual_download_instructions()
    print()
    verify_datasets(data_root)


if __name__ == "__main__":
    download_all_available()
