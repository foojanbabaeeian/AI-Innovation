"""
ASVspoof dataset ingestors.

Organizes ASVspoof 2019 LA, 2021 LA, and ASVspoof 5 audio files into
the unified data/raw/voice/ directory tree used by run_preprocessing.

After ingestion the structure looks like:
    data/raw/voice/real/asvspoof{year}/{utt_id}.flac
    data/raw/voice/fake/asvspoof{year}/{system_id}/{utt_id}.flac
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────

def _link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    """Create a symlink from dst → src, falling back to copy on Windows."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return  # already ingested

    if copy:
        shutil.copy2(src, dst)
    else:
        try:
            os.symlink(src.resolve(), dst)
        except (OSError, NotImplementedError):
            # Windows without developer mode, or unsupported filesystem
            shutil.copy2(src, dst)


def _parse_asvspoof_protocol(protocol_path: Path) -> list:
    """Parse an ASVspoof CM protocol file.

    Protocol line format:
        SPEAKER_ID  UTTERANCE_ID  -  SYSTEM_ID  LABEL

    Returns:
        List of dicts with keys: utt_id, system_id, label (0=bonafide, 1=spoof).
    """
    records = []
    if not protocol_path.exists():
        logger.error("Protocol file not found: %s", protocol_path)
        return records

    with open(protocol_path, "r") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            utt_id = parts[1]
            system_id = parts[3]
            label_str = parts[4]
            label = 0 if label_str.lower() in ("bonafide", "genuine") else 1
            records.append({"utt_id": utt_id, "system_id": system_id, "label": label})

    return records


# ──────────────────────────────────────────────
# Public ingestor functions
# ──────────────────────────────────────────────

def ingest_asvspoof2019(
    source_dir: str,
    output_dir: str = "data/raw/voice",
    copy: bool = False,
) -> int:
    """Ingest ASVspoof 2019 LA dataset.

    Args:
        source_dir: Root of ASVspoof2019_LA (contains ASVspoof2019_LA_cm_protocols/).
        output_dir: Destination root (files go into output_dir/real/ and output_dir/fake/).
        copy: If True, copy files instead of symlinking.

    Returns:
        Number of files ingested.
    """
    src = Path(source_dir)
    out = Path(output_dir)
    count = 0

    protocols_dir = src / "ASVspoof2019_LA_cm_protocols"
    protocol_files = {
        "train": (protocols_dir / "ASVspoof2019.LA.cm.train.trn.txt",
                  src / "ASVspoof2019_LA_train" / "flac"),
        "dev":   (protocols_dir / "ASVspoof2019.LA.cm.dev.trl.txt",
                  src / "ASVspoof2019_LA_dev" / "flac"),
        "eval":  (protocols_dir / "ASVspoof2019.LA.cm.eval.trl.txt",
                  src / "ASVspoof2019_LA_eval" / "flac"),
    }

    for partition, (proto, audio_dir) in protocol_files.items():
        records = _parse_asvspoof_protocol(proto)
        for rec in records:
            utt_id = rec["utt_id"]
            src_file = audio_dir / f"{utt_id}.flac"
            if not src_file.exists():
                continue

            if rec["label"] == 0:
                dst = out / "real" / "asvspoof2019" / f"{utt_id}.flac"
            else:
                dst = out / "fake" / "asvspoof2019" / rec["system_id"] / f"{utt_id}.flac"

            _link_or_copy(src_file, dst, copy)
            count += 1

    logger.info("ASVspoof 2019: ingested %d files → %s", count, out)
    return count


def ingest_asvspoof2021(
    source_dir: str,
    output_dir: str = "data/raw/voice",
    copy: bool = False,
) -> int:
    """Ingest ASVspoof 2021 LA evaluation dataset.

    Args:
        source_dir: Root of ASVspoof2021_LA.
        output_dir: Destination root.
        copy: If True, copy files instead of symlinking.

    Returns:
        Number of files ingested.
    """
    src = Path(source_dir)
    out = Path(output_dir)
    count = 0

    proto = (
        src / "ASVspoof2021_LA_cm_protocols"
        / "ASVspoof2021.LA.cm.eval.trl.txt"
    )
    audio_dir = src / "flac"

    records = _parse_asvspoof_protocol(proto)
    for rec in records:
        utt_id = rec["utt_id"]
        src_file = audio_dir / f"{utt_id}.flac"
        if not src_file.exists():
            continue

        if rec["label"] == 0:
            dst = out / "real" / "asvspoof2021" / f"{utt_id}.flac"
        else:
            dst = out / "fake" / "asvspoof2021" / rec["system_id"] / f"{utt_id}.flac"

        _link_or_copy(src_file, dst, copy)
        count += 1

    logger.info("ASVspoof 2021: ingested %d files → %s", count, out)
    return count


def ingest_asvspoof5(
    source_dir: str,
    output_dir: str = "data/raw/voice",
    copy: bool = False,
) -> int:
    """Ingest ASVspoof 5 dataset.

    ASVspoof 5 follows the same directory/protocol convention as 2021.

    Args:
        source_dir: Root of ASVspoof5 dataset.
        output_dir: Destination root.
        copy: If True, copy files instead of symlinking.

    Returns:
        Number of files ingested.
    """
    src = Path(source_dir)
    out = Path(output_dir)
    count = 0

    # Try common protocol file patterns for ASVspoof5
    possible_protocols = list(src.rglob("*.cm.*.txt")) + list(src.rglob("*.trl.txt"))
    audio_dirs = [src / "flac", src / "audio", src]

    audio_dir: Optional[Path] = None
    for candidate in audio_dirs:
        if candidate.exists() and any(candidate.glob("*.flac")):
            audio_dir = candidate
            break

    if audio_dir is None:
        logger.warning("Could not find audio directory in %s", src)
        return 0

    for proto in possible_protocols:
        records = _parse_asvspoof_protocol(proto)
        for rec in records:
            utt_id = rec["utt_id"]
            src_file = audio_dir / f"{utt_id}.flac"
            if not src_file.exists():
                continue

            if rec["label"] == 0:
                dst = out / "real" / "asvspoof5" / f"{utt_id}.flac"
            else:
                dst = out / "fake" / "asvspoof5" / rec["system_id"] / f"{utt_id}.flac"

            _link_or_copy(src_file, dst, copy)
            count += 1

    logger.info("ASVspoof 5: ingested %d files → %s", count, out)
    return count
