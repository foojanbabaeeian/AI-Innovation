"""
Master data ingestion script.

Reads a YAML config specifying where each source dataset lives on disk,
runs the appropriate ingestor for each, then runs preprocessing to
produce the final manifest and segmented audio.

Usage:
    python scripts/ingest_all.py --config scripts/ingest_config.yaml
    python scripts/ingest_all.py --config scripts/ingest_config.yaml --preprocess
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.ingestors.asvspoof import ingest_asvspoof2019, ingest_asvspoof2021, ingest_asvspoof5
from src.data.ingestors.wavefake import ingest_wavefake
from src.data.ingestors.fakeavceleb import ingest_fakeavceleb
from src.data.ingestors.music import download_musiccaps_audio, ingest_ai_music
from src.data.ingestors.audioset import download_audioset_subset, ingest_ai_sfx
from src.data.preprocessing import run_preprocessing

logger = logging.getLogger(__name__)


INGESTOR_REGISTRY = {
    "asvspoof2019": {
        "fn": ingest_asvspoof2019,
        "required_keys": ["source_dir"],
        "defaults": {"output_dir": "data/raw/voice", "copy": False},
    },
    "asvspoof2021": {
        "fn": ingest_asvspoof2021,
        "required_keys": ["source_dir"],
        "defaults": {"output_dir": "data/raw/voice", "copy": False},
    },
    "asvspoof5": {
        "fn": ingest_asvspoof5,
        "required_keys": ["source_dir"],
        "defaults": {"output_dir": "data/raw/voice", "copy": False},
    },
    "wavefake": {
        "fn": ingest_wavefake,
        "required_keys": ["source_dir"],
        "defaults": {"output_dir": "data/raw/voice", "copy": False},
    },
    "fakeavceleb": {
        "fn": ingest_fakeavceleb,
        "required_keys": ["source_dir"],
        "defaults": {"output_dir": "data/raw/voice"},
    },
    "musiccaps": {
        "fn": download_musiccaps_audio,
        "required_keys": ["csv_path"],
        "defaults": {"output_dir": "data/raw/music/real/musiccaps", "max_clips": 0},
    },
    "ai_music_suno": {
        "fn": lambda **kw: ingest_ai_music(generator_name="suno", **kw),
        "required_keys": ["source_dir"],
        "defaults": {"output_dir": "data/raw/music/fake", "copy": False},
    },
    "ai_music_udio": {
        "fn": lambda **kw: ingest_ai_music(generator_name="udio", **kw),
        "required_keys": ["source_dir"],
        "defaults": {"output_dir": "data/raw/music/fake", "copy": False},
    },
    "audioset": {
        "fn": download_audioset_subset,
        "required_keys": ["segments_csv"],
        "defaults": {"output_dir": "data/raw/non_human/real/audioset", "max_clips": 5000},
    },
    "ai_sfx_elevenlabs": {
        "fn": lambda **kw: ingest_ai_sfx(generator_name="elevenlabs_sfx", **kw),
        "required_keys": ["source_dir"],
        "defaults": {"output_dir": "data/raw/non_human/fake", "copy": False},
    },
    "ai_sfx_bark": {
        "fn": lambda **kw: ingest_ai_sfx(generator_name="bark_sfx", **kw),
        "required_keys": ["source_dir"],
        "defaults": {"output_dir": "data/raw/non_human/fake", "copy": False},
    },
}


def run_ingestion(config_path: str):
    """Run all ingestors defined in the config file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    datasets = config.get("datasets", {})
    results = {}

    for name, params in datasets.items():
        if not params.get("enabled", True):
            logger.info(f"SKIP (disabled): {name}")
            continue

        if name not in INGESTOR_REGISTRY:
            logger.warning(f"Unknown dataset: {name} -- skipping")
            continue

        registry = INGESTOR_REGISTRY[name]

        # Check required keys
        missing = [k for k in registry["required_keys"] if k not in params]
        if missing:
            logger.error(f"Dataset {name} missing required keys: {missing}")
            continue

        # Build kwargs: user params override defaults
        kwargs = {**registry["defaults"], **{k: v for k, v in params.items() if k != "enabled"}}

        logger.info(f"Ingesting: {name}")
        try:
            count = registry["fn"](**kwargs)
            results[name] = count
            logger.info(f"  -> {count} files ingested")
        except Exception as e:
            logger.error(f"  -> FAILED: {e}")
            results[name] = 0

    # Summary
    print("\n=== Ingestion Summary ===")
    for name, count in results.items():
        print(f"  {name:<25} {count:>8} files")
    print(f"  {'TOTAL':<25} {sum(results.values()):>8} files")

    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Ingest all datasets into data/raw/")
    parser.add_argument("--config", type=str, default="scripts/ingest_config.yaml")
    parser.add_argument("--preprocess", action="store_true", help="Run preprocessing after ingestion")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    run_ingestion(args.config)

    if args.preprocess:
        logger.info("Running preprocessing pipeline...")
        run_preprocessing(
            raw_dir="data/raw",
            output_dir="data/processed",
            manifest_path="data/metadata/master_manifest.csv",
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
