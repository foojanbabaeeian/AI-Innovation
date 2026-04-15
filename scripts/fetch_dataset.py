"""
Standalone dataset fetcher — run multiple instances in parallel terminals.

Usage (open 3 Anaconda Prompt windows, each with: conda activate tf-gpu-210):

  Terminal 1 — finish LJSpeech + ingest (real voice):
    python scripts/fetch_dataset.py ljspeech

  Terminal 2 — generate SpeechT5 AI voice (GPU, ~45 min):
    python scripts/fetch_dataset.py speecht5

  Terminal 3 — generate AudioGen AI sounds (GPU, ~45 min):
    python scripts/fetch_dataset.py audiogen

  Terminal 4 — clone ESC-50 + ingest (if not done yet):
    python scripts/fetch_dataset.py esc50

NOTE: speecht5 and audiogen both use the GPU — run them one at a time,
      or run one in the notebook and one here.
"""

import argparse
import csv
import importlib.util
import logging
import os
import sys
import urllib.request
import tarfile
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

DRIVE_ROOT = Path("C:/Users/fooja/Google Drive Streaming/My Drive/AI-Innovation-Data")
RAW = DRIVE_ROOT / "raw"
MANIFEST = ROOT / "data" / "metadata" / "master_manifest.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def _load_manifest():
    if not MANIFEST.exists():
        return [], set()
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows, {r["sample_id"] for r in rows}


def _save_manifest(rows):
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample_id", "file_path", "label", "domain", "source_dataset", "generator", "split"]
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_ingestor(name):
    """Load an ingestor module directly — bypasses src/data/__init__.py (which imports torch)."""
    path = ROOT / "src" / "data" / "ingestors" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── LJSpeech ─────────────────────────────────────────────────────────────────

def fetch_ljspeech():
    lj_dir = RAW / "voice" / "real" / "LJSpeech-1.1"
    lj_tar = RAW / "voice" / "real" / "LJSpeech-1.1.tar.bz2"
    wavs_dir = lj_dir / "wavs"

    existing = list(wavs_dir.glob("*.wav")) if wavs_dir.exists() else []
    log.info(f"LJSpeech: {len(existing)} wavs already present")

    if not lj_dir.exists() or len(existing) < 13000:
        if not lj_tar.exists():
            log.info("Downloading LJSpeech (~2.6 GB)...")
            lj_tar.parent.mkdir(parents=True, exist_ok=True)

            def _progress(count, block, total):
                pct = min(count * block / total * 100, 100)
                print(f"\r  {pct:.1f}%", end="", flush=True)

            urllib.request.urlretrieve(
                "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
                lj_tar, reporthook=_progress,
            )
            print()
            log.info("Download complete.")

        if not lj_dir.exists():
            log.info("Extracting LJSpeech...")
            with tarfile.open(lj_tar, "r:bz2") as tf:
                tf.extractall(lj_dir.parent)
            log.info("Extracted.")

    # Register in manifest directly — no file copying (avoids filling Drive cache)
    existing_rows, existing_ids = _load_manifest()
    new_rows = []
    wavs = sorted(wavs_dir.glob("*.wav"))
    for wav in wavs:
        sample_id = f"ljspeech_{wav.stem}"
        if sample_id in existing_ids:
            continue
        new_rows.append({
            "sample_id":      sample_id,
            "file_path":      str(wav).replace("\\", "/"),
            "label":          0.0,
            "domain":         "voice",
            "source_dataset": "ljspeech",
            "generator":      "human",
            "split":          "train",
        })

    if not new_rows:
        log.info(f"LJSpeech already in manifest ({len(wavs)} files). Nothing to add.")
        return

    _save_manifest(existing_rows + new_rows)
    log.info(f"LJSpeech done: registered {len(new_rows)} files in manifest (no copy)")


# ── SpeechT5 (AI voice, replaces WaveFake) ───────────────────────────────────

def fetch_speecht5():
    """
    Generate 1000 AI voice clips using Microsoft SpeechT5 TTS.
    Uses LibriSpeech transcripts + 5 CMU Arctic speaker embeddings.
    No large downloads — model weights are ~100 MB from HuggingFace.
    """
    import torch
    out_dir = RAW / "voice" / "fake" / "speecht5"
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = list(out_dir.glob("speecht5_*.wav"))
    if len(existing) >= 1000:
        log.info(f"SpeechT5 already complete: {len(existing)} files")
        return

    log.info(f"SpeechT5: {len(existing)}/1000 already done")

    # Install datasets if needed
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets"], check=True)

    from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
    from datasets import load_dataset
    import torchaudio

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {device}")

    log.info("Loading SpeechT5 model...")
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

    log.info("Loading speaker embeddings (CMU Arctic)...")
    embed_ds = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_ids = [7306, 7307, 7308, 7309, 7310]
    speakers = [
        torch.tensor(embed_ds[sid]["xvector"]).unsqueeze(0).to(device)
        for sid in speaker_ids
    ]

    log.info("Loading LibriSpeech texts...")
    libri = load_dataset("librispeech_asr", "clean", split="test", trust_remote_code=True)
    texts = [row["text"] for row in libri.select(range(min(1000, len(libri))))]

    log.info(f"Generating {len(texts)} clips...")
    start = len(existing)
    for i, text in enumerate(texts):
        out_path = out_dir / f"speecht5_{i:05d}.wav"
        if out_path.exists():
            continue
        try:
            spk = speakers[i % len(speakers)]
            inputs = processor(text=text[:200], return_tensors="pt").to(device)
            with torch.no_grad():
                speech = model.generate_speech(inputs["input_ids"], spk, vocoder=vocoder)
            torchaudio.save(str(out_path), speech.unsqueeze(0).cpu(), 16000)
        except Exception as e:
            log.warning(f"  Skipped clip {i}: {e}")

        if (i + 1) % 50 == 0:
            done = len(list(out_dir.glob("speecht5_*.wav")))
            log.info(f"  {done}/1000 clips done")
            if device == "cuda":
                torch.cuda.empty_cache()

    final = list(out_dir.glob("speecht5_*.wav"))
    log.info(f"SpeechT5 generation done: {len(final)} clips")

    # Register in manifest
    existing_rows, existing_ids = _load_manifest()
    new_rows = []
    for wav in sorted(final):
        sample_id = f"speecht5_{wav.stem}"
        if sample_id in existing_ids:
            continue
        new_rows.append({
            "sample_id":      sample_id,
            "file_path":      str(wav).replace("\\", "/"),
            "label":          1.0,
            "domain":         "voice",
            "source_dataset": "speecht5",
            "generator":      "speecht5",
            "split":          "train",
        })
    if new_rows:
        _save_manifest(existing_rows + new_rows)
        log.info(f"SpeechT5: registered {len(new_rows)} files in manifest")
    else:
        log.info("SpeechT5 already fully registered in manifest.")


# ── AudioGen (AI non-human sounds) ───────────────────────────────────────────

def fetch_audiogen():
    import torch
    import torchaudio
    import numpy as np
    # numpy 2.x removed bool8/complex256/float128 — patch for audiocraft's tensorboard dep
    if not hasattr(np, 'bool8'):
        np.bool8 = np.bool_

    try:
        from audiocraft.models import AudioGen
    except ImportError:
        log.info("Installing audiocraft...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "audiocraft"], check=True)
        import numpy as np
        if not hasattr(np, 'bool8'):
            np.bool8 = np.bool_
        from audiocraft.models import AudioGen

    out_dir = RAW / "non_human" / "fake" / "audiogen"
    out_dir.mkdir(parents=True, exist_ok=True)

    PROMPTS = [
        "heavy rain on a tin roof", "light drizzle on pavement", "tropical downpour",
        "rain falling on leaves", "thunderstorm with heavy rain", "hailstorm on glass",
        "ocean waves crashing on rocks", "gentle ocean surf", "river flowing over rocks",
        "small stream babbling", "waterfall in a forest", "water dripping in a cave",
        "lake water lapping shore", "waves on a pebble beach", "rain in a forest",
        "loud thunder clap", "distant rolling thunder", "thunder rumble after lightning",
        "strong wind gusts", "wind howling through trees", "gentle breeze in leaves",
        "wind in tall grass", "gusty wind through a canyon", "stormy wind at night",
        "wind whistling through cracks", "crackling campfire", "roaring bonfire",
        "fire in a fireplace", "small fire burning dry wood", "burning leaves",
        "fire popping and hissing", "dog barking in distance", "dog growling",
        "multiple dogs barking", "cat meowing", "cat purring", "rooster crowing at dawn",
        "cow mooing in a field", "horse neighing", "sheep bleating", "pig oinking",
        "frog croaking at night", "crickets at night", "cicadas in summer heat",
        "birds chirping in morning", "owl hooting at night", "crow cawing",
        "hen clucking", "duck quacking", "wolf howling at moon", "bear growling",
        "buzzing bee", "mosquito buzzing", "grasshoppers in field",
        "insect sounds at night", "cicadas and crickets together",
        "car engine idling", "car horn honking", "car passing on highway",
        "motorcycle engine revving", "train passing on tracks",
        "helicopter flying overhead", "airplane flying above", "jet engine roar",
        "ambulance siren", "police car siren", "fire truck siren",
        "chainsaw running", "electric drill", "jackhammer on concrete",
        "lawnmower", "vacuum cleaner running", "washing machine cycle",
        "refrigerator humming", "air conditioner", "keyboard typing fast",
        "mouse clicking", "clock ticking", "clock alarm ringing", "phone ringing",
        "door knocking", "door creaking open", "glass shattering", "dishes clattering",
        "keys jingling", "coins dropping on floor", "paper rustling",
        "book pages turning", "scissors cutting paper",
        "crowd cheering at stadium", "crowd applause", "busy street ambience",
        "restaurant background noise", "market crowd sounds",
        "construction site ambience", "airport terminal ambience",
        "subway train arriving", "school playground sounds",
        "fireworks exploding", "fireworks show", "single firework pop",
        "balloon popping", "book dropped on floor", "door slamming shut",
        "tropical jungle ambience", "forest ambience at dawn",
        "forest ambience at night", "desert wind ambience",
        "mountain stream with birds", "swamp ambience with frogs",
        "summer meadow with insects", "winter wind in bare trees",
        "spring rain on grass", "autumn leaves blowing",
        "blizzard snow and wind", "fog horn in harbor", "hail on car roof",
        "ice cracking", "lion roaring", "elephant trumpeting", "monkey chattering",
        "dolphin clicking", "whale song", "seagulls calling", "geese honking",
        "woodpecker drumming on tree", "turkey gobbling", "peacock calling",
        "toilet flushing", "sink water running", "shower running",
        "kettle boiling", "popcorn popping", "bacon sizzling",
        "ice cubes in glass", "fizzy drink opening",
        "bus engine starting", "bus doors opening", "train whistle",
        "boat engine on water", "bicycle bell ringing",
        "skateboard rolling on pavement", "rollerskates on floor",
        "thunder and heavy rain together", "hail and thunder",
        "wind and rain at sea", "storm on open ocean",
        "stream with frogs at dusk", "birds and stream together",
        "forest fire crackling", "dry leaves in wind",
        "microwave beeping", "oven timer", "smoke alarm beeping",
        "camera shutter", "notification sound",
        "basketball bouncing", "tennis ball hit", "golf club swing",
        "baseball bat hit", "football crowd roar",
        "hammer hitting nail", "saw cutting wood", "welding torch",
        "electric sander", "power tool drilling",
        "morning birds with light wind", "evening crickets with frogs",
        "midnight rain on rooftop", "sunrise bird chorus",
        "thunderstorm approaching", "storm passing with rain",
        "after rain bird sounds", "snow falling silently",
    ][:200]

    existing = list(out_dir.glob("audiogen_*.wav"))
    start_idx = len(existing)
    log.info(f"AudioGen: {start_idx}/200 already done")

    if start_idx >= 200:
        log.info("AudioGen already complete.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading AudioGen model on {device}...")
    model = AudioGen.get_pretrained("facebook/audiogen-medium")
    model.set_generation_params(duration=5)

    BATCH = 4
    for i in range(start_idx, len(PROMPTS), BATCH):
        batch = PROMPTS[i:i + BATCH]
        with torch.no_grad():
            wavs = model.generate(batch)
        for j, wav in enumerate(wavs):
            idx = i + j
            p = out_dir / f"audiogen_{idx:04d}.wav"
            wav_16k = torchaudio.functional.resample(wav.cpu(), model.sample_rate, 16000)
            torchaudio.save(str(p), wav_16k, 16000)
        done = min(i + BATCH, 200)
        log.info(f"  {done}/200 clips")
        if i % 40 == 0:
            torch.cuda.empty_cache()

    final = list(out_dir.glob("audiogen_*.wav"))
    log.info(f"AudioGen generation done: {len(final)} clips")

    # Register in manifest
    existing_rows, existing_ids = _load_manifest()
    new_rows = []
    for wav in sorted(final):
        sample_id = f"audiogen_{wav.stem}"
        if sample_id in existing_ids:
            continue
        new_rows.append({
            "sample_id":      sample_id,
            "file_path":      str(wav).replace("\\", "/"),
            "label":          1.0,
            "domain":         "non_human",
            "source_dataset": "audiogen",
            "generator":      "audiogen",
            "split":          "train",
        })
    if new_rows:
        _save_manifest(existing_rows + new_rows)
        log.info(f"AudioGen: registered {len(new_rows)} files in manifest")
    else:
        log.info("AudioGen already fully registered in manifest.")


# ── ESC-50 ───────────────────────────────────────────────────────────────────

def fetch_esc50():
    # Clone to Drive so it doesn't fill C: — it's ~600 MB of audio
    esc50_dir = RAW / "non_human" / "real" / "ESC-50"
    out_dir = RAW / "non_human" / "real" / "esc50_processed"

    # Check manifest first
    existing_rows, existing_ids = _load_manifest()
    esc50_in_manifest = sum(1 for r in existing_rows if r.get("source_dataset") == "esc50")
    if esc50_in_manifest >= 1400:
        log.info(f"ESC-50 already in manifest: {esc50_in_manifest} entries. Nothing to do.")
        return

    existing = list(out_dir.glob("*.wav")) if out_dir.exists() else []
    if len(existing) < 1400:
        if not esc50_dir.exists():
            log.info("Installing git-lfs and cloning ESC-50 to Drive...")
            os.system("git lfs install")
            esc50_dir.parent.mkdir(parents=True, exist_ok=True)
            ret = os.system(f'git lfs clone https://github.com/karolpiczak/ESC-50.git "{esc50_dir}"')
            if ret != 0:
                raise RuntimeError("ESC-50 clone failed.")
        else:
            log.info("ESC-50 already cloned.")

        log.info("Ingesting ESC-50...")
        audioset = _load_ingestor("audioset")
        n = audioset.ingest_esc50(str(esc50_dir), str(out_dir), copy=True, non_human_only=True)
        log.info(f"ESC-50 ingest done: {n} clips -> {out_dir}")
        existing = list(out_dir.glob("*.wav"))

    # Register in manifest
    new_rows = []
    for wav in sorted(existing):
        sample_id = f"esc50_{wav.stem}"
        if sample_id in existing_ids:
            continue
        new_rows.append({
            "sample_id":      sample_id,
            "file_path":      str(wav).replace("\\", "/"),
            "label":          0.0,
            "domain":         "non_human",
            "source_dataset": "esc50",
            "generator":      "real",
            "split":          "train",
        })

    if not new_rows:
        log.info("ESC-50 already fully registered in manifest.")
        return

    _save_manifest(existing_rows + new_rows)
    log.info(f"ESC-50 done: registered {len(new_rows)} files in manifest")


# ── Status check ─────────────────────────────────────────────────────────────

def status():
    checks = {
        "voice/real/ljspeech":       (RAW / "voice" / "real" / "ljspeech",         13000, "*.wav"),
        "voice/fake/speecht5":        (RAW / "voice" / "fake" / "speecht5",          1000, "speecht5_*.wav"),
        "non_human/real/esc50":       (RAW / "non_human" / "real" / "esc50_processed", 1400, "*.wav"),
        "non_human/fake/audiogen":    (RAW / "non_human" / "fake" / "audiogen",       200, "audiogen_*.wav"),
    }
    print("\n=== Data Status ===")
    for name, (path, target, pattern) in checks.items():
        n = len(list(path.glob(pattern))) if path.exists() else 0
        bar = "#" * int(n / target * 20) + "-" * (20 - int(n / target * 20))
        status = "DONE" if n >= target else f"{n}/{target}"
        print(f"  [{bar}] {status:>12}  {name}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

COMMANDS = {
    "ljspeech": fetch_ljspeech,
    "speecht5": fetch_speecht5,
    "audiogen":  fetch_audiogen,
    "esc50":     fetch_esc50,
    "status":    status,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=list(COMMANDS.keys()))
    args = parser.parse_args()
    COMMANDS[args.dataset]()
