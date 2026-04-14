"""
Standalone dataset fetcher — run multiple instances in parallel terminals.

Usage (open 3 Anaconda Prompt windows, each with conda activate tf-gpu-210):

  Terminal 1 — finish LJSpeech + ingest:
    python scripts/fetch_dataset.py ljspeech

  Terminal 2 — download WaveFake melgan + ingest:
    python scripts/fetch_dataset.py wavefake

  Terminal 3 — generate AudioGen AI sounds (uses GPU):
    python scripts/fetch_dataset.py audiogen

  Terminal 4 — clone ESC-50 + ingest (if not done yet):
    python scripts/fetch_dataset.py esc50

All output goes to data/raw/ which is linked to Google Drive.
"""

import argparse
import logging
import os
import sys
import urllib.request
import tarfile
import zipfile
import requests
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

DRIVE_ROOT = Path("C:/Users/fooja/Google Drive Streaming/My Drive/AI-Innovation-Data")
RAW = DRIVE_ROOT / "raw"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── LJSpeech ──────────────────────────────────────────────────────────────────

def fetch_ljspeech():
    lj_dir = RAW / "voice" / "real" / "LJSpeech-1.1"
    lj_tar = RAW / "voice" / "real" / "LJSpeech-1.1.tar.bz2"
    wavs_dir = lj_dir / "wavs"

    existing = list(wavs_dir.glob("*.wav")) if wavs_dir.exists() else []
    log.info(f"LJSpeech: {len(existing)} wavs already present")

    if len(existing) >= 13000:
        log.info("LJSpeech already complete.")
    else:
        if not lj_dir.exists():
            if not lj_tar.exists():
                log.info("Downloading LJSpeech (~2.6 GB)...")
                lj_tar.parent.mkdir(parents=True, exist_ok=True)

                def _progress(count, block, total):
                    pct = min(count * block / total * 100, 100)
                    print(f"\r  {pct:.1f}%", end="", flush=True)

                urllib.request.urlretrieve(
                    "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
                    lj_tar,
                    reporthook=_progress,
                )
                print()
                log.info("Download complete.")
            log.info("Extracting LJSpeech...")
            with tarfile.open(lj_tar, "r:bz2") as tf:
                tf.extractall(lj_dir.parent)
            log.info("Extracted.")
        else:
            log.info("LJSpeech folder exists, skipping download.")

    log.info("Ingesting LJSpeech...")
    from src.data.ingestors.voice import ingest_ljspeech
    n = ingest_ljspeech(
        source_dir=str(lj_dir),
        output_dir=str(RAW / "voice"),
        copy=False,
    )
    log.info(f"LJSpeech done: {n} files ingested → {RAW}/voice/real/ljspeech/")


# ── WaveFake ──────────────────────────────────────────────────────────────────

def fetch_wavefake():
    out_dir = RAW / "voice" / "fake" / "wavefake"
    melgan_dir = out_dir / "ljspeech_melgan"
    melgan_zip = out_dir / "ljspeech_melgan.zip"
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = list(melgan_dir.glob("*.wav")) if melgan_dir.exists() else []
    if len(existing) >= 13000:
        log.info(f"WaveFake melgan already complete: {len(existing)} files")
    else:
        if not melgan_zip.exists():
            log.info("Querying Zenodo for WaveFake download URL...")
            resp = requests.get("https://zenodo.org/api/records/5642694", timeout=30)
            resp.raise_for_status()
            files = resp.json().get("files", [])

            url = None
            for f in files:
                key = f["key"].lower()
                if "melgan" in key and key.endswith(".zip") and "large" not in key and "multi" not in key:
                    url = f["links"]["self"]
                    size_gb = f.get("size", 0) / 1e9
                    log.info(f"Found: {f['key']} ({size_gb:.1f} GB)")
                    break

            if not url:
                log.error("Could not find melgan zip on Zenodo. Available files:")
                for f in files:
                    log.error(f"  {f['key']}  {f.get('size',0)/1e9:.1f} GB")
                return

            log.info(f"Downloading WaveFake melgan (~7 GB)...")
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                downloaded = 0
                with open(melgan_zip, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        fh.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded / total * 100
                            print(f"\r  {pct:.1f}%  ({downloaded/1e9:.2f}/{total/1e9:.2f} GB)", end="", flush=True)
            print()
            log.info("Download complete.")

        log.info("Extracting WaveFake melgan...")
        with zipfile.ZipFile(melgan_zip, "r") as zf:
            zf.extractall(out_dir)
        log.info("Extracted.")

    log.info("Ingesting WaveFake melgan...")
    from src.data.ingestors.voice import ingest_generic_audio
    n = ingest_generic_audio(
        source_dir=str(melgan_dir),
        output_dir=str(RAW / "voice" / "fake" / "wavefake_melgan"),
        label=1,
        source_name="wavefake_melgan",
        copy=False,
    )
    log.info(f"WaveFake done: {n} files ingested")


# ── AudioGen ──────────────────────────────────────────────────────────────────

def fetch_audiogen():
    import torch
    if not torch.cuda.is_available():
        log.warning("No GPU found — AudioGen will be very slow on CPU.")
    else:
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    try:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "audiocraft"], check=True)
    except Exception as e:
        log.error(f"audiocraft install failed: {e}")
        return

    import torchaudio
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
    ]
    PROMPTS = PROMPTS[:200]

    existing = list(out_dir.glob("audiogen_*.wav"))
    start_idx = len(existing)
    log.info(f"AudioGen: {start_idx}/200 already generated")

    if start_idx >= 200:
        log.info("AudioGen already complete.")
        return

    log.info("Loading AudioGen model...")
    model = AudioGen.get_pretrained("facebook/audiogen-medium")
    model.set_generation_params(duration=5)
    log.info("Model loaded. Generating...")

    BATCH = 4
    for i in range(start_idx, len(PROMPTS), BATCH):
        batch = PROMPTS[i:i + BATCH]
        with torch.no_grad():
            wavs = model.generate(batch)
        for j, wav in enumerate(wavs):
            idx = i + j
            path = out_dir / f"audiogen_{idx:04d}.wav"
            wav_16k = torchaudio.functional.resample(wav.cpu(), model.sample_rate, 16000)
            torchaudio.save(str(path), wav_16k, 16000)
        done = min(i + BATCH, 200)
        log.info(f"  {done}/200 clips generated")
        if i % 40 == 0:
            torch.cuda.empty_cache()

    final = len(list(out_dir.glob("audiogen_*.wav")))
    log.info(f"AudioGen done: {final} clips saved → {out_dir}")


# ── ESC-50 ────────────────────────────────────────────────────────────────────

def fetch_esc50():
    esc50_dir = Path("C:/tmp/ESC-50")
    out_dir = RAW / "non_human" / "real" / "esc50_processed"

    existing = list(out_dir.glob("*.wav")) if out_dir.exists() else []
    if len(existing) >= 1400:
        log.info(f"ESC-50 already ingested: {len(existing)} files. Nothing to do.")
        return

    if not esc50_dir.exists():
        log.info("Installing git-lfs and cloning ESC-50...")
        os.system("git lfs install")
        ret = os.system(f"git lfs clone https://github.com/karolpiczak/ESC-50.git {esc50_dir}")
        if ret != 0:
            raise RuntimeError("ESC-50 clone failed.")
    else:
        log.info("ESC-50 already cloned.")

    log.info("Ingesting ESC-50...")
    from src.data.ingestors.audioset import ingest_esc50
    n = ingest_esc50(str(esc50_dir), str(out_dir), copy=False, non_human_only=True)
    log.info(f"ESC-50 done: {n} clips ingested")


# ── Main ──────────────────────────────────────────────────────────────────────

COMMANDS = {
    "ljspeech": fetch_ljspeech,
    "wavefake": fetch_wavefake,
    "audiogen": fetch_audiogen,
    "esc50": fetch_esc50,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch individual datasets in parallel terminals")
    parser.add_argument("dataset", choices=list(COMMANDS.keys()),
                        help="Which dataset to fetch")
    args = parser.parse_args()
    COMMANDS[args.dataset]()
