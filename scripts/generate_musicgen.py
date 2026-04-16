"""
Generate AI music clips with Meta's MusicGen, via HuggingFace transformers.

Uses `transformers.pipeline("text-to-audio", ...)` instead of the `audiocraft`
package because audiocraft's build often fails on Colab due to xformers/torch
version mismatches. transformers is pre-installed and always works.

Prereqs on Colab (if not already installed):
    !pip install -q transformers scipy soundfile

Usage:
    python scripts/generate_musicgen.py \\
        --out-dir   /content/drive/MyDrive/AI-Innovation-Data/raw/music/fake/musicgen \\
        --n-clips   2000 \\
        --duration  10 \\
        --model     facebook/musicgen-small

Notes:
  - MusicGen generates at 32 kHz. We save as-is (32 kHz WAV); preprocess_segments.py
    will resample to the target domain rate (44.1 kHz for music) automatically.
  - ~50 audio tokens = 1 second. So --duration 10 -> max_new_tokens=500.
  - Resumable: skips clips that already exist in out_dir.
"""

import argparse
import logging
import random
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Prompt pool: genre × mood × instrument × descriptor ──────────────────────
# 8 × 8 × 8 × 4 = 2,048 unique combinations
GENRES = [
    "pop", "rock", "jazz", "classical", "hip hop", "electronic", "folk", "country"
]
MOODS = [
    "upbeat and energetic", "melancholic and slow", "dark and cinematic",
    "bright and cheerful", "nostalgic and warm", "tense and dramatic",
    "dreamy and atmospheric", "aggressive and driving",
]
INSTRUMENTS = [
    "with acoustic guitar", "with electric piano", "with synthesizer lead",
    "with string orchestra", "with punchy drums", "with warm brass section",
    "with soft female vocals", "with distorted electric guitar",
]
DESCRIPTORS = [
    "studio-quality recording", "vintage analog sound",
    "modern production", "live concert feel",
]


def build_prompts(n: int, seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    all_combos = [
        f"A {g} track, {m}, {i}, {d}."
        for g in GENRES for m in MOODS for i in INSTRUMENTS for d in DESCRIPTORS
    ]
    rng.shuffle(all_combos)
    if n > len(all_combos):
        all_combos = (all_combos * ((n // len(all_combos)) + 1))[:n]
    return all_combos[:n]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir",   required=True, help="Directory to write .wav files")
    p.add_argument("--n-clips",   type=int, default=2000)
    p.add_argument("--duration",  type=float, default=10.0, help="Seconds per clip")
    p.add_argument("--model",     default="facebook/musicgen-small",
                   help="HF model id: musicgen-small / musicgen-medium / musicgen-large")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Prompts per batch (reduce if OOM).")
    p.add_argument("--seed",      type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import scipy.io.wavfile
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    dtype  = torch.float16 if device == 0 else torch.float32
    log.info("Device: %s  dtype: %s", "cuda:0" if device == 0 else "cpu", dtype)
    log.info("Loading %s ...", args.model)
    pipe = pipeline(
        "text-to-audio",
        model=args.model,
        device=device,
        torch_dtype=dtype,
    )
    # 50 audio tokens per second of output
    max_new_tokens = int(args.duration * 50)
    generate_kwargs = {
        "do_sample": True,
        "temperature": 1.0,
        "max_new_tokens": max_new_tokens,
    }

    prompts = build_prompts(args.n_clips, seed=args.seed)
    log.info("Generating %d clips at %ss each (max_new_tokens=%d)",
             len(prompts), args.duration, max_new_tokens)

    done = 0
    skipped = 0
    for batch_start in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[batch_start : batch_start + args.batch_size]
        batch_ids = [f"musicgen_{batch_start + i:05d}" for i in range(len(batch_prompts))]

        # Skip clips that already exist (resumability)
        pending_prompts, pending_ids = [], []
        for pid, prom in zip(batch_ids, batch_prompts):
            if (out_dir / f"{pid}.wav").exists():
                skipped += 1
            else:
                pending_prompts.append(prom)
                pending_ids.append(pid)

        if not pending_prompts:
            continue

        # The pipeline accepts a list of prompts; returns a list of dicts
        outputs = pipe(
            pending_prompts,
            batch_size=args.batch_size,
            generate_kwargs=generate_kwargs,
        )
        # outputs is a list (one per prompt); each is {"audio": ndarray, "sampling_rate": int}
        for pid, out in zip(pending_ids, outputs):
            audio = out["audio"]
            sr = out["sampling_rate"]
            # audio shape is (1, n_samples) or (n_channels, n_samples) — take first channel
            if audio.ndim == 3:
                audio = audio[0]
            if audio.ndim == 2:
                audio = audio[0]
            # Clip to [-1, 1] and convert to int16 for WAV
            audio = np.clip(audio, -1.0, 1.0)
            audio_i16 = (audio * 32767).astype(np.int16)
            scipy.io.wavfile.write(str(out_dir / f"{pid}.wav"), sr, audio_i16)
            done += 1

        if (batch_start // args.batch_size) % 25 == 0:
            log.info(
                "  %d generated / %d skipped / %d total (%.1f%%)",
                done, skipped, batch_start + len(batch_prompts),
                100 * (batch_start + len(batch_prompts)) / len(prompts),
            )

    log.info("Done: generated=%d  skipped=%d  out=%s", done, skipped, out_dir)
    log.info("Next: python scripts/register_generated.py \\\n"
             "        --dir %s --domain music --generator musicgen --source musicgen",
             out_dir)


if __name__ == "__main__":
    main()
