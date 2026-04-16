"""
Generate AI music clips with Meta's MusicGen (via audiocraft) for the MUSIC/fake domain.

Designed to run on Colab with a T4 / A100 GPU. ~2,000 clips in ~2-3 hours on T4.

Prereqs on Colab:
    !pip install -q audiocraft soundfile

Usage:
    python scripts/generate_musicgen.py \\
        --out-dir   /content/drive/MyDrive/AI-Innovation-Data/raw/music/fake/musicgen \\
        --n-clips   2000 \\
        --duration  10 \\
        --model     facebook/musicgen-small

Notes:
  - We use `musicgen-small` by default (300M params) — ~5-10 s/clip on T4.
    `musicgen-medium` (1.5B) gives better quality but is 3x slower.
  - Prompts cover a diverse set of genres/moods/instruments to mirror the
    real-music distribution (MusicCaps is multi-genre, multi-era).
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
# 8 × 8 × 8 × 4 = 2,048 unique combinations — matches our 2,000-clip target.
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
    """Deterministically sample n unique prompts from the cartesian product."""
    rng = random.Random(seed)
    all_combos = [
        f"A {g} track, {m}, {i}, {d}."
        for g in GENRES for m in MOODS for i in INSTRUMENTS for d in DESCRIPTORS
    ]
    rng.shuffle(all_combos)
    if n > len(all_combos):
        # repeat the pool — rare, only if user asks for > 2,048
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

    # Lazy import so `--help` works without the heavy deps.
    import torch
    from audiocraft.models import MusicGen
    from audiocraft.data.audio import audio_write

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)
    log.info("Loading %s ...", args.model)
    model = MusicGen.get_pretrained(args.model, device=device)
    model.set_generation_params(duration=args.duration)

    prompts = build_prompts(args.n_clips, seed=args.seed)
    log.info("Generating %d clips at %ss each", len(prompts), args.duration)

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

        with torch.no_grad():
            wav_batch = model.generate(pending_prompts, progress=False)

        for pid, wav in zip(pending_ids, wav_batch):
            # audio_write strips the extension — pass path without .wav
            audio_write(
                str(out_dir / pid),
                wav.cpu(),
                model.sample_rate,
                strategy="loudness",
                loudness_compressor=True,
            )
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
