"""
Generate AI non-human sound clips with Meta's AudioGen (via audiocraft).

Designed to run on Colab with a T4 / A100 GPU. ~1,000 clips in ~1-2 hours on T4.

Prereqs on Colab:
    !pip install -q audiocraft soundfile

Usage:
    python scripts/generate_audiogen.py \\
        --out-dir   /content/drive/MyDrive/AI-Innovation-Data/raw/non_human/fake/audiogen_v2 \\
        --n-clips   1000 \\
        --duration  5 \\
        --model     facebook/audiogen-medium

Notes:
  - Prompts cover ESC-50 style environmental sound categories so the fake set
    covers the same distribution as our real (ESC-50) data.
  - AudioGen-medium (1.5B) is the only public size; runs fine on a T4.
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

# ── Prompt pool: 50 ESC-50 classes × ~20 paraphrases each = ~1000 prompts ────
ESC50_CLASSES = [
    "dog barking", "rooster crowing", "pig grunting", "cow mooing", "frog croaking",
    "cat meowing", "hen clucking", "insects buzzing", "sheep bleating", "crow cawing",
    "rain falling", "sea waves", "crackling fire", "crickets chirping", "birds chirping",
    "water drops", "wind blowing", "pouring water", "toilet flushing", "thunderstorm",
    "baby crying", "sneezing", "clapping", "breathing", "coughing",
    "footsteps", "laughter", "brushing teeth", "snoring", "drinking water",
    "door knocking", "mouse clicking", "keyboard typing", "door creaking", "canned opening",
    "washing machine", "vacuum cleaner", "clock alarm", "clock ticking", "glass breaking",
    "helicopter flying", "chainsaw running", "siren wailing", "car horn honking",
    "engine revving", "train passing", "church bells", "airplane taking off",
    "fireworks exploding", "hand sawing",
]

PARAPHRASE_TEMPLATES = [
    "{c}",
    "The sound of {c}",
    "Recording of {c}",
    "A short clip of {c}",
    "Loud {c}",
    "Quiet {c}",
    "{c} outdoors",
    "{c} indoors",
    "{c} in a small room",
    "{c} in a large hall",
    "High-quality recording of {c}",
    "Field recording: {c}",
    "{c}, close microphone",
    "{c}, distant microphone",
    "{c} at night",
    "{c} in the morning",
    "{c} with some background noise",
    "{c}, clear and isolated",
    "Several seconds of {c}",
    "{c}, ambient recording",
]


def build_prompts(n: int, seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    all_prompts = [t.format(c=c) for c in ESC50_CLASSES for t in PARAPHRASE_TEMPLATES]
    rng.shuffle(all_prompts)
    if n > len(all_prompts):
        all_prompts = (all_prompts * ((n // len(all_prompts)) + 1))[:n]
    return all_prompts[:n]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir",   required=True)
    p.add_argument("--n-clips",   type=int, default=1000)
    p.add_argument("--duration",  type=float, default=5.0)
    p.add_argument("--model",     default="facebook/audiogen-medium")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed",      type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from audiocraft.models import AudioGen
    from audiocraft.data.audio import audio_write

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)
    log.info("Loading %s ...", args.model)
    model = AudioGen.get_pretrained(args.model, device=device)
    model.set_generation_params(duration=args.duration)

    prompts = build_prompts(args.n_clips, seed=args.seed)
    log.info("Generating %d clips at %ss each", len(prompts), args.duration)

    done = 0
    skipped = 0
    for batch_start in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[batch_start : batch_start + args.batch_size]
        batch_ids = [f"audiogen_v2_{batch_start + i:05d}" for i in range(len(batch_prompts))]

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
             "        --dir %s --domain non_human --generator audiogen_v2 --source audiogen_v2",
             out_dir)


if __name__ == "__main__":
    main()
