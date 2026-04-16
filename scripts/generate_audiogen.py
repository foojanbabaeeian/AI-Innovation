"""
Generate AI non-human sound clips with AudioLDM2 (via diffusers).

AudioLDM2 is a reliable, well-supported text-to-audio model for environmental
sounds / sound effects. We use it instead of Meta's AudioGen because AudioGen
is only distributed through `audiocraft`, which builds flakily on Colab.

Prereqs on Colab (if not already installed):
    !pip install -q diffusers accelerate scipy soundfile

Usage:
    python scripts/generate_audiogen.py \\
        --out-dir   /content/drive/MyDrive/AI-Innovation-Data/raw/non_human/fake/audioldm2 \\
        --n-clips   1000 \\
        --duration  5 \\
        --model     cvssp/audioldm2

Notes:
  - AudioLDM2 natively outputs 16 kHz. preprocess_segments.py will resample if
    needed for the non_human domain (44.1 kHz target).
  - Prompts cover ESC-50 style environmental sound categories.
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
    p.add_argument("--duration",  type=float, default=5.0, help="Seconds per clip")
    p.add_argument("--model",     default="cvssp/audioldm2")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-inference-steps", type=int, default=50,
                   help="AudioLDM2 denoise steps. 50 = good quality; 25 = faster")
    p.add_argument("--guidance-scale", type=float, default=3.5)
    p.add_argument("--seed",      type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import scipy.io.wavfile
    import torch
    from diffusers import AudioLDM2Pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32
    log.info("Device: %s  dtype: %s", device, dtype)
    log.info("Loading %s ...", args.model)
    pipe = AudioLDM2Pipeline.from_pretrained(args.model, torch_dtype=dtype).to(device)
    # disable the NSFW-style safety checker that exists for AudioLDM2 — it flags legit SFX
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None

    generator = torch.Generator(device=device).manual_seed(args.seed)

    prompts = build_prompts(args.n_clips, seed=args.seed)
    log.info("Generating %d clips at %ss each", len(prompts), args.duration)
    sr = 16_000  # AudioLDM2 native output

    done = 0
    skipped = 0
    for batch_start in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[batch_start : batch_start + args.batch_size]
        batch_ids = [f"audioldm2_{batch_start + i:05d}" for i in range(len(batch_prompts))]

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
            result = pipe(
                pending_prompts,
                num_inference_steps=args.num_inference_steps,
                audio_length_in_s=args.duration,
                guidance_scale=args.guidance_scale,
                generator=generator,
            )
        audios = result.audios  # shape (batch, n_samples) or list of arrays

        for pid, audio in zip(pending_ids, audios):
            audio = np.asarray(audio).squeeze()
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
             "        --dir %s --domain non_human --generator audioldm2 --source audioldm2",
             out_dir)


if __name__ == "__main__":
    main()
