#!/usr/bin/env python
"""Generate and score the fixed held-out reconstruction diagnostic."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SAMPLE_RATE = 44100
TARGET_SECONDS = 5


def generate(out: Path, label: str, checkpoint: Path) -> None:
    import soundfile as sf

    sys.path.insert(0, str(ROOT / "Audio-Omni"))
    from audio_omni.api import AudioOmni

    rows = json.loads((out / "items.json").read_text(encoding="utf-8"))
    target = out / label
    target.mkdir()
    model = AudioOmni(
        str(ROOT / "Audio-Omni/model/Audio-Omni.json"),
        str(checkpoint),
        device="cuda:0",
        conditioner_device="cuda:1",
        dtype="float16",
    )
    model.sample_size = SAMPLE_RATE * TARGET_SECONDS
    manifest = []
    for index, row in enumerate(rows):
        audio = model.reconstruct(
            row["input"],
            steps=100,
            cfg_scale=1.0,
            seconds_total=10,
            seed=1234 + index,
        )
        destination = target / f"{row['id']}.wav"
        sf.write(
            destination,
            audio.float().cpu().numpy().T,
            model.sample_rate,
            subtype="PCM_16",
        )
        manifest.append(
            {**row, "out": str(destination), "seed": 1234 + index, "steps": 100, "cfg": 1.0}
        )
        print(label, index + 1, len(rows), flush=True)
    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def score(out: Path, base_dir: Path, labels: list[str]) -> None:
    sys.path.insert(0, str(ROOT / "runs/Benchmark"))
    from common import metrics

    rows = json.loads((out / "items.json").read_text(encoding="utf-8"))
    references = [row["ref"] for row in rows]
    reference_embeddings = metrics.clap_audio_embeddings(references, "cuda:0")
    reference_norm = reference_embeddings / (
        np.linalg.norm(reference_embeddings, axis=1, keepdims=True) + 1e-8
    )
    directories = {"base": base_dir, **{label: out / label for label in labels}}
    results = {}
    for label, directory in directories.items():
        generated = [str(directory / f"{row['id']}.wav") for row in rows]
        embeddings = metrics.clap_audio_embeddings(generated, "cuda:0")
        normalized = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        cosine = np.sum(normalized * reference_norm, axis=1)
        results[label] = {
            "n": len(generated),
            "clap_pair_cosine_mean": float(cosine.mean()),
            "clap_pair_cosine_std": float(cosine.std()),
            "fad_clap": metrics.frechet_distance(embeddings, reference_embeddings),
            "lsd": metrics.lsd(generated, references),
        }
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["generate", "score"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument("--base-dir", type=Path)
    parser.add_argument("--labels", nargs="*", default=[])
    args = parser.parse_args()
    if args.mode == "generate":
        if not args.label or args.ckpt is None:
            parser.error("generate requires --label and --ckpt")
        generate(args.out, args.label, args.ckpt)
    else:
        if args.base_dir is None:
            parser.error("score requires --base-dir")
        score(args.out, args.base_dir, args.labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
