"""Paired T2A generation after in-memory Base/Mix block restoration."""

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from audio_omni.api import AudioOmni
from diagnose_condition_alignment import _load_checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Benchmark"))
from common.audio_io import save_audio
from common.metrics import clap_audio_embeddings, get_clap


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--mix", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("refusing existing output directory")
    groups = json.loads(args.selection.read_text())["groups"]
    if len(groups) != 8 or len({g["group_id"] for g in groups}) != 8:
        parser.error("protocol requires the previous eight unique SHA groups")
    prompts = [g["captions"][0] for g in groups]
    if not all(prompts):
        parser.error("empty caption")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    args.out.mkdir(parents=True)
    model = AudioOmni("Audio-Omni/model/Audio-Omni.json", "Audio-Omni/model/model.ckpt",
                      device="cuda:0", conditioner_device="cuda:1", dtype="float16")
    # Use a nominal 10-second latent window; record VAE rounding explicitly.
    original_sample_size = model.sample_size
    model.sample_size = model.sample_rate * 10
    dit = model.model.model
    layers = dit.model.transformer.layers
    if len(layers) != 36:
        raise ValueError("expected 36 DiT blocks")
    base_blocks = [{k: v.cpu().clone() for k, v in layer.state_dict().items()} for layer in layers]
    mix = _load_checkpoint(args.mix)["dit"]
    arms = {"base": None, "mix": None, "mix_base_middle": (12, 24), "mix_base_all": (0, 36)}
    contract = dict(selection=str(args.selection), mix=str(args.mix), groups=groups,
                    arms=arms, steps=100, cfg=6.0, seconds=10, seed_base=1234,
                    dtype="float16", original_sample_size=original_sample_size,
                    sample_size=model.sample_size, sample_rate=model.sample_rate,
                    downsampling_ratio=model.model.pretransform.downsampling_ratio)
    (args.out / "contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    rows = []
    with (args.out / "manifest.jsonl").open("x", encoding="utf-8") as manifest:
        for arm, span in arms.items():
            if arm != "base":
                dit.load_state_dict(mix, strict=True)
                if span:
                    for i in range(*span):
                        layers[i].load_state_dict(base_blocks[i], strict=True)
            (args.out / arm).mkdir()
            for i, (group, prompt) in enumerate(zip(groups, prompts)):
                start = time.monotonic()
                seed = 1235 + i
                audio = model.generate("T2A", prompt=prompt, steps=100,
                                       cfg_scale=6.0, seconds_total=10, seed=seed)
                if not torch.isfinite(audio.float()).all() or not audio.any():
                    raise RuntimeError(f"invalid audio: {arm}/{i}")
                path = args.out / arm / f"{i:02d}.wav"
                save_audio(str(path), audio, model.sample_rate)
                info = sf.info(path)
                row = dict(arm=arm, index=i, sha=group["group_id"], prompt=prompt,
                           seed=seed, out=str(path), duration=info.duration,
                           frames=info.frames, sample_rate=info.samplerate,
                           generation_seconds=time.monotonic() - start)
                rows.append(row)
                manifest.write(json.dumps(row) + "\n")
                manifest.flush()
                print(f"generated {arm} {i+1}/8 in {row['generation_seconds']:.1f}s", flush=True)
    del model, dit, layers, base_blocks, mix, audio
    gc.collect()
    torch.cuda.empty_cache()
    clap = get_clap("cuda:0")
    text = np.asarray(clap.get_text_embedding(prompts, use_tensor=False), dtype=np.float64)
    text /= np.linalg.norm(text, axis=1, keepdims=True) + 1e-8
    summary = {}
    for arm in arms:
        arm_rows = [r for r in rows if r["arm"] == arm]
        # Reset preprocessing RNG as well, keeping CLAP crops paired across arms.
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        embeddings = clap_audio_embeddings([r["out"] for r in arm_rows], "cuda:0")
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        for i, row in enumerate(arm_rows):
            row["clap"] = float(embeddings[i] @ text[i])
            row["shuffled_clap"] = float(embeddings[i] @ text[(i + 1) % len(text)])
        scores = np.array([r["clap"] for r in arm_rows])
        summary[arm] = dict(mean_clap=float(scores.mean()), per_sample=scores.tolist(),
                            mean_shuffled_clap=float(np.mean([r["shuffled_clap"] for r in arm_rows])))
    baseline = np.array(summary["base"]["per_sample"])
    mixed = np.array(summary["mix"]["per_sample"])
    for arm, values in summary.items():
        scores = np.array(values["per_sample"])
        values.update(delta_base=float((scores-baseline).mean()), delta_mix=float((scores-mixed).mean()),
                      wins_vs_mix=int((scores > mixed).sum()))
    (args.out / "results.json").write_text(json.dumps(dict(contract=contract, rows=rows, summary=summary),
                                                       indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
