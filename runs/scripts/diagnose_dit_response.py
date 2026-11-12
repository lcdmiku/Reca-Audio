"""Compare DiT responses on shared held-out latents, noise and conditions."""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from audio_omni.loader import load_model_bundle
from diagnose_condition_alignment import _load_checkpoint, select_groups
from reca_audio.train.conditioning import build_reca_metadata, build_t2a_metadata
from reca_audio.train.data import AudioManifestDataset


def relative_delta(x, y):
    return ((x - y).norm() / y.norm().clamp_min(1e-8)).item()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--swap", action="store_true", help="Compare Base/Mix weight swaps")
    args = parser.parse_args()
    if args.groups < 2 or args.out.exists():
        parser.error("need at least two groups and a fresh output path")
    random.seed(42)
    if args.swap:
        torch.manual_seed(42)
    dataset = AudioManifestDataset(args.manifest, seconds=10)
    groups = select_groups(dataset, args.groups, 1)
    if len(groups) != args.groups:
        raise ValueError("insufficient unique audio groups")
    config = json.loads(Path("Audio-Omni/model/Audio-Omni.json").read_text())
    bundle, _, _, _ = load_model_bundle(
        config, "Audio-Omni/model/model.ckpt", device="cuda:0",
        conditioner_device="cuda:1", dtype="float16",
    )
    dit = bundle.model
    dtype = next(dit.parameters()).dtype
    layers = dit.model.transformer.layers
    generator = torch.Generator(device="cuda:0").manual_seed(42)
    inputs, audio_conditions, text_conditions = [], [], []
    for index, group in enumerate(groups):
        clip = dataset[group["row_index"]]
        latent = bundle.pretransform.encode(clip.audio[None].to("cuda:0")).float()
        noise = torch.randn(latent.shape, device=latent.device, generator=generator)
        inputs.append((latent, noise))
        for metadata, storage in (
            (build_reca_metadata(clip, "cuda:1", omni_downsample=4,
                                 prompt_rng=random.Random(42)), audio_conditions),
            (build_t2a_metadata(clip, "cuda:1"), text_conditions),
        ):
            condition = bundle.get_conditioning_inputs(bundle.conditioner([metadata], "cuda:1"))
            storage.append({
                key: value.to("cuda:0", dtype=dtype if value.is_floating_point() else value.dtype)
                if value is not None else None for key, value in condition.items()
            })
    baseline, rows = {}, []
    capture = {}

    def hook(key):
        def save(_, __, output):
            capture[key] = output.detach().float().mean(dim=1).cpu().flatten()
        return save

    handles = []
    for index, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(hook(f"block:{index}")))
        handles.append(layer.cross_attn.register_forward_hook(hook(f"attention:{index}")))
    try:
        arms = ("base", "pure", "mix")
        if args.swap:
            arms = ("base", "base_mix_embed", "mix", "mix_base_embed",
                    "mix_base_early_attention", "mix_base_embed_early_attention")
            mix_state = _load_checkpoint(args.train_root / "mix" / "dit_exposure2000.pt")["dit"]
            modules = {"embed": dit.model.to_cond_embed}
            # Include the pre-attention normalization in each restored branch.
            for i, layer in enumerate(layers[:6]):
                modules[f"attention:{i}"] = layer.cross_attn
                modules[f"norm:{i}"] = layer.cross_attend_norm
            paths = {id(module): name for name, module in dit.named_modules()}
            base_parts = {
                name: {key: value.cpu().clone() for key, value in module.state_dict().items()}
                for name, module in modules.items()
            }
        for arm in arms:
            if args.swap:
                if arm == "mix":
                    dit.load_state_dict(mix_state, strict=True)
                elif arm != "base":
                    for name, module in modules.items():
                        if arm == "base_mix_embed" and name != "embed":
                            continue
                        restore = (name == "embed" and "base_embed" in arm) or (
                            name != "embed" and "early_attention" in arm
                        )
                        if restore:
                            state = base_parts[name]
                        else:
                            prefix = paths[id(module)] + "."
                            state = {key: mix_state[prefix + key] for key in base_parts[name]}
                        module.load_state_dict(state, strict=True)
            elif arm != "base":
                payload = _load_checkpoint(args.train_root / arm / "dit_exposure2000.pt")
                dit.load_state_dict(payload["dit"], strict=True)
                del payload
            for index, (latent, noise) in enumerate(inputs):
                negative = (index + 1) % len(inputs)
                for time in (0.2, 0.5, 0.8):
                    t = torch.full((1,), time, device=latent.device, dtype=dtype)
                    x = ((1 - time) * latent + time * noise).to(dtype)
                    views, errors = {}, {}
                    for name, condition in (
                        ("audio", audio_conditions[index]),
                        ("text", text_conditions[index]),
                        ("shuffled", text_conditions[negative]),
                    ):
                        capture.clear()
                        velocity = dit(x, t, **condition, cfg_scale=1.0).float()
                        errors[name] = F.mse_loss(velocity, noise - latent).item()
                        views[name] = dict(capture, velocity=velocity.cpu().flatten())
                    if len(views["text"]) != 2 * len(layers) + 1:
                        raise RuntimeError("missing layer captures")
                    key = (index, time)
                    if arm == "base":
                        baseline[key] = views["text"]
                    rows.append({
                        "arm": arm, "sha": groups[index]["group_id"],
                        "negative_sha": groups[negative]["group_id"], "t": time,
                        "flow_mse": errors,
                        "stages": {
                            stage: {
                                "audio_text_cosine": F.cosine_similarity(views["audio"][stage], text, dim=0).item(),
                                "caption_sensitivity": relative_delta(views["shuffled"][stage], text),
                                "text_drift_from_base": relative_delta(text, baseline[key][stage]),
                            } for stage, text in views["text"].items()
                        },
                    })
            print(f"completed {arm}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump({"seed": 42, "swap": args.swap, "early_attention_layers": list(range(6)) if args.swap else [],
                   "groups": groups, "rows": rows}, handle, allow_nan=False)


if __name__ == "__main__":
    main()
