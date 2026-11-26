"""Paired ReCA/T2A derivatives and endpoint displacements on a fixed CPU probe."""
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch

from audio_omni.loader import load_model_bundle
from reca_audio.train.conditioning import build_reca_metadata, build_t2a_metadata
from reca_audio.train.data import AudioManifestDataset
from reca_audio.train.flow import rectified_flow_loss


REFERENCE = Path("runs/audits/audiocaps_mix_pool4_eval_v1/results.json")
OUT = Path("runs/audits/module_gradients_low_lr_v1")


def write(name, value):
    dest = OUT / name
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, allow_nan=False, indent=2))
    tmp.replace(dest)


def group(name):
    parts = name.split(".")
    if parts[1:3] == ["transformer", "layers"]:
        kind = {"pre_norm": "layer_norm", "ff_norm": "layer_norm", "cross_attend_norm": "layer_norm",
                "self_attn": "self_attention", "cross_attn": "cross_attention", "ff": "ffn"}[parts[4]]
        return f"block{int(parts[3]):02d}/{kind}"
    if parts[1] in {"to_timestep_embed", "timestep_features"}:
        return "timestep_embedding"
    if parts[1] in {"to_cond_embed", "to_global_embed"}:
        return parts[1]
    if parts[1] in {"preprocess_conv", "postprocess_conv"} or parts[1:3] in (
            ["transformer", "project_in"], ["transformer", "project_out"]):
        return "input_output"
    raise ValueError(f"Unclassified parameter {name}")


def square_norm(x):
    return float(torch.linalg.vector_norm(x)) ** 2


def main():
    torch.manual_seed(42)
    raw = REFERENCE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != "615f17aff936c843e3b38cefc90ac9be3ea16f604135e70654861785d705d2fa":
        raise ValueError("Reference protocol changed")
    protocol = json.loads(raw)["protocol"]
    OUT.mkdir(exist_ok=False)
    config = json.loads(Path("Audio-Omni/model/Audio-Omni.json").read_text())
    base = protocol["checkpoints"]["base"]["path"]
    model, *_ = load_model_bundle(config, base, device="cpu", dtype="float32")
    model.eval().requires_grad_(False)
    dit = model.model.requires_grad_(True)
    named = dict(dit.named_parameters())
    keys = {n: group(n) for n in named}
    original = torch.load(base, mmap=True, map_location="cpu", weights_only=True)["state_dict"]
    arms = list(protocol["checkpoints"])
    write("protocol.json", dict(reference=str(REFERENCE), checkpoints=protocol["checkpoints"],
                                 parameters={n: [keys[n], p.numel()] for n, p in named.items()},
                                 deploy=Path("DEPLOYED_COMMIT").read_text(), dtype="float32", device="cpu",
                                 pool_width=1, cfg_dropout=0, flow_seed_start=300042))
    drift = {}
    for arm in arms:
        updated = (torch.load(protocol["checkpoints"][arm]["path"], mmap=True, map_location="cpu",
                              weights_only=True)["dit"] if arm != "base" else None)
        totals = defaultdict(lambda: dict(numel=0, base_sq=0., delta_sq=0.))
        for n, p in named.items():
            row = totals[keys[n]]
            before = original["model." + n].float()
            row["numel"] += p.numel()
            row["base_sq"] += square_norm(before)
            if updated is not None:
                row["delta_sq"] += square_norm(updated[n].float() - before)
        for v in totals.values():
            v["relative_l2"] = math.sqrt(v["delta_sq"]/v["base_sq"]) if v["base_sq"] else None
        drift[arm] = totals
        del updated
    write("drift.json", drift)
    dataset = AudioManifestDataset(protocol["test_manifest"], seconds=10)
    items = [r for r in protocol["audio"] if r["exploratory"]][:4]
    assert len({r["sha"] for r in items}) == 4
    probes = []
    with torch.no_grad():
        for i, item in enumerate(items):
            random.seed(100042+i)
            clip = dataset[item["row"]]
            assert clip.caption == item["prompt"]
            torch.manual_seed(400042+i)
            latent = model.pretransform.encode(clip.audio.unsqueeze(0))
            metas = [build_reca_metadata(clip, "cpu", omni_downsample=4, prompt_rng=random.Random(200042+i)),
                     build_t2a_metadata(clip, "cpu")]
            cond = [model.get_conditioning_inputs(model.conditioner([m], "cpu")) for m in metas]
            probes.append((latent,cond))
    write("samples.json", items)
    for arm in arms:
        if arm != "base":
            state = torch.load(protocol["checkpoints"][arm]["path"], mmap=True, map_location="cpu",
                               weights_only=True)["dit"]
            dit.load_state_dict(state, strict=True)
            del state
        records = []
        for i, (z, conditions) in enumerate(probes):
            grads, stats = [], []
            for cond in conditions:
                gen = torch.Generator().manual_seed(300042+i)
                loss, detail = rectified_flow_loss(dit, z, cond, cfg_dropout_prob=0., generator=gen)
                grads.append(torch.autograd.grad(loss, tuple(named.values()), allow_unused=True))
                stats.append(detail)
                del loss
            assert stats[0]["t_mean"] == stats[1]["t_mean"]
            totals = defaultdict(lambda: dict(reca_sq=0., t2a_sq=0., dot=0., reca_unused=0, t2a_unused=0))
            for n, a, b in zip(named, *grads):
                v = totals[keys[n]]
                for g, task in [(a,"reca"), (b,"t2a")]:
                    if g is None:
                        v[task+"_unused"] += named[n].numel()
                    else:
                        v[task+"_sq"] += square_norm(g)
                if a is not None and b is not None:
                    v["dot"] += float(torch.dot(a.flatten(), b.flatten()))
            for v in totals.values():
                scale = math.sqrt(v["reca_sq"]*v["t2a_sq"])
                v["cosine"] = v["dot"]/scale if scale else None
            records.append(dict(sha=items[i]["sha"], flow=stats, groups=totals))
            write(arm+".json", records)
            del grads, a, b, g
            print(arm, i+1, flush=True)
    (OUT/"COMPLETE").touch()


if __name__ == "__main__":
    main()
