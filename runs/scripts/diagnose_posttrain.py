"""Paired four-checkpoint evaluation. Run from the repository root."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from audio_omni.api import AudioOmni
from audio_omni.inference.generation import generate_diffusion_cond
from audio_omni.prompts import build_prompt
from diagnose_condition_alignment import _load_checkpoint
from reca_audio.train.conditioning import build_reca_metadata
from reca_audio.train.data import AudioManifestDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Benchmark"))
from common import audio_io, metrics

ARMS = ("base", "pure", "t2a", "mix")
BASE = Path("Audio-Omni/model/model.ckpt")
CONFIG = Path("Audio-Omni/model/Audio-Omni.json")
TRAIN = Path("runs/train/reca_long_matrix_b_10s_v1")
DATA = Path("runs/Benchmark/_data")
HIST = Path("runs/Benchmark/output/LongMatrixB_10s_eval_v1")
HIST_BASE = Path("runs/Benchmark/output/Matrix_v1_eval3_6gpu_9ce3984")
SETTINGS = {
    "audio_10_cfg6": ("audio", 441000, 6),
    "audio_11_cfg6": ("audio", 485100, 6),
    "audio_10_cfg1": ("audio", 441000, 1),
    "audio_10_cfg2": ("audio", 441000, 2),
    "audio_10_cfg4": ("audio", 441000, 4),
    "music_10_cfg6": ("music", 441000, 6),
    "reca_matched": ("reca", 441000, 1),
    "reca_mismatched": ("reca", 441000, 1),
}


def sha(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def prepare(root, smoke):
    if root.exists():
        raise FileExistsError(root)
    test = DATA / "audiocaps/reca_split_v1/test_manifest.jsonl"
    dataset = AudioManifestDataset(str(test), seconds=10)
    exploratory = json.loads(Path("runs/audits/dit_block_swap_v1/results.json").read_text())
    prior = {row["group_id"] for row in exploratory["groups"]}
    audio = {}
    hashes = {}
    for index, row in enumerate(dataset.rows):
        path = row["audio"]
        if path not in hashes:
            hashes[path] = sha(path)
        digest = hashes[path]
        audio.setdefault(digest, dict(sha=digest, ref=path, row=index,
                                     prompt=row["caption"], exploratory=digest in prior))
    if len(audio) != 89 or len(dataset) != 445 or not prior <= audio.keys():
        raise ValueError("held-out selection does not match the audited split")
    audio = sorted(audio.values(), key=lambda row: (row["exploratory"], row["sha"]))
    music_paths = {p.stem: p for p in audio_io.list_audio(DATA / "musiccaps/ref")}
    music_rows = audio_io.read_jsonl(DATA / "musiccaps/prompts1000.jsonl")
    random.Random(42).shuffle(music_rows)
    music = {}
    for row in music_rows:
        path = music_paths[row["id"]]
        digest = sha(path)
        music.setdefault(digest, dict(sha=digest, ref=str(path), prompt=row["prompt"], exploratory=False))
        if len(music) == 89:
            break
    if len(music) != 89:
        raise ValueError("insufficient MusicCaps references")
    root.mkdir(parents=True)
    checkpoints = {arm: BASE if arm == "base" else TRAIN / arm / "dit_exposure2000.pt" for arm in ARMS}
    if smoke:
        provenance = {arm: dict(path=str(path)) for arm, path in checkpoints.items()}
    else:
        provenance = {}
        for arm, path in checkpoints.items():
            provenance[arm] = dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)
            if arm != "base":
                schedule = TRAIN / f"{arm}_schedule.jsonl"
                records = audio_io.read_jsonl(schedule)
                provenance[arm].update(task_counts=dict(Counter(row["task"] for row in records)),
                                       schedule_sha256=sha(schedule),
                                       contract=json.loads((TRAIN / arm / "contract.json").read_text()))
    history = {}
    for domain in ("AudioCapsHeldout", "MusicCaps"):
        for arm in ARMS:
            directory = (HIST_BASE if arm == "base" else HIST) / domain / arm
            files = audio_io.read_jsonl(directory / "manifest.jsonl")
            info = sf.info(files[0]["out"])
            history[f"{domain}/{arm}"] = dict(metrics=json.loads((directory / "results.json").read_text()),
                                             first_frames=info.frames, sample_rate=info.samplerate,
                                             path=str(directory))
    write_json(root / "protocol.json", dict(
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        deployed_commit=Path("DEPLOYED_COMMIT").read_text().strip(),
        smoke=smoke, steps=2 if smoke else 100, dtype="float16", sample_rate=44100,
        seed=1234, settings=SETTINGS, checkpoints=provenance, history=history,
        exploration_overlap=len(prior), test_manifest=str(test),
        audio=audio[:2] if smoke else audio, music=list(music.values())[:2] if smoke else list(music.values()),
    ))


def load_protocol(root):
    return json.loads((root / "protocol.json").read_text())


def set_arm(dit, protocol, arm):
    if arm == "base":
        return
    meta = protocol["checkpoints"][arm]
    checkpoint = _load_checkpoint(Path(meta["path"]))
    if (checkpoint["exposure"] != meta.get("exposure", 2000)
            or checkpoint["optimizer_step"] != meta.get("optimizer_step", 500)):
        raise ValueError(f"incorrect checkpoint exposure or update count: {arm}")
    if "schedule_sha256" in meta and checkpoint["schedule_digest"] != meta["schedule_sha256"]:
        raise ValueError(f"checkpoint schedule mismatch: {arm}")
    dit.load_state_dict(checkpoint["dit"], strict=True)


def clip_for(dataset, row):
    random.seed(42 + row["row"])
    return dataset[row["row"]]


def condition(api, kind, item, clip=None):
    if kind == "reca":
        metadata = [build_reca_metadata(clip, "cuda:1", omni_downsample=4,
                                        prompt_rng=random.Random(42))]
    else:
        task = "T2M" if kind == "music" else "T2A"
        prompt = item["prompt"]
        metadata = api._build_conditioning(qwen_prompt=build_prompt(task, prompt), prompt=prompt,
                                           task=task, video_path=None, voice_prompt_path=None,
                                           voice_ref_text=None, seconds_total=10)
    raw = api.model.conditioner(metadata, "cuda:1")
    return {key: tuple(t.detach().cpu() if torch.is_tensor(t) else t for t in value)
            for key, value in raw.items()}


def move_condition(raw):
    return {key: tuple(t.to("cuda:0") if torch.is_tensor(t) else t for t in value)
            for key, value in raw.items()}


@torch.inference_mode()
def generate(root, shard, shards, *, skip_base=False):
    protocol = load_protocol(root)
    torch.manual_seed(42)
    api = AudioOmni(str(CONFIG), str(BASE), device="cuda:0", conditioner_device="cuda:1", dtype="float16")
    dataset = AudioManifestDataset(protocol["test_manifest"], seconds=10)
    cached = {}
    for arm in ARMS:
        if skip_base and arm == "base":
            continue
        set_arm(api.model.model, protocol, arm)
        for setting, (domain, size, cfg) in SETTINGS.items():
            items = protocol["music" if domain == "music" else "audio"]
            for index in range(shard, len(items), shards):
                item = items[index]
                source_index = (index + 1) % len(items) if setting == "reca_mismatched" else index
                source = items[source_index]
                key = (domain, source_index)
                folder = root / setting / arm
                folder.mkdir(parents=True, exist_ok=True)
                wav, record = folder / f"{index:03d}.wav", folder / f"{index:03d}.json"
                if record.exists():
                    if not wav.exists():
                        raise FileNotFoundError(wav)
                    continue
                if key not in cached:
                    clip = clip_for(dataset, source) if domain == "reca" else None
                    cached[key] = condition(api, domain, source, clip)
                if domain == "reca" and setting == "reca_matched" and arm == "base":
                    ref = root / "reca_ref" / f"{index:03d}.wav"
                    ref.parent.mkdir(exist_ok=True)
                    sf.write(ref, clip_for(dataset, item).audio.numpy().T, 44100, subtype="FLOAT")
                seed = protocol["seed"] + index + 1
                output = generate_diffusion_cond(
                    api.model, conditioning_tensors=move_condition(cached[key]), cfg_scale=cfg,
                    steps=protocol["steps"], sample_size=size, sample_rate=44100,
                    seed=seed, device="cuda:0", conditioner_device="cuda:1")
                frames = size // api.model.pretransform.downsampling_ratio * api.model.pretransform.downsampling_ratio
                if output.shape[-1] != frames or not torch.isfinite(output).all() or not output.any():
                    raise ValueError(f"invalid output: {wav}, shape={output.shape}")
                audio_io.save_audio(wav, api._postprocess(output), 44100)
                write_json(record, dict(**item, seed=seed, frames=frames, source_sha=source["sha"], wav=str(wav)))
            print(f"complete shard={shard} arm={arm} setting={setting}", flush=True)


def partition(name):
    match = re.search(r"transformer\.layers\.(\d+)\.(.+)", name)
    if match:
        block, module = match.groups()
        kind = ("norm" if "norm" in module else "cross_attention" if "cross_attn" in module
                else "self_attention" if "self_attn" in module else "ffn" if module.startswith("ff")
                else "other")
        return (f"block/{block}", f"block/{block}/{kind}", kind)
    return (name.rsplit(".", 1)[0] if "to_" not in name else ".".join(name.split(".")[:2]),)


def parameter_drift(protocol, dit):
    original = torch.load(BASE, map_location="cpu", mmap=True, weights_only=True)["state_dict"]
    params = dict(dit.named_parameters())
    result = {}
    for arm in ARMS:
        updated = original if arm == "base" else _load_checkpoint(Path(protocol["checkpoints"][arm]["path"]))["dit"]
        totals = defaultdict(lambda: [0., 0., 0])
        for name, parameter in params.items():
            before = original["model." + name]
            after = updated["model." + name if arm == "base" else name]
            if before.shape != parameter.shape or after.shape != parameter.shape:
                raise ValueError(f"shape mismatch: {arm}/{name}")
            base_norm, delta_norm = 0., 0.
            for a, b in zip(before.flatten().split(2**20), after.flatten().split(2**20)):
                a, b = a.float(), b.float()
                if not torch.isfinite(b).all():
                    raise ValueError(f"nonfinite checkpoint parameter: {arm}/{name}")
                base_norm += float(a.square().sum())
                delta_norm += float((a-b).square().sum())
            for key in ("global", *partition(name)):
                totals[key][0] += base_norm
                totals[key][1] += delta_norm
                totals[key][2] += parameter.numel()
        result[arm] = {key: dict(relative_l2=(delta / base)**.5 if base else None,
                                 delta_l2=delta**.5, base_l2=base**.5, parameters=count)
                       for key, (base, delta, count) in totals.items()}
    return result


@torch.inference_mode()
def geometry(root):
    protocol = load_protocol(root)
    torch.manual_seed(42)
    api = AudioOmni(str(CONFIG), str(BASE), device="cuda:0", conditioner_device="cuda:1", dtype="float16")
    dit = api.model.model
    write_json(root / "parameter_drift.json", parameter_drift(protocol, dit))
    dataset = AudioManifestDataset(protocol["test_manifest"], seconds=10)
    fixed = []
    generator = torch.Generator(device="cuda:0").manual_seed(42)
    for item in protocol["audio"][:8]:
        clip = clip_for(dataset, item)
        latent = api.model.pretransform.encode(clip.audio[None].to("cuda:0")).float()
        noise = torch.randn(latent.shape, generator=generator, device="cuda:0")
        for kind in ("audio", "reca"):
            raw = move_condition(condition(api, kind, item, clip))
            kwargs = {k: v.half() if torch.is_tensor(v) and v.is_floating_point() else v
                      for k, v in api.model.get_conditioning_inputs(raw).items()}
            fixed.append((item["sha"], "text" if kind == "audio" else "audio", latent, noise, kwargs))
    base, rows = {}, []
    for arm in ARMS:
        set_arm(dit, protocol, arm)
        for digest, kind, latent, noise, kwargs in fixed:
            for time in (.2, .5, .8):
                x = ((1-time)*latent + time*noise).half()
                t = x.new_full((1,), time)
                pair = []
                forward = dit.model._forward

                def capture(*args, **kw):
                    out = forward(*args, **kw)
                    pair.extend(out.chunk(2))
                    return out

                with patch.object(dit.model, "_forward", side_effect=capture):
                    guided2 = dit(x, t, **kwargs, cfg_scale=2)
                cond, uncond = pair
                g = cond.float() - uncond.float()
                target = noise - latent
                vectors = dict(cond=cond, uncond=uncond, guidance=g, cfg2=guided2)
                for cfg in (1, 4, 6):
                    vectors[f"cfg{cfg}"] = dit(x, t, **kwargs, cfg_scale=cfg)
                vectors = {k: v.float().flatten().cpu() for k, v in vectors.items()}
                key = (digest, kind, time)
                if arm == "base":
                    base[key] = vectors
                ref = base[key]
                drift = {k: float((v-ref[k]).norm() / ref[k].norm().clamp_min(1e-8))
                         for k, v in vectors.items()}
                rows.append(dict(arm=arm, sha=digest, condition=kind, t=time, drift=drift,
                                 guidance_cosine=float(torch.nn.functional.cosine_similarity(
                                     vectors["guidance"], ref["guidance"], dim=0)),
                                 guidance_norm_ratio=float(g.norm().cpu() / ref["guidance"].norm().clamp_min(1e-8)),
                                 flow_mse={k: float((v-target.flatten().cpu()).square().mean())
                                           for k, v in vectors.items() if k != "guidance"}))
    write_json(root / "geometry.json", rows)


def normalized(values):
    return values / (np.linalg.norm(values, axis=1, keepdims=True) + 1e-8)


def paired(values, baseline, rows):
    groups = defaultdict(list)
    delta = values - baseline
    for i, row in enumerate(rows):
        groups[row["sha"]].append(delta[i])
    group_delta = np.array([np.mean(group) for group in groups.values()])
    draws = np.random.default_rng(42).choice(group_delta, (10000, len(group_delta)))
    return dict(n=len(rows), unique_sha=len(groups), mean=float(values.mean()),
                median=float(np.median(values)), std=float(values.std()),
                mean_delta=float(group_delta.mean()),
                win=int((delta > 1e-6).sum()), tie=int((abs(delta) <= 1e-6).sum()),
                loss=int((delta < -1e-6).sum()),
                delta_ci95=np.quantile(draws.mean(axis=1), [.025, .975]).tolist())


def score(root):
    protocol = load_protocol(root)
    results = {}
    clap = metrics.get_clap("cuda:0")
    refs = {}
    for setting, (domain, _, _) in SETTINGS.items():
        items = protocol["music" if domain == "music" else "audio"]
        if domain not in refs:
            paths = ([str(root / "reca_ref" / f"{i:03d}.wav") for i in range(len(items))]
                     if domain == "reca" else [r["ref"] for r in items])
            np.random.seed(42)
            reference = metrics.clap_audio_embeddings(paths, "cuda:0")
            texts = [r["prompt"] for r in items]
            target = reference if domain == "reca" else np.concatenate([
                clap.get_text_embedding(texts[i:i+32], use_tensor=False) for i in range(0, len(texts), 32)])
            refs[domain] = paths, reference, normalized(target)
        paths, reference, target = refs[domain]
        results[setting] = {}
        for arm in ARMS:
            records = [json.loads((root / setting / arm / f"{i:03d}.json").read_text())
                       for i in range(len(items))]
            for i, (row, item) in enumerate(zip(records, items)):
                if row["sha"] != item["sha"] or row["seed"] != protocol["seed"] + i + 1:
                    raise ValueError(f"invalid pairing: {setting}/{arm}/{i}")
                info = sf.info(row["wav"])
                if info.frames != row["frames"] or info.samplerate != 44100:
                    raise ValueError(f"invalid WAV: {row['wav']}")
            wavs = [r["wav"] for r in records]
            np.random.seed(42)
            embeddings = metrics.clap_audio_embeddings(wavs, "cuda:0")
            unit = normalized(embeddings)
            values = np.sum(unit * target, axis=1)
            if arm == "base":
                baseline = values
            mask = np.array([not row["exploratory"] for row in records])
            summary = dict(all=paired(values, baseline, records),
                           confirmatory=paired(values[mask], baseline[mask], [r for r, ok in zip(records, mask) if ok]),
                           shuffled=float(np.sum(unit * np.roll(target, -1, axis=0), axis=1).mean()))
            if protocol["smoke"]:
                if domain == "audio" and setting == "audio_10_cfg6" and arm == "base":
                    metrics.panns_outputs(wavs, "cuda:0")
            else:
                summary["fad_clap"] = metrics.frechet_distance(embeddings, reference)
            if domain == "reca":
                summary["lsd"] = metrics.lsd(wavs, paths)
            elif not protocol["smoke"]:
                panns = metrics.panns_scores(wavs, paths, "cuda:0")
                if set(panns) != {"kl", "is", "fd"}:
                    raise RuntimeError("PANNs scoring incomplete")
                summary.update(panns)
            for row, value in zip(records, values):
                row["clap"] = float(value)
            results[setting][arm] = dict(summary=summary, rows=records)
        print(f"scored {setting}", flush=True)
        write_json(root / "scores_progress.json", results)
    comparisons = {}
    for arm in ARMS:
        for label, a, b in [("duration_11_minus_10", "audio_11_cfg6", "audio_10_cfg6"),
                            ("reca_matched_minus_mismatched", "reca_matched", "reca_mismatched")]:
            if a not in results or b not in results:
                continue
            rows = results[a][arm]["rows"]
            mask = [i for i, row in enumerate(rows) if not row["exploratory"]]
            comparisons.setdefault(label, {})[arm] = paired(
                np.array([rows[i]["clap"] for i in mask]),
                np.array([results[b][arm]["rows"][i]["clap"] for i in mask]), [rows[i] for i in mask])
    payload = dict(protocol=protocol, generation=results, paired_contrasts=comparisons)
    for key in ("geometry", "parameter_drift"):
        path = root / f"{key}.json"
        if path.exists():
            payload[key] = json.loads(path.read_text())
    write_json(root / "results.json", payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "generate", "geometry", "score"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.out, args.smoke)
    elif args.phase == "generate":
        if not 0 <= args.shard < args.shards:
            parser.error("invalid shard")
        generate(args.out, args.shard, args.shards)
    elif args.phase == "geometry":
        geometry(args.out)
    else:
        score(args.out)


if __name__ == "__main__":
    main()
