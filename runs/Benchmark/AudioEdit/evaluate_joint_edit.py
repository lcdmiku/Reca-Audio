"""Compare native and joint audio/text conditioning with one Audio-Omni Base."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from shutil import copy2
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
from torchaudio.functional import resample

from audio_omni.prompts import SYSTEM_EDITING, _wrap_audio
from reca_audio.train.conditioning import _apply_bottleneck

from evaluate_transfer import CONFIG, FloatEdit, digest, edit, write
from report_joint_edit import render_report
from score_transfer import score_run


ROOT = Path("runs/Benchmark/output/AudioEdit_joint_qwen_v1")
HISTORICAL = Path("runs/Benchmark/output/AudioEdit_reca_zero_shot_v1/formal")
MANIFEST_SHA = "5e1760d86105bbf31f2934beace88ea045199f9c1445c7c7686c47928882672a"
BASE_SHA = "8ddfdd6c965b7b1fc5c5af01e306cdfd78e44f56fd584789b8987233438e65b3"
ARMS = ("a0", "a1")


def tensor_digest(wave):
    return hashlib.sha256(wave.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def edit_and_trace(api, row, joint):
    conditioner = api.model.conditioner
    omni = conditioner.conditioners["omni_prompt"]
    tokenizer = omni.processor.tokenizer
    audio_id = tokenizer.convert_tokens_to_ids("<|AUDIO|>")
    if not isinstance(audio_id, int) or tokenizer.convert_ids_to_tokens(audio_id) != "<|AUDIO|>":
        raise ValueError("Could not identify the Qwen audio token")
    calls = []
    evidence = {}

    def inspect_thinker(module, args, kwargs):
        ids = kwargs["input_ids"][0]
        count = int((ids == audio_id).sum())
        decoded = tokenizer.decode(ids.tolist(), skip_special_tokens=True)
        features = kwargs.get("input_features")
        feature_mask = kwargs.get("feature_attention_mask")
        kept = " ".join(row["instruction"].split()) in " ".join(decoded.split())
        audio_feature_frames = int(feature_mask.sum()) if feature_mask is not None else 0
        calls.append(dict(audio_tokens=count, audio_feature_frames=audio_feature_frames,
                          instruction_present=kept,
                          same_thinker_call=bool(features is not None and audio_feature_frames and count and kept)))

    original = conditioner.forward

    def condition(metadata, device):
        (native,) = metadata
        omni_prompt = native["omni_prompt"]
        if omni_prompt["audio_prompt"] is not None or "reca_audio_pool_width" in omni_prompt:
            raise ValueError("Expected native Edit conditioning")
        if native["text_prompt"] != row["instruction"]:
            raise ValueError(f"Native instruction differs for {row['id']}")
        raw = native["audio_input_prompt"]["audio_input_wav"]
        evidence["mel_source_sha"] = tensor_digest(raw)
        evidence["mel_source_frames"] = raw.numel()
        if joint:
            audio = _apply_bottleneck(resample(raw.detach().cpu(), 44100, 16000), downsample=4)
            if audio.numel() != 160000 or not torch.isfinite(audio).all() or audio.abs().max() <= 1e-8:
                raise ValueError(f"Invalid Qwen audio for {row['id']}")
            evidence["qwen_audio_sha"] = tensor_digest(audio)
            evidence["qwen_waveform_samples"] = audio.numel()
            native = dict(native, omni_prompt=dict(omni_prompt,
                          text_prompt=_wrap_audio(SYSTEM_EDITING, native["text_prompt"]),
                          audio_prompt=audio.numpy()))
        output = original([native], device)
        evidence["mel_projection_sha"] = tensor_digest(output["audio_input_prompt"][0])
        evidence["qwen_projection_sha"] = tensor_digest(output["omni_prompt"][0])
        return output

    hook = omni.model.thinker.register_forward_pre_hook(inspect_thinker, with_kwargs=True)
    try:
        with patch.object(conditioner, "forward", side_effect=condition):
            audio = edit(api, row, steps=100)
    finally:
        hook.remove()
    (thinker,) = calls
    if not thinker["instruction_present"] or thinker["same_thinker_call"] != joint:
        raise ValueError(f"Qwen modality evidence differs for {row['id']}: {thinker}")
    if not joint and (thinker["audio_tokens"] or thinker["audio_feature_frames"]):
        raise ValueError("Native Edit unexpectedly used Qwen audio")
    return audio, dict(evidence, **thinker)


def generate_case(api, row, out, arm):
    folder = out / "audio" / arm
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{row['id']}.wav"
    meta = path.with_suffix(".json")
    if digest(row["source"]) != row["source_sha256"]:
        raise ValueError(f"Source changed: {row['id']}")
    if meta.exists():
        saved = json.loads(meta.read_text())
        if saved["seed"] != row["seed"] or saved["wav_sha256"] != digest(path):
            raise ValueError(f"Existing WAV differs from its metadata: {path}")
        return saved
    audio, evidence = edit_and_trace(api, row, joint=arm == "a1")
    if audio.ndim != 2 or audio.shape[-1] != 483328 or not torch.isfinite(audio).all() or not audio.abs().max() > 1e-8:
        raise ValueError(f"Invalid output: {arm}/{row['id']}")
    tmp = path.with_suffix(".tmp.wav")
    sf.write(tmp, audio.T.numpy(), 44100, subtype="FLOAT")
    tmp.replace(path)
    record = dict(id=row["id"], seed=row["seed"], frames=audio.shape[-1],
                  wav_sha256=digest(path), **evidence)
    write(meta, record)
    print(arm, row["id"], evidence["audio_tokens"], flush=True)
    return record


def pilot(api, rows, out):
    kinds = ("add", "remove", "swap_instruction", "swap_source")
    group = next(row["group"] for row in rows if row["kind"] == "swap_instruction")
    chosen = [next(row for row in rows if row["group"] == group and row["kind"] == k) for k in kinds]
    records = []
    for row in chosen:
        a0, a1 = (generate_case(api, row, out / "pilot", a) for a in ARMS)
        for key in ("mel_source_sha", "mel_source_frames", "mel_projection_sha"):
            if a0[key] != a1[key]:
                raise ValueError(f"Altered native Mel for {row['id']}")
        if a0["qwen_projection_sha"] == a1["qwen_projection_sha"]:
            raise ValueError(f"Qwen output was unaffected for {row['id']}")
        previous = HISTORICAL / "audio/base" / f"{row['id']}.wav"
        new = out / "pilot/audio/a0" / previous.name
        a0_wave, new_rate = sf.read(new, dtype="float32")
        reference_wave, reference_rate = sf.read(previous, dtype="float32")
        # Float WAV headers may store metadata that differs between writes.
        same_waveform = new_rate == reference_rate == 44100 and np.array_equal(a0_wave, reference_wave)
        records.append(dict(id=row["id"], kind=row["kind"], a0=a0, a1=a1,
                            a0_waveform_matches_archive=same_waveform,
                            a0_file_matches_archive=a0["wav_sha256"] == digest(previous)))
    native, _, change_instruction, change_source = records
    if not (native["a1"]["qwen_audio_sha"] == change_instruction["a1"]["qwen_audio_sha"]
            and native["a1"]["qwen_audio_sha"] != change_source["a1"]["qwen_audio_sha"]
            and native["a1"]["qwen_projection_sha"] != change_instruction["a1"]["qwen_projection_sha"]
            and native["a0"]["qwen_projection_sha"] == change_source["a0"]["qwen_projection_sha"]
            and native["a1"]["qwen_projection_sha"] != change_source["a1"]["qwen_projection_sha"]
            and native["a1"]["mel_projection_sha"] == change_instruction["a1"]["mel_projection_sha"]
            and native["a1"]["mel_projection_sha"] != change_source["a1"]["mel_projection_sha"]):
        raise ValueError("Pilot source/instruction intervention failed")
    reuse = all(row["a0_waveform_matches_archive"] for row in records)
    write(out / "condition_checks.json", dict(rows=records, reuse_a0=reuse))
    return reuse


@torch.inference_mode()
def run(out):
    if (out / "GENERATION_COMPLETE").exists():
        raise ValueError(f"Generation already complete: {out}")
    if not (HISTORICAL / "COMPLETE").is_file() or digest(HISTORICAL / "manifest.jsonl") != MANIFEST_SHA:
        raise ValueError("Original edit benchmark is incomplete or differs from the frozen manifest")
    prior = json.loads((HISTORICAL / "inference_protocol.json").read_text())
    base = prior["checkpoints"]["base"]
    if base["sha256"] != BASE_SHA or digest(base["path"]) != BASE_SHA:
        raise ValueError("Unexpected Base checkpoint")
    shared = dict(config_sha=digest(CONFIG), sample_rate=44100, input_frames=441000,
                  requested_frames=485100, expected_frames=483328, steps=100, cfg=7,
                  dtype="float16", manifest_sha=MANIFEST_SHA, qwen_snapshot=os.environ.get("QWEN_OMNI_MODEL_PATH"))
    if any(prior.get(k) != v for k, v in shared.items()):
        raise ValueError("Existing benchmark protocol is incompatible with this run")
    out.mkdir(parents=True, exist_ok=True)
    raw = (HISTORICAL / "manifest.jsonl").read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    if len(rows) != len({row["id"] for row in rows}) or len(rows) != 120:
        raise ValueError("Unexpected manifest cardinality")
    (out / "manifest.jsonl").write_bytes(raw)
    protocol = dict(shared, checkpoint=base, qwen_downsample=dict(a0=None, a1=4),
                    gpu_visibility=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    deployment=Path("DEPLOYED_COMMIT").read_text().strip())
    protocol_path = out / "inference_protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Existing outputs have a different protocol")
    write(protocol_path, protocol)
    api = FloatEdit(str(CONFIG), base["path"], device="cuda:0", conditioner_device="cuda:1", dtype="float16")
    reuse_a0 = pilot(api, rows, out)
    print(f"pilot passed; reuse_a0={reuse_a0}", flush=True)
    (out / "audio").mkdir(exist_ok=True)
    if reuse_a0:
        link = out / "audio/a0"
        destination = (HISTORICAL / "audio/base").resolve()
        if not link.is_symlink():
            link.symlink_to(destination, target_is_directory=True)
        if link.resolve() != destination:
            raise ValueError("Unexpected A0 output source")
    for arm in ARMS[1:] if reuse_a0 else ARMS:
        for row in rows:
            artifact = f"{row['id']}.wav"
            pilot_wav = out / "pilot" / "audio" / arm / artifact
            full_wav = out / "audio" / arm / artifact
            if pilot_wav.exists() and not full_wav.with_suffix(".json").exists():
                full_wav.parent.mkdir(parents=True, exist_ok=True)
                copy2(pilot_wav, full_wav)
                copy2(pilot_wav.with_suffix(".json"), full_wav.with_suffix(".json"))
            generate_case(api, row, out, arm)
    write(out / "GENERATION_COMPLETE", dict(rows=len(rows), arms=list(ARMS), reused_a0=reuse_a0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("generate", "score"))
    parser.add_argument("--out", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.phase == "generate":
        run(args.out)
    else:
        result = score_run(args.out, "cuda:0", arms=ARMS, contrasts=(("a1", "a0"),))
        render_report(args.out, result)
