"""Run the existing Audio-Omni edit path for a fixed set of DiT checkpoints."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import soundfile as sf
import torch

from audio_omni.api import AudioOmni
from audio_omni.inference.generation import generate_diffusion_cond
from audio_omni.prompts import SYSTEM_EDITING, _wrap


CONFIG = Path("Audio-Omni/model/Audio-Omni.json")
REFERENCE = Path("runs/audits/audiocaps_mix_pool4_eval_v1/results.json")
REFERENCE_SHA = "615f17aff936c843e3b38cefc90ac9be3ea16f604135e70654861785d705d2fa"
ARMS = ("base", "pure_7070", "t2a_7070", "mix_7070", "mix_pool4_7070", "text_slots_7070")
SEED = 37021


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


class FloatEdit(AudioOmni):
    def _postprocess(self, output):
        return output.squeeze(0).detach().float().cpu()

    def edit_instruction(self, instruction, source, seed, steps):
        wav = self._load_audio_mono(source).to(self.device)
        condition = [
            dict(omni_prompt=dict(text_prompt=_wrap(SYSTEM_EDITING, instruction), video_prompt=None,
                                  audio_prompt=None),
                 text_prompt=instruction,
                 audio_input_prompt=dict(audio_input_wav=wav, TASK_TTS=False),
                 sync_feature=torch.zeros(1, 240, 768, device=self.device), speech_prompt=None)
        ]
        return self._postprocess(generate_diffusion_cond(
            self.model, conditioning=condition, steps=steps, cfg_scale=7, seed=seed,
            conditioner_device=self.conditioner_device, sample_size=self.sample_size,
            sample_rate=self.sample_rate, device=self.device,
        ))


def smoke_rows():
    src = Path("Audio-Omni/example/edit")
    rows = [
        dict(id="demo_add", task="Add", source=str(src / "add/add.mp3"), desc="skateboarding"),
        dict(id="demo_remove", task="Remove", source=str(src / "remove/remove.mp3"), desc="female singing"),
        dict(id="demo_extract", task="Extract", source=str(src / "extract/extract.mp3"), desc="wood thrush calling"),
        dict(id="demo_style", task="Style Transfer", source=str(src / "transfer/example.mp3"),
             source_category="playing electric guitar", target_category="playing saxophone"),
        dict(id="demo_custom", task="Custom", source=str(src / "add/add.mp3"),
             instruction="Reduce the low frequencies in the input audio."),
    ]
    for index, row in enumerate(rows):
        row["seed"] = SEED + index
    return rows


def load_arm(api, spec):
    state = torch.load(spec["path"], map_location="cpu", mmap=True, weights_only=True)
    assert state["exposure"] == spec["exposure"]
    assert state["optimizer_step"] == spec["optimizer_step"]
    assert state["schedule_digest"] == spec["schedule_sha256"]
    assert state["contract"] == spec["contract"]
    api.model.model.load_state_dict(state["dit"], strict=True)


def edit(api, row, steps):
    # The native Mel projection expects exactly 1,723 frames at 44.1 kHz.
    wav = api._load_audio_mono(row["source"])
    if wav.numel() != 441000 or not torch.isfinite(wav).all() or not wav.abs().max() > 1e-8:
        raise ValueError(f"Invalid 10 s edit source: {row['id']}")
    if row["task"] == "Custom":
        return api.edit_instruction(row["instruction"], row["source"], row["seed"], steps)
    kwargs = {key: row[key] for key in ("desc", "source_category", "target_category") if row.get(key)}
    return api.edit(row["task"], row["source"], cfg_scale=7, steps=steps, seed=row["seed"], **kwargs)


def dependency_protocol(verify_hashes):
    if digest(REFERENCE) != REFERENCE_SHA:
        raise ValueError("Reference checkpoint protocol changed")
    archived = json.loads(REFERENCE.read_text())["protocol"]
    checkpoints = {arm: archived["checkpoints"][arm] for arm in ARMS}
    for arm, spec in checkpoints.items():
        ckpt = Path(spec["path"])
        if not ckpt.is_file() or (verify_hashes and digest(ckpt) != spec["sha256"]):
            raise ValueError(f"Invalid checkpoint: {arm}")
        print("checkpoint", arm, ckpt.stat().st_size, "verified" if verify_hashes else "found", flush=True)
    return checkpoints


def check_condition_inputs(api, rows):
    seen = []

    def capture(_, args, output):
        meta = args[0][0]
        assert meta["omni_prompt"]["audio_prompt"] is None
        assert meta["audio_input_prompt"]["audio_input_wav"].abs().max() > 0
        seen.append({key: output[key][0].detach().cpu()
                     for key in ("omni_prompt", "audio_input_prompt")})

    hook = api.model.conditioner.register_forward_hook(capture)
    try:
        base, wrong_source = rows[:2]
        edit(api, base, 1)
        edit(api, dict(base, desc="a bell ringing"), 1)
        edit(api, dict(base, source=wrong_source["source"]), 1)
    finally:
        hook.remove()
    same_source = torch.equal(seen[0]["audio_input_prompt"], seen[1]["audio_input_prompt"])
    changed_source = not torch.equal(seen[0]["audio_input_prompt"], seen[2]["audio_input_prompt"])
    same_instruction = torch.equal(seen[0]["omni_prompt"], seen[2]["omni_prompt"])
    changed_instruction = not torch.equal(seen[0]["omni_prompt"], seen[1]["omni_prompt"])
    checks = dict(same_source=same_source, changed_source=changed_source,
                  same_instruction=same_instruction, changed_instruction=changed_instruction)
    if not all(checks.values()):
        raise RuntimeError(checks)
    return checks


@torch.inference_mode()
def run(out, smoke):
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_specs = dependency_protocol(verify_hashes=not smoke)
    manifest = out / "manifest.jsonl"
    rows = smoke_rows() if smoke else [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    if not rows or len(rows) != len({row["id"] for row in rows}):
        raise ValueError("Empty or duplicate edit manifest")
    if smoke:
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    protocol = dict(checkpoints=checkpoint_specs, reference_sha=REFERENCE_SHA, config_sha=digest(CONFIG),
                    sample_rate=44100, input_frames=441000, requested_frames=485100, expected_frames=483328,
                    steps=12 if smoke else 100, cfg=7, dtype="float16", smoke=smoke,
                    gpu_visibility=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    qwen_snapshot=os.environ.get("QWEN_OMNI_MODEL_PATH"),
                    manifest_sha=None if smoke else digest(manifest),
                    deployment=Path("DEPLOYED_COMMIT").read_text().strip())
    protocol_file = out / "inference_protocol.json"
    if protocol_file.is_file():
        previous = json.loads(protocol_file.read_text(encoding="utf-8"))
        if previous != protocol:
            raise ValueError("The existing evaluation used a different protocol")
    else:
        write(protocol_file, protocol)
    api = FloatEdit(str(CONFIG), checkpoint_specs["base"]["path"], device="cuda:0",
                    conditioner_device="cuda:1", dtype="float16")
    assert api.sample_size == protocol["requested_frames"] and api.sample_rate == protocol["sample_rate"]
    if smoke:
        write(out / "condition_checks.json", check_condition_inputs(api, rows))
    failures = []
    for arm in ARMS:
        folder = out / "audio" / arm
        folder.mkdir(parents=True, exist_ok=True)
        if arm != "base":
            load_arm(api, checkpoint_specs[arm])
        complete = 0
        for row in rows:
            target = folder / f"{row['id']}.wav"
            meta = target.with_suffix(".json")
            try:
                if meta.is_file():
                    recorded = json.loads(meta.read_text())
                    assert recorded["seed"] == row["seed"] and recorded["wav_sha256"] == digest(target)
                else:
                    audio = edit(api, row, protocol["steps"])
                    if audio.ndim != 2 or audio.shape[-1] != protocol["expected_frames"]:
                        raise ValueError(f"Unexpected output shape {tuple(audio.shape)}")
                    if not torch.isfinite(audio).all() or not audio.abs().max() > 1e-8:
                        raise ValueError("Nonfinite or silent output")
                    tmp = target.with_suffix(".tmp.wav")
                    sf.write(tmp, audio.T.numpy(), 44100, subtype="FLOAT")
                    tmp.replace(target)
                    write(meta, dict(id=row["id"], seed=row["seed"], frames=audio.shape[-1],
                                     wav_sha256=digest(target)))
                complete += 1
                print(arm, complete, "/", len(rows), row["id"], flush=True)
            except (RuntimeError, ValueError, AssertionError, OSError) as exc:
                failures.append(dict(arm=arm, id=row["id"], error=str(exc)))
                write(out / "failures.json", failures)
                raise
        write(folder / "complete.json", dict(rows=complete))
    write(out / "GENERATION_COMPLETE", dict(rows=len(rows), arms=list(ARMS)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    opts = parser.parse_args()
    run(opts.out, opts.smoke)
