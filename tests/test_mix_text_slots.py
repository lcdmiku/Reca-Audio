"""A CPU toy model exercises the actual optimizer loop and its schedule accounting."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _runner():
    path = Path(__file__).resolve().parents[1] / "src/reca_audio/train/run_gradient_ab.py"
    spec = importlib.util.spec_from_file_location("mix_text_slots_runner", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


def test_balanced_group_validation():
    runner = _runner()
    schedule = runner._helpers().generate_schedule(8, 2, 42, task_mode="balanced_mix")
    runner.validate_mix_text_slots(schedule)
    with pytest.raises(ValueError, match="four-slot"):
        runner.validate_mix_text_slots(schedule[:-1])
    with pytest.raises(ValueError, match="nonconsecutive"):
        runner.validate_mix_text_slots([schedule[1], schedule[0], *schedule[2:]])
    with pytest.raises(ValueError, match="two ReCA"):
        runner.validate_mix_text_slots([dict(schedule[0], task=schedule[1]["task"]), *schedule[1:]])


def test_text_slot_loop_matches_mix_targets_and_quarter_weighted_text_gradient(tmp_path, monkeypatch):
    runner = _runner()
    prepared, forwards = [], []

    class ToyVAE(torch.nn.Module):
        def encode(self, audio):
            # The real VAE likewise uses the global CPU RNG.
            latent = audio + torch.randn_like(audio)
            prepared.append(latent.item())
            return latent

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pretransform = ToyVAE()
            self.model = torch.nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                self.model.weight.fill_(1.0)

        def conditioner(self, metadata, device):
            return metadata[0]

        def get_conditioning_inputs(self, mode):
            return mode

    def toy_loss(dit, latent, mode, **kwargs):
        forwards.append(mode)
        return (dit(latent) - 0.5).square().mean(), {}

    modules = {
        "audio_omni.loader": dict(load_model_bundle=lambda *a, **k: (ToyModel(), None, None, None)),
        "transformers.optimization": dict(Adafactor=lambda p, lr, **k: torch.optim.SGD(p, lr=lr)),
        "reca_audio.train.data": dict(
            AudioManifestDataset=lambda *a, **k: [0.0, 1.0, 2.0],
            collate_clips=lambda clips: {"audio": torch.tensor([clips])},
        ),
        "reca_audio.train.flow": dict(rectified_flow_loss=toy_loss),
        "reca_audio.train.train": dict(
            build_metadata=lambda batch, index, mode, *a, **k: mode,
            move_cond=lambda cond, device: cond,
        ),
    }
    for name, attributes in modules.items():
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(**attributes))

    helpers = runner._helpers()
    schedule = helpers.generate_schedule(8, 3, 42, task_mode="balanced_mix")
    schedule_path = tmp_path / "mix.jsonl"
    digest = helpers.write_schedule(schedule_path, schedule)
    config = tmp_path / "model.json"
    config.write_text("{}")
    def run(mode):
        prepared.clear()
        forwards.clear()
        out = tmp_path / mode
        args = runner.parse_args([
            "--manifest", "train", "--diagnostic-manifest", "probe", "--out", str(out),
            "--arm", "b", "--schedule", str(schedule_path), "--schedule-digest", digest,
            "--task-mode", mode, "--model-config", str(config),
            "--device", "cpu", "--conditioner-device", "cpu", "--vae-device", "cpu",
            "--total-exposures", "8", "--warmup-exposures", "1", "--diagnostic-exposures", "0",
            "--diagnostic-samples", "1", "--save-exposures", "",
        ])
        assert runner._run(args) == 0
        steps = [r for r in map(json.loads, (out / "metrics.jsonl").read_text().splitlines())
                 if r["kind"] == "optimizer_step"]
        return prepared[2:], forwards[2:], steps

    mix_latents, mix_forwards, mix = run("balanced_mix")
    latents, control_forwards, control = run("mix_text_slots")
    assert latents == mix_latents and len(latents) == 8
    assert mix_forwards == [r["task"] for r in schedule]
    assert control_forwards == ["t2a"] * 4
    assert [r["lr"] for r in control] == [r["lr"] for r in mix]
    assert [r["optimizer_step"] for r in control] == [1, 2]
    assert [r["active_training_exposures"] for r in control] == [2, 4]
    assert control[-1]["active_task_counts"] == {"t2a": 4, "reca": 0}
    assert [r["scheduled_slots"] for r in control] == [4, 8]
    assert all(r["loss_divisor"] == 4 and r["reca_loss"] is None for r in control)
    assert control[0]["t2a_loss"] == mix[0]["t2a_loss"]
    first_text = [z for z, r in zip(latents[:4], schedule[:4]) if r["task"] == "t2a"]
    expected = abs(sum(2 * (z - 0.5) * z for z in first_text) / 4)
    assert control[0]["gradient"]["pre_clip_norm"] == pytest.approx(expected)
