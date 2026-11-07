import hashlib
import importlib.util
import json
import math
import random
from copy import deepcopy
from pathlib import Path

import pytest
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "reca_audio"
    / "train"
    / "gradient_experiment.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("gradient_experiment_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_exposure_lr_has_one_based_warmup_and_cosine_boundaries():
    exposure_lr = _module().exposure_lr

    peak = 2e-5
    assert exposure_lr(1, 400, 20, peak) == pytest.approx(peak / 20)
    assert exposure_lr(20, 400, 20, peak) == pytest.approx(peak)
    assert exposure_lr(400, 400, 20, peak) == pytest.approx(0.0, abs=1e-15)

    midpoint = 210
    expected = peak * 0.5 * (1.0 + math.cos(math.pi * 0.5))
    assert exposure_lr(midpoint, 400, 20, peak) == pytest.approx(expected)


def test_update_lr_averages_each_exposure_position():
    module = _module()
    exposure_lr, update_lr = module.exposure_lr, module.update_lr

    expected = sum(exposure_lr(i, 400, 20, 2e-5) for i in range(21, 25)) / 4
    assert update_lr(21, 4, 400, 20, 2e-5) == pytest.approx(expected)


@pytest.mark.parametrize("exposure", [0, 401])
def test_exposure_lr_rejects_out_of_range_positions(exposure):
    exposure_lr = _module().exposure_lr

    with pytest.raises(ValueError, match="exposure"):
        exposure_lr(exposure, 400, 20, 2e-5)


def test_validate_experiment_rejects_partial_accumulation_group():
    validate_experiment_shape = _module().validate_experiment_shape

    with pytest.raises(ValueError, match="divisible"):
        validate_experiment_shape(total_exposures=10, accumulation=4)


def test_schedule_is_deterministic_and_contains_shared_task_sequence():
    generate_schedule = _module().generate_schedule

    first = generate_schedule(400, manifest_size=707, seed=42)
    second = generate_schedule(400, manifest_size=707, seed=42)
    different = generate_schedule(400, manifest_size=707, seed=43)

    assert first == second
    assert first != different
    assert [row["exposure"] for row in first] == list(range(1, 401))
    assert {row["task"] for row in first} == {"reca", "t2a"}
    assert all(0 <= row["sample_index"] < 707 for row in first)
    assert all({"data_seed", "flow_seed", "prompt_seed"} <= row.keys() for row in first)


def test_schedule_digest_detects_modified_jsonl(tmp_path):
    module = _module()
    generate_schedule = module.generate_schedule
    load_schedule = module.load_schedule
    write_schedule = module.write_schedule

    path = tmp_path / "schedule.jsonl"
    records = generate_schedule(8, manifest_size=5, seed=42)
    digest = write_schedule(path, records)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert load_schedule(path, expected_digest=digest) == records

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["task"] = "t2a" if rows[0]["task"] == "reca" else "reca"
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        load_schedule(path, expected_digest=digest)


def test_streaming_gradient_moments_match_exact_vector_statistics():
    module = _module()
    moments = module.GradientMoments()
    samples = [
        {"blocks.0.weight": torch.tensor([1.0, 2.0]), "head.weight": torch.tensor([3.0])},
        {"blocks.0.weight": torch.tensor([3.0, 4.0]), "head.weight": torch.tensor([5.0])},
    ]
    originals = [{name: value.clone() for name, value in sample.items()} for sample in samples]
    for sample in samples:
        moments.add(sample)

    summary = moments.summary()
    stacked = torch.stack([torch.cat(list(sample.values())) for sample in samples])
    mean = stacked.mean(dim=0)
    expected_variance = (stacked.square().sum(dim=1).mean() - mean.square().sum()).item()
    assert summary["count"] == 2
    assert summary["mean_gradient_norm"] == pytest.approx(mean.norm().item())
    assert summary["gradient_variance"] == pytest.approx(expected_variance)
    assert summary["mean_sample_norm"] == pytest.approx(
        stacked.norm(dim=1).mean().item()
    )
    for original, sample in zip(originals, samples):
        assert all(torch.equal(original[name], sample[name]) for name in original)


def test_gradient_cosine_and_zero_norm_are_defined_without_flattening():
    module = _module()
    left = {"a": torch.tensor([1.0, 0.0]), "b": torch.tensor([1.0])}
    right = {"a": torch.tensor([0.0, 1.0]), "b": torch.tensor([1.0])}
    assert module.gradient_cosine(left, right) == pytest.approx(0.5)
    assert module.gradient_cosine(left, {name: torch.zeros_like(x) for name, x in left.items()}) is None


def test_block_statistics_group_representative_dit_layers():
    module = _module()
    gradients = {
        "transformer_blocks.0.attn.weight": torch.tensor([3.0]),
        "transformer_blocks.0.ff.weight": torch.tensor([4.0]),
        "transformer_blocks.11.attn.weight": torch.tensor([12.0]),
        "proj_out.weight": torch.tensor([5.0]),
    }
    norms = module.block_gradient_norms(gradients)
    assert norms == pytest.approx(
        {"transformer_blocks.0": 5.0, "transformer_blocks.11": 12.0, "proj_out": 5.0}
    )


def test_clipping_metrics_report_scale_trigger_and_post_norm():
    module = _module()
    assert module.clipping_metrics(2.5, 1.0) == pytest.approx(
        {"pre_clip_norm": 2.5, "post_clip_norm": 1.0, "clip_scale": 0.4, "clipped": 1.0}
    )
    assert module.clipping_metrics(0.5, 1.0) == pytest.approx(
        {"pre_clip_norm": 0.5, "post_clip_norm": 0.5, "clip_scale": 1.0, "clipped": 0.0}
    )


def test_parameter_drift_reports_global_and_block_relative_l2_without_mutation():
    module = _module()
    baseline = {
        "transformer_blocks.0.weight": torch.tensor([3.0, 4.0]),
        "proj_out.weight": torch.tensor([12.0]),
    }
    current = {
        "transformer_blocks.0.weight": torch.tensor([6.0, 8.0]),
        "proj_out.weight": torch.tensor([12.0]),
    }
    originals = {name: value.clone() for name, value in current.items()}
    drift = module.parameter_drift(baseline, current)
    assert drift["global_relative_l2"] == pytest.approx(5.0 / 13.0)
    assert drift["blocks"]["transformer_blocks.0"] == pytest.approx(1.0)
    assert drift["blocks"]["proj_out"] == pytest.approx(0.0)
    assert all(torch.equal(originals[name], current[name]) for name in current)


def _assert_nested_tensors_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_tensors_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for lhs, rhs in zip(left, right):
            _assert_nested_tensors_equal(lhs, rhs)
    else:
        assert left == right


def test_diagnostic_gradient_restores_rng_parameters_gradients_and_optimizer_state():
    module = _module()
    torch.manual_seed(123)
    random.seed(456)
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    for index, parameter in enumerate(model.parameters()):
        parameter.grad = torch.full_like(parameter, float(index + 1))

    params_before = {name: value.detach().clone() for name, value in model.named_parameters()}
    grads_before = {
        name: value.grad.detach().clone() if value.grad is not None else None
        for name, value in model.named_parameters()
    }
    optimizer_before = deepcopy(optimizer.state_dict())
    python_rng_before = random.getstate()
    torch_rng_before = torch.random.get_rng_state().clone()

    def loss_closure():
        scale = random.random() + torch.rand(()).item()
        return model(torch.tensor([[0.25, -0.5]])).square().mean() * scale

    result = module.diagnostic_gradient(model, loss_closure, seed=999)

    assert result["loss"] >= 0
    assert result["gradients"].keys() == dict(model.named_parameters()).keys()
    assert random.getstate() == python_rng_before
    assert torch.equal(torch.random.get_rng_state(), torch_rng_before)
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, params_before[name])
        assert torch.equal(parameter.grad, grads_before[name])
    _assert_nested_tensors_equal(optimizer.state_dict(), optimizer_before)


def test_diagnostic_gradient_is_repeatable_for_a_fixed_seed():
    module = _module()
    model = torch.nn.Linear(2, 1)

    def loss_closure():
        return model(torch.rand(1, 2)).square().mean() * (1.0 + random.random())

    first = module.diagnostic_gradient(model, loss_closure, seed=77)
    second = module.diagnostic_gradient(model, loss_closure, seed=77)
    assert first["loss"] == pytest.approx(second["loss"])
    assert module.gradient_cosine(first["gradients"], second["gradients"]) == pytest.approx(1.0)
