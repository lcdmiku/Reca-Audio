import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest


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
