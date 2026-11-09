import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "src" / "reca_audio" / "train" / "run_gradient_ab.py"
LAUNCHER = ROOT / "runs" / "scripts" / "train_reca_gradient_ab.sh"


def _runner():
    spec = importlib.util.spec_from_file_location("gradient_ab_runner_under_test", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cli_contract_uses_equal_exposure_and_different_update_counts():
    module = _runner()
    common = [
        "--manifest", "train.jsonl",
        "--diagnostic-manifest", "test.jsonl",
        "--out", "output",
        "--schedule", "schedule.jsonl",
        "--total-exposures", "400",
        "--warmup-exposures", "20",
        "--peak-lr", "2e-5",
        "--allow-formal",
        "--dry-run",
    ]
    arm_a = module.experiment_contract(module.parse_args(["--arm", "a", *common]))
    arm_b = module.experiment_contract(module.parse_args(["--arm", "b", *common]))
    assert arm_a["total_exposures"] == arm_b["total_exposures"] == 400
    assert arm_a["optimizer_updates"] == 400
    assert arm_b["optimizer_updates"] == 100
    assert arm_a["accumulation"] == 1
    assert arm_b["accumulation"] == 4
    assert arm_a["schedule"] == arm_b["schedule"] == "schedule.jsonl"


def test_formal_run_requires_explicit_gate_but_smoke_does_not():
    module = _runner()
    base = [
        "--arm", "a",
        "--manifest", "train.jsonl",
        "--diagnostic-manifest", "test.jsonl",
        "--out", "output",
        "--schedule", "schedule.jsonl",
    ]
    with pytest.raises(ValueError, match="allow-formal"):
        module.validate_args(module.parse_args([*base, "--total-exposures", "400"]))
    args = module.parse_args([
        *base,
        "--total-exposures", "4",
        "--warmup-exposures", "1",
        "--diagnostic-exposures", "0,4",
        "--save-exposures", "4",
    ])
    module.validate_args(args)


def test_adafactor_contract_disables_internal_relative_step():
    module = _runner()
    assert module.adafactor_kwargs(2e-5, 0.01) == {
        "lr": 2e-5,
        "scale_parameter": False,
        "relative_step": False,
        "warmup_init": False,
        "weight_decay": 0.01,
    }


def test_prepare_only_is_not_mistaken_for_formal_training():
    module = _runner()
    args = module.parse_args([
        "--arm", "a",
        "--manifest", "train.jsonl",
        "--schedule", "schedule.jsonl",
        "--total-exposures", "400",
        "--warmup-exposures", "20",
        "--prepare-only",
    ])
    module.validate_args(args)


@pytest.mark.parametrize("task_mode", ["random_mix", "pure", "t2a", "balanced_mix"])
def test_runner_contract_records_task_mode(task_mode):
    module = _runner()
    args = module.parse_args([
        "--arm", "b",
        "--manifest", "train.jsonl",
        "--diagnostic-manifest", "test.jsonl",
        "--out", "output",
        "--schedule", "schedule.jsonl",
        "--task-mode", task_mode,
        "--total-exposures", "4",
        "--warmup-exposures", "1",
        "--diagnostic-exposures", "0,4",
        "--save-exposures", "4",
    ])
    assert module.experiment_contract(args)["task_mode"] == task_mode


def test_runner_rejects_unknown_task_mode():
    with pytest.raises(SystemExit):
        _runner().parse_args(["--task-mode", "unknown"])


def test_prepare_schedule_uses_selected_task_mode(tmp_path):
    module = _runner()
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(json.dumps({"audio": "unused.wav"}) + "\n", encoding="utf-8")
    schedule = tmp_path / "schedule.jsonl"
    args = module.parse_args([
        "--arm", "b",
        "--manifest", str(manifest),
        "--schedule", str(schedule),
        "--task-mode", "balanced_mix",
        "--total-exposures", "4",
        "--warmup-exposures", "1",
        "--prepare-only",
    ])

    module._prepare_schedule(args)

    tasks = [json.loads(line)["task"] for line in schedule.read_text().splitlines()]
    assert tasks.count("reca") == tasks.count("t2a") == 2


def test_launcher_guards_formal_run_and_reuses_one_schedule():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert 'ALLOW_FORMAL' in text
    assert 'validate_gpus' in text
    assert '--query-gpu=memory.used' in text
    assert 'schedule.jsonl' in text
    assert '--arm a' in text
    assert '--arm b' in text
    assert '--prepare-only' in text
    assert '--keep-final-checkpoint' in text


def test_training_gradients_are_released_before_periodic_diagnostics():
    text = RUNNER.read_text(encoding="utf-8")
    step = text.index("scaler.step(optimizer)")
    update = text.index("scaler.update()", step)
    release = text.index("optimizer.zero_grad(set_to_none=True)", update)
    diagnostic = text.index("if exposure in diagnostic_exposures", update)
    assert update < release < diagnostic
