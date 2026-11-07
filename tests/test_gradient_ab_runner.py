import importlib.util
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


def test_launcher_guards_formal_run_and_reuses_one_schedule():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert 'ALLOW_FORMAL' in text
    assert 'schedule.jsonl' in text
    assert '--arm a' in text
    assert '--arm b' in text
    assert '--prepare-only' in text


def test_training_gradients_are_released_before_periodic_diagnostics():
    text = RUNNER.read_text(encoding="utf-8")
    step = text.index("scaler.step(optimizer)")
    update = text.index("scaler.update()", step)
    release = text.index("optimizer.zero_grad(set_to_none=True)", update)
    diagnostic = text.index("if exposure in diagnostic_exposures", update)
    assert update < release < diagnostic
