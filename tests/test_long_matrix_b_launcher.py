from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "runs" / "scripts" / "train_reca_long_matrix_b.sh"


def _text() -> str:
    return LAUNCHER.read_text(encoding="utf-8")


def test_launcher_pins_exact_b_training_contract():
    text = _text()
    assert 'TOTAL=2000' in text
    assert 'WARMUP=100' in text
    assert '--peak-lr 2e-5' in text
    assert '--arm b' in text
    assert 'CLIP_SECONDS="${CLIP_SECONDS:-5}"' in text
    assert '--seconds "${CLIP_SECONDS}"' in text
    assert '--task-mode "${task_mode}"' in text
    assert 'prepare_schedule pure pure' in text
    assert 'prepare_schedule t2a t2a' in text
    assert 'prepare_schedule mix balanced_mix' in text


def test_launcher_guards_formal_outputs_and_six_unique_free_gpus():
    text = _text()
    assert 'ALLOW_FORMAL' in text
    assert 'refusing formal matrix without ALLOW_FORMAL=1' in text
    assert 'refusing existing output root' in text
    assert 'GPU_PAIRS' in text
    assert 'duplicate GPU' in text
    assert '--query-gpu=memory.used' in text


def test_launcher_runs_three_formal_arms_in_parallel_and_propagates_failure():
    text = _text()
    assert 'run_arm pure pure' in text
    assert 'run_arm t2a t2a' in text
    assert 'run_arm mix balanced_mix' in text
    assert 'pids+=("$!")' in text
    assert 'wait "${pid}"' in text
    assert 'if (( failed )); then\n  echo "one or more matrix arms failed" >&2\n  exit 1' in text


def test_launcher_smoke_is_sequential_and_keeps_only_final_checkpoint():
    text = _text()
    assert 'TOTAL=4' in text
    assert 'WARMUP=1' in text
    assert '--save-exposures "${TOTAL}"' in text
    assert '--keep-final-checkpoint' in text
    assert 'SMOKE_ARMS="${SMOKE_ARMS:-pure t2a mix}"' in text
    assert 'for arm in ${SMOKE_ARMS}; do' in text
    assert 'run_smoke_arm "${arm}" "${task_modes[${arm}]}"' in text
