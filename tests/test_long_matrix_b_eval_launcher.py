from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "runs" / "scripts" / "eval_long_matrix_b.sh"


def _text() -> str:
    return LAUNCHER.read_text(encoding="utf-8")


def test_eval_reuses_fixed_base_and_evaluates_all_three_new_arms():
    text = _text()
    assert "Matrix_v1_eval3_6gpu_9ce3984" in text
    assert "reca_long_matrix_b_v1/pure/dit_exposure2000.pt" in text
    assert "reca_long_matrix_b_v1/t2a/dit_exposure2000.pt" in text
    assert "reca_long_matrix_b_v1/mix/dit_exposure2000.pt" in text
    assert "for arm in pure t2a mix" in text


def test_eval_matches_prior_generation_protocol_and_benchmarks():
    text = _text()
    assert "AudioCapsHeldout" in text
    assert "MusicCaps" in text
    assert "STEPS=100" in text
    assert "CFG=6" in text
    assert "DTYPE=float16" in text
    assert "NUM_SHARDS=3" in text
    assert 'GPU_IDS="${GPU_IDS}"' in text


def test_eval_merges_one_checkpoint_in_tmp_and_always_cleans_it():
    text = _text()
    assert "mktemp -d /tmp/reca-long-matrix-b-eval." in text
    assert "trap cleanup EXIT INT TERM" in text
    assert "reca_audio.train.merge_ckpt" in text
    assert 'rm -f "${merged}"' in text
    assert "refusing existing output root" in text
