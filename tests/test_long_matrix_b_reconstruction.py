from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "runs" / "scripts" / "eval_reconstruction_matrix.py"
LAUNCHER = ROOT / "runs" / "scripts" / "eval_long_matrix_b_reconstruction.sh"


def test_reconstruction_protocol_matches_prior_matrix_diagnostic():
    text = PYTHON.read_text(encoding="utf-8")
    assert "SAMPLE_RATE = 44100" in text
    assert "TARGET_SECONDS = 5" in text
    assert "steps=100" in text
    assert "cfg_scale=1.0" in text
    assert "seconds_total=10" in text
    assert "seed=1234 + index" in text


def test_reconstruction_launcher_reuses_base_and_evaluates_three_arms():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "Reconstruction_v4_6gpu_9ce3984" in text
    assert "reca_long_matrix_b_v1/pure/dit_exposure2000.pt" in text
    assert "reca_long_matrix_b_v1/t2a/dit_exposure2000.pt" in text
    assert "reca_long_matrix_b_v1/mix/dit_exposure2000.pt" in text
    assert "for arm in pure t2a mix" in text
    assert "reca_audio.train.merge_ckpt" in text


def test_reconstruction_launcher_uses_tmp_merge_and_cleanup():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "mktemp -d /tmp/reca-long-matrix-b-recon." in text
    assert "trap cleanup EXIT INT TERM" in text
    assert 'rm -f "${merged}"' in text
    assert "refusing existing output root" in text
