from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "runs" / "scripts" / "train_reca_matrix.sh"


def _bash() -> str:
    if os.name == "nt":
        candidate = Path(r"D:\Git\bin\bash.exe")
        if candidate.is_file():
            return str(candidate)
    found = shutil.which("bash")
    if not found:
        raise unittest.SkipTest("bash is unavailable")
    return found


def _run(action: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    merged.update(env)
    return subprocess.run(
        [_bash(), str(SCRIPT), action],
        cwd=REPO,
        env=merged,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


class MatrixLauncherTests(unittest.TestCase):
    def _valid_inputs(self, root: Path) -> dict[str, str]:
        split = root / "split"
        split.mkdir()
        (split / "train_manifest.jsonl").write_text("{}\n", encoding="utf-8")
        report = {
            "intersections": {
                "train_val": {"id": 0, "sha256": 0},
                "train_test": {"id": 0, "sha256": 0},
                "val_test": {"id": 0, "sha256": 0},
            }
        }
        (split / "split_report.json").write_text(json.dumps(report), encoding="utf-8")
        ckpt = root / "model.ckpt"
        ckpt.write_bytes(b"checkpoint")
        return {
            "SPLIT_DIR": str(split),
            "ROOT": str(root / "runs"),
            "CKPT": str(ckpt),
            "DRY_RUN": "1",
            "SKIP_GPU_CHECK": "1",
        }

    def test_status_reports_all_missing_arms_without_creating_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            result = _run("status", {"ROOT": str(root)})
            self.assertEqual(result.returncode, 0, result.stdout)
            for arm in ("pure", "t2a", "mix"):
                self.assertIn(f"{arm} MISSING", result.stdout)
            self.assertFalse(root.exists())

    def test_dry_run_emits_three_isolated_gpu_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = self._valid_inputs(Path(tmp))
            result = _run("launch", env)
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn("pure gpus=0,1 mode=reca", result.stdout)
            self.assertIn("t2a gpus=2,3 mode=t2a", result.stdout)
            self.assertIn("mix gpus=4,5 mode=mix", result.stdout)
            self.assertFalse(Path(env["ROOT"]).exists())

    def test_launch_refuses_a_nonempty_formal_arm_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = self._valid_inputs(Path(tmp))
            pure = Path(env["ROOT"]) / "pure"
            pure.mkdir(parents=True)
            (pure / "keep.txt").write_text("keep", encoding="utf-8")
            result = _run("launch", env)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing non-empty output", result.stdout)
            self.assertEqual((pure / "keep.txt").read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
