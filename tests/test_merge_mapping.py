from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path


class FakeTensor:
    def __init__(self, shape):
        self.shape = shape


def _load_merge_module():
    path = Path(__file__).resolve().parents[1] / "src" / "reca_audio" / "train" / "merge_ckpt.py"
    spec = importlib.util.spec_from_file_location("reca_test_merge", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load merge module from {path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get("torch")
    sys.modules["torch"] = types.SimpleNamespace()
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            del sys.modules["torch"]
        else:
            sys.modules["torch"] = previous
    return module


class MergeMappingTests(unittest.TestCase):
    def test_complete_mapping_returns_every_dit_replacement(self):
        prepare = _load_merge_module().prepare_dit_replacements
        full = {
            "model.block.weight": FakeTensor((2, 3)),
            "model.block.bias": FakeTensor((2,)),
            "conditioner.proj.weight": FakeTensor((4, 4)),
        }
        dit = {
            "block.weight": FakeTensor((2, 3)),
            "block.bias": FakeTensor((2,)),
        }
        replacements = prepare(full, dit, strict=True)
        self.assertEqual(set(replacements), {"model.block.weight", "model.block.bias"})
        self.assertIs(replacements["model.block.weight"], dit["block.weight"])

    def test_missing_dit_key_for_base_model_fails_in_strict_mode(self):
        prepare = _load_merge_module().prepare_dit_replacements
        full = {
            "model.block.weight": FakeTensor((2, 3)),
            "model.block.bias": FakeTensor((2,)),
        }
        dit = {"block.weight": FakeTensor((2, 3))}
        with self.assertRaisesRegex(KeyError, "uncovered base DiT"):
            prepare(full, dit, strict=True)

    def test_shape_mismatch_fails_before_mutating_checkpoint(self):
        prepare = _load_merge_module().prepare_dit_replacements
        full = {"model.block.weight": FakeTensor((2, 3))}
        dit = {"block.weight": FakeTensor((3, 2))}
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            prepare(full, dit, strict=True)


if __name__ == "__main__":
    unittest.main()

