from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def _load_schedule_module():
    path = Path(__file__).resolve().parents[1] / "src" / "reca_audio" / "train" / "scheduling.py"
    spec = importlib.util.spec_from_file_location("reca_test_scheduling", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load scheduling module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TaskScheduleTests(unittest.TestCase):
    def test_fixed_modes_do_not_depend_on_draw(self):
        choose = _load_schedule_module().choose_task_mode
        self.assertEqual(choose("reca", 1.0, 0.0, 0.999), "reca")
        self.assertEqual(choose("t2a", 0.0, 1.0, 0.001), "t2a")

    def test_one_to_one_mix_uses_half_open_boundary(self):
        choose = _load_schedule_module().choose_task_mode
        self.assertEqual(choose("mix", 1.0, 1.0, 0.499999), "reca")
        self.assertEqual(choose("mix", 1.0, 1.0, 0.5), "t2a")

    def test_four_to_one_mix_selects_reca_for_first_eighty_percent(self):
        choose = _load_schedule_module().choose_task_mode
        self.assertEqual(choose("mix", 4.0, 1.0, 0.799999), "reca")
        self.assertEqual(choose("mix", 4.0, 1.0, 0.8), "t2a")

    def test_invalid_mix_weights_and_draws_fail_before_training(self):
        choose = _load_schedule_module().choose_task_mode
        for weights in ((0.0, 0.0), (-1.0, 1.0), (1.0, -1.0)):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                choose("mix", *weights, 0.5)
        for draw in (-0.1, 1.0):
            with self.subTest(draw=draw), self.assertRaises(ValueError):
                choose("mix", 1.0, 1.0, draw)
        with self.assertRaises(ValueError):
            choose("unknown", 1.0, 1.0, 0.5)


if __name__ == "__main__":
    unittest.main()

