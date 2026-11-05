from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def _load_prompts_module():
    path = Path(__file__).resolve().parents[1] / "Audio-Omni" / "audio_omni" / "prompts.py"
    spec = importlib.util.spec_from_file_location("reca_test_prompts", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load prompt module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReconstructPromptTests(unittest.TestCase):
    def test_plain_instruction_has_exactly_one_audio_placeholder(self):
        prompt = _load_prompts_module().build_reconstruct_prompt("Reconstruct faithfully")
        self.assertEqual(prompt.count("<|AUDIO|>"), 1)

    def test_empty_instruction_has_exactly_one_audio_placeholder(self):
        prompt = _load_prompts_module().build_reconstruct_prompt("")
        self.assertEqual(prompt.count("<|AUDIO|>"), 1)

    def test_prefixed_instruction_is_normalized_to_one_audio_placeholder(self):
        prompt = _load_prompts_module().build_reconstruct_prompt(
            "  <|AUDIO|>\n<|AUDIO|>  Inspect the sound events"
        )
        self.assertEqual(prompt.count("<|AUDIO|>"), 1)
        self.assertIn("Inspect the sound events", prompt)


if __name__ == "__main__":
    unittest.main()
