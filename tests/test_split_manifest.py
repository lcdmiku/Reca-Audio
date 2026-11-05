from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def _load_split_module():
    path = Path(__file__).resolve().parents[1] / "src" / "reca_audio" / "train" / "split_manifest.py"
    spec = importlib.util.spec_from_file_location("reca_test_split_manifest", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load split module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class SplitManifestTests(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        audio_dir = root / "audio"
        audio_dir.mkdir()
        rows = []
        for index in range(12):
            payload = b"same-audio" if index in (0, 1) else f"audio-{index}".encode()
            path = audio_dir / f"sample-{index}.wav"
            path.write_bytes(payload)
            rows.append({"audio": str(path), "caption": f"caption {index}"})
        manifest = root / "source.jsonl"
        manifest.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        return manifest

    def _assignment(self, out: Path) -> dict[str, str]:
        result = {}
        for split in ("train", "val", "test"):
            for row in _read_jsonl(out / f"{split}_manifest.jsonl"):
                result[Path(row["audio"]).stem] = split
        return result

    def test_split_is_deterministic_and_keeps_duplicate_audio_together(self):
        module = _load_split_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._fixture(root)
            report1 = module.build_split(manifest, root / "split-a", seed=17, ratios=(0.6, 0.2, 0.2))
            report2 = module.build_split(manifest, root / "split-b", seed=17, ratios=(0.6, 0.2, 0.2))

            assignment1 = self._assignment(root / "split-a")
            assignment2 = self._assignment(root / "split-b")
            self.assertEqual(assignment1, assignment2)
            self.assertEqual(assignment1["sample-0"], assignment1["sample-1"])
            self.assertEqual(report1["source_manifest_sha256"], report2["source_manifest_sha256"])

            sha_sets = []
            for split in ("train", "val", "test"):
                hashes = {
                    hashlib.sha256(Path(row["audio"]).read_bytes()).hexdigest()
                    for row in _read_jsonl(root / "split-a" / f"{split}_manifest.jsonl")
                }
                sha_sets.append(hashes)
            self.assertFalse(sha_sets[0] & sha_sets[1])
            self.assertFalse(sha_sets[0] & sha_sets[2])
            self.assertFalse(sha_sets[1] & sha_sets[2])

    def test_prompts_and_reference_files_match_each_held_out_manifest(self):
        module = _load_split_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._fixture(root)
            module.build_split(manifest, root / "split", seed=23, ratios=(0.6, 0.2, 0.2))

            for split in ("val", "test"):
                rows = _read_jsonl(root / "split" / f"{split}_manifest.jsonl")
                prompts = _read_jsonl(root / "split" / f"{split}_prompts.jsonl")
                expected_ids = {Path(row["audio"]).stem for row in rows}
                self.assertEqual({row["id"] for row in prompts}, expected_ids)
                self.assertEqual(
                    {path.stem for path in (root / "split" / f"{split}_ref").glob("*.wav")},
                    expected_ids,
                )

    def test_existing_output_directory_is_never_overwritten(self):
        module = _load_split_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._fixture(root)
            out = root / "split"
            out.mkdir()
            sentinel = out / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                module.build_split(manifest, out, seed=1, ratios=(0.8, 0.1, 0.1))
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()

