from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "runs" / "Benchmark" / "AudioCaps" / "build_unique_sha_subset.py"
SPEC = importlib.util.spec_from_file_location("reca_unique_sha_subset", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
build_unique_sha_subset = MODULE.build_unique_sha_subset


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class UniqueShaSubsetTests(unittest.TestCase):
    def _fixture(self, root: Path) -> dict[str, Path]:
        audio = root / "audio"
        audio.mkdir()
        first = audio / "first.wav"
        duplicate = audio / "duplicate.wav"
        second = audio / "second.flac"
        first.write_bytes(b"same-audio")
        duplicate.write_bytes(b"same-audio")
        second.write_bytes(b"different-audio")

        manifest = root / "test_manifest.jsonl"
        rows = [
            {"caption": "first caption", "audio": str(first)},
            {"caption": "duplicate caption", "audio": str(duplicate)},
            {"caption": "second caption", "audio": str(second)},
        ]
        manifest.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        report = root / "split_report.json"
        report.write_text(
            json.dumps(
                {
                    "intersections": {
                        "train_val": {"id": 0, "sha256": 0},
                        "train_test": {"id": 0, "sha256": 0},
                        "val_test": {"id": 0, "sha256": 0},
                    },
                    "partitions": {"test": {"unique_sha256": 2}},
                }
            ),
            encoding="utf-8",
        )
        return {
            "manifest": manifest,
            "report": report,
            "prompts": root / "derived" / "prompts.jsonl",
            "refs": root / "derived" / "ref",
            "metadata": root / "derived" / "selection.json",
            "first": first,
            "second": second,
        }

    def _build(self, paths: dict[str, Path], expected_count: int = 2) -> dict:
        return build_unique_sha_subset(
            paths["manifest"],
            paths["report"],
            paths["prompts"],
            paths["refs"],
            paths["metadata"],
            expected_count,
        )

    def test_selects_first_row_per_sha_and_records_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._fixture(Path(tmp))

            metadata = self._build(paths)

            prompts = [
                json.loads(line)
                for line in paths["prompts"].read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                prompts,
                [
                    {"id": "first", "prompt": "first caption"},
                    {"id": "second", "prompt": "second caption"},
                ],
            )
            first_link = paths["refs"] / "first.wav"
            second_link = paths["refs"] / "second.flac"
            self.assertTrue(first_link.samefile(paths["first"]))
            self.assertTrue(second_link.samefile(paths["second"]))
            if first_link.is_symlink():
                self.assertTrue(first_link.readlink().is_absolute())
            if second_link.is_symlink():
                self.assertTrue(second_link.readlink().is_absolute())
            self.assertEqual(metadata["count"], 2)
            self.assertEqual(metadata["expected_count"], 2)
            self.assertEqual(metadata["source_manifest_sha256"], _sha256(paths["manifest"]))
            self.assertEqual(metadata["prompts_sha256"], _sha256(paths["prompts"]))
            self.assertEqual(
                [item["id"] for item in metadata["selected"]], ["first", "second"]
            )
            self.assertEqual(
                [item["audio_sha256"] for item in metadata["selected"]],
                [_sha256(paths["first"]), _sha256(paths["second"])],
            )
            self.assertEqual(
                json.loads(paths["metadata"].read_text(encoding="utf-8")), metadata
            )

    def test_rejects_missing_audio_without_publishing_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._fixture(Path(tmp))
            paths["second"].unlink()

            with self.assertRaisesRegex(FileNotFoundError, "second.flac"):
                self._build(paths)

            self.assertFalse(paths["prompts"].exists())
            self.assertFalse(paths["refs"].exists())
            self.assertFalse(paths["metadata"].exists())

    def test_rejects_duplicate_ids_for_distinct_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._fixture(Path(tmp))
            rows = [json.loads(line) for line in paths["manifest"].read_text().splitlines()]
            rows[2]["id"] = "first"
            paths["manifest"].write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "duplicate selected id"):
                self._build(paths)

            self.assertFalse(paths["prompts"].exists())

    def test_rejects_nonzero_split_intersection(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._fixture(Path(tmp))
            report = json.loads(paths["report"].read_text(encoding="utf-8"))
            report["intersections"]["train_test"]["sha256"] = 1
            paths["report"].write_text(json.dumps(report), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "train_test.sha256=1"):
                self._build(paths)

            self.assertFalse(paths["prompts"].exists())

    def test_rejects_nonempty_output_target_without_overwriting_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._fixture(Path(tmp))
            paths["refs"].mkdir(parents=True)
            sentinel = paths["refs"] / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "output target"):
                self._build(paths)

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
            self.assertFalse(paths["prompts"].exists())
            self.assertFalse(paths["metadata"].exists())

    def test_rejects_expected_count_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._fixture(Path(tmp))

            with self.assertRaisesRegex(ValueError, "expected 3 unique audio hashes, found 2"):
                self._build(paths, expected_count=3)

            self.assertFalse(paths["prompts"].exists())
            self.assertFalse(paths["refs"].exists())
            self.assertFalse(paths["metadata"].exists())


if __name__ == "__main__":
    unittest.main()
