import hashlib
import importlib.util
import json
import random
import wave
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "tools" / "audit_reca_dataset.py"


def load_module():
    spec = importlib.util.spec_from_file_location("audit_reca_dataset", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_wav(path: Path, frames: int, sample_rate: int = 10) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\0\0" * frames)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_manifest_summary_distinguishes_rows_paths_hashes_and_captions(tmp_path):
    module = load_module()
    audio_a = tmp_path / "a.wav"
    audio_b = tmp_path / "b.wav"
    write_wav(audio_a, 100)
    audio_b.write_bytes(audio_a.read_bytes())
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(
        manifest,
        [
            {"audio": "a.wav", "caption": "first caption"},
            {"audio": "a.wav", "caption": "second caption"},
            {"audio": str(audio_b), "caption": ""},
        ],
    )

    summary, rows = module.summarize_manifest(manifest, repo_root=tmp_path)

    assert summary == {
        "rows": 3,
        "unique_paths": 2,
        "unique_sha256": 1,
        "empty_captions": 1,
        "missing_files": 0,
    }
    assert rows[0]["resolved_audio"] == str(audio_a.resolve())
    assert rows[0]["sha256"] == hashlib.sha256(audio_a.read_bytes()).hexdigest()


def test_crop_start_matches_training_rng_and_short_clip_padding():
    module = load_module()
    seed = 987654321
    expected = random.Random(seed).randint(0, 220_500)

    assert module.training_crop_start(441_000, 220_500, seed) == expected
    assert module.training_crop_start(100_000, 220_500, seed) == 0


def test_intersection_report_counts_ids_stems_and_hashes_separately(tmp_path):
    module = load_module()
    train_audio = tmp_path / "shared.wav"
    other_audio = tmp_path / "other.wav"
    same_bytes = tmp_path / "renamed.wav"
    write_wav(train_audio, 100)
    write_wav(other_audio, 80)
    same_bytes.write_bytes(train_audio.read_bytes())

    train = [
        {"id": "train-id", "resolved_audio": str(train_audio), "sha256": module.sha256_file(train_audio)},
        {"id": "shared", "resolved_audio": str(other_audio), "sha256": module.sha256_file(other_audio)},
    ]
    benchmark = [
        {"id": "shared", "resolved_audio": str(same_bytes), "sha256": module.sha256_file(same_bytes)},
    ]

    report = module.intersection_report(train, benchmark)

    assert report == {"id": 1, "stem": 0, "sha256": 1}


def test_schedule_audit_reproduces_exposure_crop_and_row_selection(tmp_path):
    module = load_module()
    schedule = tmp_path / "schedule.jsonl"
    write_jsonl(
        schedule,
        [
            {"exposure": 1, "sample_index": 1, "task": "t2a", "data_seed": 7},
            {"exposure": 2, "sample_index": 3, "task": "reca", "data_seed": 8},
        ],
    )
    rows = [
        {"id": "a", "num_frames_44100": 441_000},
        {"id": "b", "num_frames_44100": 300_000},
    ]

    records, summary = module.audit_schedule(schedule, rows, crop_frames=220_500)

    assert [record["row_index"] for record in records] == [1, 1]
    assert records[0]["id"] == "b"
    assert records[0]["crop_start_frame"] == random.Random(7).randint(0, 79_500)
    assert summary["exposures"] == 2
    assert summary["tasks"] == {"reca": 1, "t2a": 1}
    assert summary["unique_rows"] == 1

