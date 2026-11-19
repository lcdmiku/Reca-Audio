"""Stable-partition the archived low-LR Mix schedule into ReCA then T2A."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from reca_audio.train.gradient_experiment import load_schedule, write_schedule


MIX_SHA256 = "bd5ca9ffa653c06af7d0c1eefb1e36e2fbf6ed09f8bf1b496afb9da1107126a0"
TASKS = ("reca", "t2a")
PHASE = 3536
ACCUMULATION = 4


def prepare(source, coverage, out):
    provenance = json.loads(coverage.read_text(encoding="utf-8"))["mix_7070"]
    sidecar = source.with_suffix(source.suffix + ".sha256")
    if provenance["schedule_sha256"] != MIX_SHA256 or sidecar.read_text().strip() != MIX_SHA256:
        raise ValueError("Source coverage or sidecar does not match archived Mix schedule")
    original = load_schedule(source, expected_digest=MIX_SHA256)
    if len(original) != 2 * PHASE or any(row["exposure"] != index for index, row in enumerate(original, 1)):
        raise ValueError("Invalid historical exposure positions")
    for start in range(0, len(original), ACCUMULATION):
        if Counter(row["task"] for row in original[start:start + ACCUMULATION]) != {"reca": 2, "t2a": 2}:
            raise ValueError(f"Unbalanced source update at exposure {start + 1}")
    phases = {task: [row for row in original if row["task"] == task] for task in TASKS}
    counts = {task: len(rows) for task, rows in phases.items()}
    unique = {task: len({row["sample_index"] for row in rows}) for task, rows in phases.items()}
    if (counts != provenance["exposures"] or unique != provenance["unique_audio_by_task"]
            or len({row["sample_index"] for row in original}) != provenance["unique_audio"]):
        raise ValueError("Source task counts or sample coverage disagree with the audit")
    ordered = phases["reca"] + phases["t2a"]
    records = [dict(row, exposure=index) for index, row in enumerate(ordered, 1)]
    out.parent.mkdir(parents=True, exist_ok=True)
    digest_path = out.with_suffix(out.suffix + ".sha256")
    audit_path = out.with_suffix(out.suffix + ".audit.json")
    created = []
    try:
        for dest in (out, digest_path, audit_path):
            dest.touch(exist_ok=False)
            created.append(dest)
        digest = write_schedule(out, records)
        if load_schedule(out, expected_digest=digest) != records:
            raise ValueError("Schedule did not round-trip")
        digest_path.write_text(digest + "\n", encoding="ascii")
        audit = dict(source=str(source), source_sha256=MIX_SHA256, coverage=str(coverage),
                     coverage_sha256=hashlib.sha256(coverage.read_bytes()).hexdigest(),
                     schedule_sha256=digest, phase_order=list(TASKS), exposures=counts,
                     planned_updates_per_phase=PHASE // ACCUMULATION, unique_audio=provenance["unique_audio"],
                     unique_audio_by_task=unique, manifest=provenance["manifest"])
        audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    except Exception:
        for dest in created:
            dest.unlink()
        raise
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("--source", "--coverage", "--out"):
        parser.add_argument(flag, required=True, type=Path)
    args = parser.parse_args()
    prepare(args.source, args.coverage, args.out)
