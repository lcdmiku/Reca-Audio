#!/usr/bin/env python3
"""Build a deterministic one-caption-per-audio AudioCaps evaluation subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_split_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    bad = []
    intersections = report.get("intersections", {})
    for pair in ("train_val", "train_test", "val_test"):
        counts = intersections.get(pair)
        if not isinstance(counts, dict):
            bad.append(f"{pair}=missing")
            continue
        for key in ("id", "sha256"):
            if counts.get(key) != 0:
                bad.append(f"{pair}.{key}={counts.get(key)!r}")
    if bad:
        raise ValueError("non-disjoint split: " + ", ".join(bad))
    return report


def _ref_name(item_id: str, audio_path: Path) -> str:
    if not item_id or Path(item_id).name != item_id or item_id in {".", ".."}:
        raise ValueError(f"unsafe selected id: {item_id!r}")
    suffix = audio_path.suffix or ".wav"
    return f"{item_id}{suffix}"


def build_unique_sha_subset(
    manifest: Path,
    split_report: Path,
    prompts_out: Path,
    refs_out: Path,
    metadata_out: Path,
    expected_count: int,
) -> dict[str, Any]:
    """Select the first manifest row for each audio SHA and publish its artifacts."""
    manifest = Path(manifest)
    split_report = Path(split_report)
    prompts_out = Path(prompts_out)
    refs_out = Path(refs_out)
    metadata_out = Path(metadata_out)

    for target in (prompts_out, refs_out, metadata_out):
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"output target already exists: {target}")
    if expected_count < 1:
        raise ValueError("expected_count must be positive")

    _validate_split_report(split_report)
    selected: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    selected_ids: set[str] = set()
    for row_index, raw_line in enumerate(manifest.read_text(encoding="utf-8").splitlines()):
        line = raw_line.strip()
        if not line:
            continue
        row = json.loads(line)
        item_id = str(row["id"])
        prompt = str(row["prompt"])
        audio_path = Path(row["audio"]).expanduser().resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(f"missing audio file: {audio_path}")
        audio_sha = _sha256_file(audio_path)
        if audio_sha in seen_hashes:
            continue
        if item_id in selected_ids:
            raise ValueError(f"duplicate selected id: {item_id}")
        ref_name = _ref_name(item_id, audio_path)
        seen_hashes.add(audio_sha)
        selected_ids.add(item_id)
        selected.append(
            {
                "row_index": row_index,
                "id": item_id,
                "prompt": prompt,
                "source_audio": str(audio_path),
                "reference_name": ref_name,
                "audio_sha256": audio_sha,
            }
        )

    if len(selected) != expected_count:
        raise ValueError(
            f"expected {expected_count} unique audio hashes, found {len(selected)}"
        )

    prompts_out.parent.mkdir(parents=True, exist_ok=True)
    refs_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    stage_root = Path(
        tempfile.mkdtemp(prefix=".unique-sha-subset-", dir=str(prompts_out.parent))
    )
    stage_prompts = stage_root / "prompts.jsonl"
    stage_refs = stage_root / "ref"
    stage_metadata = stage_root / "selection.json"
    published: list[Path] = []
    try:
        stage_refs.mkdir()
        with stage_prompts.open("w", encoding="utf-8", newline="\n") as handle:
            for item in selected:
                handle.write(
                    json.dumps(
                        {"id": item["id"], "prompt": item["prompt"]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                destination = stage_refs / item["reference_name"]
                source = Path(item["source_audio"])
                try:
                    destination.symlink_to(source)
                except OSError:
                    os.link(source, destination)

        metadata = {
            "count": len(selected),
            "expected_count": expected_count,
            "source_manifest": str(manifest.resolve()),
            "source_manifest_sha256": _sha256_file(manifest),
            "split_report": str(split_report.resolve()),
            "split_report_sha256": _sha256_file(split_report),
            "prompts_sha256": _sha256_file(stage_prompts),
            "selected": selected,
        }
        stage_metadata.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        for staged, target in (
            (stage_refs, refs_out),
            (stage_prompts, prompts_out),
            (stage_metadata, metadata_out),
        ):
            os.replace(staged, target)
            published.append(target)
        return metadata
    except BaseException:
        for target in reversed(published):
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split-report", type=Path, required=True)
    parser.add_argument("--prompts-out", type=Path, required=True)
    parser.add_argument("--refs-out", type=Path, required=True)
    parser.add_argument("--metadata-out", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=89)
    args = parser.parse_args()
    metadata = build_unique_sha_subset(
        args.manifest,
        args.split_report,
        args.prompts_out,
        args.refs_out,
        args.metadata_out,
        args.expected_count,
    )
    print(
        f"[subset] selected={metadata['count']} prompts={args.prompts_out} "
        f"refs={args.refs_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
