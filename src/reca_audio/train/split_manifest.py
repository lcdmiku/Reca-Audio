"""Create a deterministic, content-hash-grouped train/val/test manifest split."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _resolve_audio(path_text: str, manifest: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    from_cwd = (Path.cwd() / path).resolve()
    if from_cwd.is_file():
        return from_cwd
    return (manifest.parent / path).resolve()


def _split_group_counts(group_count: int, ratios: Sequence[float]) -> tuple[int, int, int]:
    if group_count < 3:
        raise ValueError("at least three unique audio hashes are required")
    if len(ratios) != 3 or any(value <= 0 for value in ratios):
        raise ValueError("ratios must contain three positive values")
    if abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("ratios must sum to 1")

    train = max(1, min(group_count - 2, round(group_count * ratios[0])))
    val = max(1, min(group_count - train - 1, round(group_count * ratios[1])))
    test = group_count - train - val
    return train, val, test


def _link_audio(source: Path, destination: Path) -> str:
    try:
        os.symlink(source.resolve(), destination)
        return "symlink"
    except OSError:
        os.link(source.resolve(), destination)
        return "hardlink"


def build_split(
    source_manifest: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
) -> dict:
    """Build a split without allowing identical audio bytes to cross splits."""
    manifest = Path(source_manifest).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing split directory: {output}")
    if not manifest.is_file():
        raise FileNotFoundError(manifest)

    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("source manifest is empty")

    groups: dict[str, list[dict]] = defaultdict(list)
    for index, row in enumerate(rows):
        audio_text = row.get("audio") or row.get("path") or row.get("wav")
        if not audio_text:
            raise KeyError(f"manifest row {index} has no audio/path/wav field")
        audio_path = _resolve_audio(str(audio_text), manifest)
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        content_hash = _sha256_file(audio_path)
        groups[content_hash].append(
            {
                "row": row,
                "audio_path": audio_path,
                "id": audio_path.stem,
                "index": index,
            }
        )

    ordered_hashes = sorted(
        groups,
        key=lambda content_hash: hashlib.sha256(
            f"{seed}:{content_hash}".encode("utf-8")
        ).hexdigest(),
    )
    train_groups, val_groups, _ = _split_group_counts(len(ordered_hashes), ratios)
    hash_partitions = {
        "train": ordered_hashes[:train_groups],
        "val": ordered_hashes[train_groups : train_groups + val_groups],
        "test": ordered_hashes[train_groups + val_groups :],
    }

    output.mkdir(parents=True)
    split_rows: dict[str, list[dict]] = {}
    split_ids: dict[str, list[str]] = {}
    link_modes: dict[str, dict[str, int]] = {"val": defaultdict(int), "test": defaultdict(int)}

    for split, hashes in hash_partitions.items():
        entries = sorted(
            (entry for content_hash in hashes for entry in groups[content_hash]),
            key=lambda entry: entry["index"],
        )
        split_rows[split] = [entry["row"] for entry in entries]
        split_ids[split] = [entry["id"] for entry in entries]
        _write_jsonl(output / f"{split}_manifest.jsonl", split_rows[split])

        if split in ("val", "test"):
            prompt_rows = [
                {
                    "id": entry["id"],
                    "prompt": str(
                        entry["row"].get("caption")
                        or entry["row"].get("prompt")
                        or "audio"
                    ),
                }
                for entry in entries
            ]
            _write_jsonl(output / f"{split}_prompts.jsonl", prompt_rows)
            ref_dir = output / f"{split}_ref"
            ref_dir.mkdir()
            seen_names: set[str] = set()
            for entry in entries:
                name = entry["audio_path"].name
                if name in seen_names:
                    raise ValueError(f"duplicate reference filename in {split}: {name}")
                seen_names.add(name)
                mode = _link_audio(entry["audio_path"], ref_dir / name)
                link_modes[split][mode] += 1

    intersections = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        intersections[f"{left}_{right}"] = {
            "id": len(set(split_ids[left]) & set(split_ids[right])),
            "sha256": len(set(hash_partitions[left]) & set(hash_partitions[right])),
        }

    report = {
        "source_manifest": str(manifest),
        "source_manifest_sha256": _sha256_file(manifest),
        "seed": seed,
        "ratios": list(ratios),
        "total_rows": len(rows),
        "unique_audio_sha256": len(groups),
        "splits": {},
        "intersections": intersections,
        "link_modes": {split: dict(counts) for split, counts in link_modes.items()},
    }
    for split in ("train", "val", "test"):
        manifest_path = output / f"{split}_manifest.jsonl"
        report["splits"][split] = {
            "rows": len(split_rows[split]),
            "unique_ids": len(set(split_ids[split])),
            "unique_sha256": len(hash_partitions[split]),
            "ids": sorted(set(split_ids[split])),
            "sha256": sorted(hash_partitions[split]),
            "manifest_sha256": _sha256_file(manifest_path),
        }
    (output / "split_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    args = parser.parse_args()
    report = build_split(
        args.manifest,
        args.out,
        seed=args.seed,
        ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

