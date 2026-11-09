#!/usr/bin/env python3
"""Read-only audit for the exact ReCA manifests and online crop schedules."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_audio(path_text: str, repo_root: str | Path) -> Path:
    path = Path(path_text)
    return path.resolve() if path.is_absolute() else (Path(repo_root) / path).resolve()


def summarize_manifest(
    manifest: str | Path, repo_root: str | Path
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    rows = load_jsonl(manifest)
    hash_cache: dict[Path, str] = {}
    enriched = []
    for index, row in enumerate(rows):
        audio_text = row.get("audio") or row.get("path") or row.get("wav") or ""
        path = resolve_audio(str(audio_text), repo_root) if audio_text else Path()
        exists = bool(audio_text) and path.is_file()
        if exists and path not in hash_cache:
            hash_cache[path] = sha256_file(path)
        caption = str(row.get("caption") or row.get("prompt") or "")
        enriched.append(
            {
                **row,
                "row_index": index,
                "id": str(row.get("id") or (path.stem if audio_text else index)),
                "caption": caption,
                "resolved_audio": str(path) if audio_text else "",
                "sha256": hash_cache.get(path, ""),
            }
        )
    summary = {
        "rows": len(enriched),
        "unique_paths": len({row["resolved_audio"] for row in enriched if row["resolved_audio"]}),
        "unique_sha256": len({row["sha256"] for row in enriched if row["sha256"]}),
        "empty_captions": sum(not row["caption"].strip() for row in enriched),
        "missing_files": sum(not row["sha256"] for row in enriched),
    }
    return summary, enriched


def training_crop_start(num_frames: int, crop_frames: int, data_seed: int) -> int:
    if num_frames <= crop_frames:
        return 0
    return random.Random(int(data_seed)).randint(0, num_frames - crop_frames)


def intersection_report(
    left: Sequence[dict[str, Any]], right: Sequence[dict[str, Any]]
) -> dict[str, int]:
    def values(rows: Sequence[dict[str, Any]], key: str) -> set[str]:
        if key == "stem":
            return {
                Path(str(row.get("resolved_audio") or "")).stem
                for row in rows
                if row.get("resolved_audio")
            }
        return {str(row.get(key)) for row in rows if row.get(key) not in (None, "")}

    return {
        key: len(values(left, key) & values(right, key))
        for key in ("id", "stem", "sha256")
    }


def audit_schedule(
    schedule: str | Path,
    manifest_rows: Sequence[dict[str, Any]],
    crop_frames: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = []
    task_counts: Counter[str] = Counter()
    for scheduled in load_jsonl(schedule):
        row_index = int(scheduled["sample_index"]) % len(manifest_rows)
        row = manifest_rows[row_index]
        start = training_crop_start(
            int(row["num_frames_44100"]), crop_frames, int(scheduled["data_seed"])
        )
        task = str(scheduled.get("task") or "")
        task_counts[task] += 1
        records.append(
            {
                **scheduled,
                "row_index": row_index,
                "id": row.get("id", ""),
                "caption": row.get("caption", ""),
                "resolved_audio": row.get("resolved_audio", ""),
                "sha256": row.get("sha256", ""),
                "crop_start_frame": start,
                "crop_start_seconds": start / 44100.0,
                "crop_end_seconds": (start + crop_frames) / 44100.0,
            }
        )
    summary = {
        "exposures": len(records),
        "tasks": dict(sorted(task_counts.items())),
        "unique_rows": len({row["row_index"] for row in records}),
        "unique_paths": len({row["resolved_audio"] for row in records}),
        "unique_sha256": len({row["sha256"] for row in records}),
    }
    return records, summary


def _audio_inventory(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import numpy as np
    import soundfile as sf

    caption_counts = Counter(row["resolved_audio"] for row in rows)
    inventory = []
    for path_text in sorted(caption_counts):
        path = Path(path_text)
        if not path.is_file():
            inventory.append({"resolved_audio": path_text, "error": "missing"})
            continue
        try:
            info = sf.info(str(path))
            wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
            mono = wav.mean(axis=1)
            peak = float(np.max(np.abs(mono))) if mono.size else 0.0
            rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
            frame = max(1, int(sr * 0.1))
            usable = mono[: mono.size - mono.size % frame]
            if usable.size:
                frame_rms = np.sqrt(np.mean(usable.reshape(-1, frame) ** 2, axis=1))
                silent_fraction = float(np.mean(frame_rms < 10 ** (-50 / 20)))
            else:
                silent_fraction = 1.0
            inventory.append(
                {
                    "resolved_audio": path_text,
                    "id": path.stem,
                    "sha256": sha256_file(path),
                    "caption_rows": caption_counts[path_text],
                    "sample_rate": int(info.samplerate),
                    "channels": int(info.channels),
                    "frames": int(info.frames),
                    "duration_seconds": float(info.duration),
                    "format": info.format,
                    "subtype": info.subtype,
                    "num_frames_44100": int(round(info.frames * 44100 / info.samplerate)),
                    "peak": peak,
                    "rms_dbfs": 20 * math.log10(max(rms, 1e-12)),
                    "dc_offset": float(np.mean(mono)) if mono.size else 0.0,
                    "clipped_fraction": float(np.mean(np.abs(mono) >= 0.999)) if mono.size else 0.0,
                    "silent_100ms_fraction": silent_fraction,
                    "error": "",
                }
            )
        except Exception as exc:  # noqa: BLE001
            inventory.append({"resolved_audio": path_text, "error": repr(exc)})

    good = [row for row in inventory if not row.get("error")]
    good_by_sha = {row["sha256"]: row for row in good}
    unique_good = list(good_by_sha.values())
    summary = {
        "paths": len(inventory),
        "readable_paths": len(good),
        "unique_sha256": len(unique_good),
        "sample_rates": dict(sorted(Counter(row["sample_rate"] for row in good).items())),
        "channels": dict(sorted(Counter(row["channels"] for row in good).items())),
        "duration_seconds": _distribution([row["duration_seconds"] for row in good]),
        "unique_sha_duration_seconds": _distribution([row["duration_seconds"] for row in unique_good]),
        "rms_dbfs": _distribution([row["rms_dbfs"] for row in good]),
        "unique_sha_rms_dbfs": _distribution([row["rms_dbfs"] for row in unique_good]),
        "silent_100ms_fraction": _distribution([row["silent_100ms_fraction"] for row in good]),
        "unique_sha_silent_100ms_fraction": _distribution([row["silent_100ms_fraction"] for row in unique_good]),
        "path_issue_counts": {
            "shorter_than_5s": sum(row["duration_seconds"] < 5.0 for row in good),
            "near_silent": sum(row["rms_dbfs"] < -45.0 for row in good),
            "clipped_over_0_1pct": sum(row["clipped_fraction"] > 0.001 for row in good),
            "more_than_half_silent": sum(row["silent_100ms_fraction"] > 0.5 for row in good),
        },
        "unique_sha_issue_counts": {
            "shorter_than_5s": sum(row["duration_seconds"] < 5.0 for row in unique_good),
            "near_silent": sum(row["rms_dbfs"] < -45.0 for row in unique_good),
            "clipped_over_0_1pct": sum(row["clipped_fraction"] > 0.001 for row in unique_good),
            "more_than_half_silent": sum(row["silent_100ms_fraction"] > 0.5 for row in unique_good),
        },
    }
    return inventory, summary


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "min": None, "p05": None, "median": None, "p95": None, "max": None, "mean": None}
    ordered = sorted(float(value) for value in values)

    def quantile(q: float) -> float:
        position = (len(ordered) - 1) * q
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)

    return {
        "n": len(ordered),
        "min": ordered[0],
        "p05": quantile(0.05),
        "median": quantile(0.5),
        "p95": quantile(0.95),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def summarize_alignment(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    crop = [float(row["caption_crop_clap"]) for row in rows]
    full = [float(row["caption_full_clap"]) for row in rows]
    delta = [crop_score - full_score for crop_score, full_score in zip(crop, full)]
    sequential = [
        float(row["caption_crop_clap"])
        for row in rows
        if row.get("sequential_caption")
    ]
    non_sequential = [
        float(row["caption_crop_clap"])
        for row in rows
        if not row.get("sequential_caption")
    ]
    crop_distribution = _distribution(crop)
    return {
        "exposures": len(rows),
        "unique_sha256": len({row.get("sha256") for row in rows if row.get("sha256")}),
        "crop_score": crop_distribution,
        "full_score": _distribution(full),
        "crop_minus_full": _distribution(delta),
        "crop_drop_over_0_15": sum(value < -0.15 for value in delta),
        "p05_problem_threshold": crop_distribution["p05"],
        "below_p05": sum(value < float(crop_distribution["p05"]) for value in crop),
        "sequential": _distribution(sequential),
        "non_sequential": _distribution(non_sequential),
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _parse_mapping(items: Iterable[str]) -> dict[str, Path]:
    result = {}
    for item in items:
        label, value = item.split("=", 1)
        result[label] = Path(value)
    return result


def _attach_inventory(rows: list[dict[str, Any]], inventory: Sequence[dict[str, Any]]) -> None:
    lookup = {row["resolved_audio"]: row for row in inventory}
    for row in rows:
        details = lookup.get(row["resolved_audio"], {})
        row["num_frames_44100"] = int(details.get("num_frames_44100") or 0)


def _benchmark_rows(prompt_path: Path, ref_dir: Path, repo_root: Path) -> list[dict[str, Any]]:
    rows = []
    for row in load_jsonl(prompt_path):
        identifier = str(row.get("id") or "")
        candidates = list(ref_dir.glob(f"{identifier}.*"))
        path = candidates[0].resolve() if len(candidates) == 1 else Path()
        rows.append(
            {
                "id": identifier,
                "caption": str(row.get("prompt") or row.get("caption") or ""),
                "resolved_audio": str(path) if path else "",
                "sha256": sha256_file(path) if path else "",
            }
        )
    return rows


def _sequential_caption(caption: str) -> bool:
    text = f" {caption.casefold()} "
    return any(token in text for token in (" followed by ", " then ", " before ", " after ", " while ", " as "))


def _clap_alignment(
    schedule_rows: list[dict[str, Any]],
    crop_frames: int,
    device: str,
    temp_root: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    from runs.Benchmark.common.metrics import get_clap

    model = get_clap(device)
    texts = [row["caption"] or "audio" for row in schedule_rows]
    text_embeddings = []
    for index in range(0, len(texts), 32):
        text_embeddings.append(np.asarray(model.get_text_embedding(texts[index:index + 32], use_tensor=False)))
    text_embeddings = np.concatenate(text_embeddings)
    text_embeddings /= np.linalg.norm(text_embeddings, axis=1, keepdims=True) + 1e-8

    with tempfile.TemporaryDirectory(prefix="reca-data-audit-", dir=temp_root or None) as temp:
        crop_paths = []
        for index, row in enumerate(schedule_rows):
            wav, sr = torchaudio.load(row["resolved_audio"])
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            if sr != 44100:
                wav = torchaudio.functional.resample(wav, sr, 44100)
            start = int(row["crop_start_frame"])
            crop = wav[..., start:start + crop_frames]
            if crop.shape[-1] < crop_frames:
                crop = torch.nn.functional.pad(crop, (0, crop_frames - crop.shape[-1]))
            crop_path = Path(temp) / f"{index:06d}.wav"
            sf.write(crop_path, crop.squeeze(0).numpy(), 44100, subtype="PCM_16")
            crop_paths.append(str(crop_path))

        audio_embeddings = []
        for index in range(0, len(crop_paths), 32):
            audio_embeddings.append(
                np.asarray(model.get_audio_embedding_from_filelist(x=crop_paths[index:index + 32], use_tensor=False))
            )
        audio_embeddings = np.concatenate(audio_embeddings)
        audio_embeddings /= np.linalg.norm(audio_embeddings, axis=1, keepdims=True) + 1e-8
        crop_scores = np.sum(audio_embeddings * text_embeddings, axis=1)

        unique_full: dict[str, str] = {}
        for row in schedule_rows:
            key = str(row.get("sha256") or row["resolved_audio"])
            unique_full.setdefault(key, row["resolved_audio"])
        full_keys = list(unique_full)
        full_embeddings = []
        full_files = [unique_full[key] for key in full_keys]
        for index in range(0, len(full_files), 32):
            full_embeddings.append(
                np.asarray(model.get_audio_embedding_from_filelist(x=full_files[index:index + 32], use_tensor=False))
            )
        full_embeddings = np.concatenate(full_embeddings)
        full_embeddings /= np.linalg.norm(full_embeddings, axis=1, keepdims=True) + 1e-8
        full_lookup = dict(zip(full_keys, full_embeddings))
        scheduled_full = np.stack(
            [full_lookup[str(row.get("sha256") or row["resolved_audio"])] for row in schedule_rows]
        )
        full_scores = np.sum(scheduled_full * text_embeddings, axis=1)

    output = []
    for row, crop_score, full_score in zip(
        schedule_rows, crop_scores.tolist(), full_scores.tolist()
    ):
        output.append(
            {
                **row,
                "caption_crop_clap": float(crop_score),
                "caption_full_clap": float(full_score),
                "crop_minus_full_clap": float(crop_score - full_score),
                "sequential_caption": _sequential_caption(row["caption"]),
            }
        )
    return output, summarize_alignment(output)


def _problems(
    inventory: Sequence[dict[str, Any]], alignment: Sequence[dict[str, Any]], threshold: float | None
) -> list[dict[str, Any]]:
    problems = []
    for row in inventory:
        issues = []
        if row.get("error"):
            issues.append("unreadable_audio")
        else:
            if row["duration_seconds"] < 5.0:
                issues.append("shorter_than_training_crop")
            if row["rms_dbfs"] < -45.0:
                issues.append("near_silent")
            if row["clipped_fraction"] > 0.001:
                issues.append("clipping_over_0.1pct")
            if row["silent_100ms_fraction"] > 0.5:
                issues.append("more_than_half_silent")
        if issues:
            problems.append({"scope": "audio", "issues": ";".join(issues), **row})
    if threshold is not None:
        for row in alignment:
            issues = []
            if row["caption_crop_clap"] < threshold:
                issues.append("bottom_5pct_caption_crop_clap")
            if row.get("crop_minus_full_clap", 0.0) < -0.15:
                issues.append("crop_clap_drop_over_0.15")
            if issues:
                problems.append(
                    {
                        "scope": "scheduled_crop",
                        "issues": ";".join(issues),
                        **row,
                    }
                )
    return problems


def _markdown_report(summary: dict[str, Any], problems: Sequence[dict[str, Any]]) -> str:
    manifests = summary["manifests"]
    audio = summary["audio"]
    lines = [
        "# ReCA Training Data Audit",
        "",
        f"- Source rows: {manifests['source']['rows']}",
        f"- Unique source paths / SHA256: {manifests['source']['unique_paths']} / {manifests['source']['unique_sha256']}",
        f"- Train/val/test rows: {manifests['train']['rows']} / {manifests['val']['rows']} / {manifests['test']['rows']}",
        f"- Readable paths / unique SHA256: {audio['readable_paths']} / {audio['unique_sha256']}",
        f"- Duration distribution: `{json.dumps(audio['duration_seconds'], ensure_ascii=False)}`",
        f"- RMS dBFS distribution: `{json.dumps(audio['rms_dbfs'], ensure_ascii=False)}`",
        f"- Problem records / unique SHA256: {len(problems)} / {summary['problem_unique_sha256']}",
        "",
        "## Schedule coverage",
        "",
    ]
    for label, record in summary["schedules"].items():
        lines.append(f"- {label}: `{json.dumps(record, ensure_ascii=False)}`")
    lines.extend(["", "## Benchmark intersections", ""])
    for label, record in summary["benchmark_intersections"].items():
        lines.append(f"- {label}: `{json.dumps(record, ensure_ascii=False)}`")
    if summary.get("caption_crop_alignment"):
        lines.extend(
            [
                "",
                "## Caption-crop alignment",
                "",
                f"`{json.dumps(summary['caption_crop_alignment'], ensure_ascii=False)}`",
                "",
                "CLAP is a triage signal only. Bottom-5% rows require listening/manual review; they are not automatically mislabeled.",
            ]
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--schedule", action="append", default=[], help="LABEL=PATH")
    parser.add_argument("--benchmark", action="append", default=[], help="LABEL=PROMPTS_JSONL")
    parser.add_argument("--benchmark-ref", action="append", default=[], help="LABEL=REF_DIR")
    parser.add_argument("--alignment-schedule", default="")
    parser.add_argument("--clip-seconds", type=float, default=5.0)
    parser.add_argument("--clap-device", default="")
    parser.add_argument("--temp-root", default="/tmp")
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    manifests = {}
    manifest_rows = {}
    for label, path in (
        ("source", args.source_manifest),
        ("train", args.train_manifest),
        ("val", args.val_manifest),
        ("test", args.test_manifest),
    ):
        manifests[label], manifest_rows[label] = summarize_manifest(path, repo_root)

    inventory, audio_summary = _audio_inventory(manifest_rows["source"])
    inventory_lookup = {row["resolved_audio"]: row for row in inventory}
    for rows in manifest_rows.values():
        _attach_inventory(rows, inventory)

    crop_frames = int(round(44100 * args.clip_seconds))
    schedules = {}
    schedule_records = {}
    for label, path in _parse_mapping(args.schedule).items():
        schedule_records[label], schedules[label] = audit_schedule(path, manifest_rows["train"], crop_frames)

    prompt_maps = _parse_mapping(args.benchmark)
    ref_maps = _parse_mapping(args.benchmark_ref)
    intersections = {}
    for label, prompt_path in prompt_maps.items():
        benchmark = _benchmark_rows(prompt_path, ref_maps[label], repo_root)
        intersections[label] = {
            "benchmark_rows": len(benchmark),
            "resolved_audio": sum(bool(row["sha256"]) for row in benchmark),
            "vs_train": intersection_report(manifest_rows["train"], benchmark),
            "vs_val": intersection_report(manifest_rows["val"], benchmark),
            "vs_test": intersection_report(manifest_rows["test"], benchmark),
        }

    alignment = []
    alignment_summary = {}
    if args.clap_device:
        alignment_label = args.alignment_schedule or next(iter(schedule_records))
        alignment, alignment_summary = _clap_alignment(
            schedule_records[alignment_label], crop_frames, args.clap_device, args.temp_root
        )

    threshold = alignment_summary.get("p05_problem_threshold") if alignment_summary else None
    problems = _problems(inventory, alignment, threshold)
    summary = {
        "contract": {
            "repo_root": str(repo_root),
            "clip_seconds": args.clip_seconds,
            "crop_sample_rate": 44100,
            "source_manifest_sha256": sha256_file(args.source_manifest),
        },
        "manifests": manifests,
        "audio": audio_summary,
        "schedules": schedules,
        "benchmark_intersections": intersections,
        "caption_crop_alignment": alignment_summary,
        "problem_counts": dict(sorted(Counter(row["issues"] for row in problems).items())),
        "problem_unique_sha256": len({row.get("sha256") for row in problems if row.get("sha256")}),
    }
    _write_csv(output / "audio_inventory.csv", inventory)
    _write_csv(output / "problem_samples.csv", problems)
    for label, rows in schedule_records.items():
        _write_csv(output / f"schedule_{label}.csv", rows)
    if alignment:
        _write_csv(output / "caption_crop_alignment.csv", alignment)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "report.md").write_text(_markdown_report(summary, problems), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
