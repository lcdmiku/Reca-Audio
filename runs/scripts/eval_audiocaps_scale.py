"""Evaluate the AudioCaps scale matrix on the preceding audit's paired inputs."""

import argparse
import json
from pathlib import Path
import statistics

import numpy as np

import diagnose_posttrain as audit

TRAIN = Path("runs/train/audiocaps_scale_matrix_v2")
LOW_LR_TRAIN = Path("runs/train/audiocaps_low_lr_v1")
HIGH_LR_RESULTS = Path("runs/audits/audiocaps_scale_eval_v2/results.json")
HIGH_LR_SHA = "0874485c2fffa85998341b26e4d0a9cedbc2ebeeacd666ee7d73cae8e5d56ddd"
PRIOR = Path("runs/audits/posttrain_four_arm_v1")
ARMS = ("base", "pure_7070", "t2a_7070", "mix_7070", "t2a_707")
SETTINGS = ("audio_10_cfg6", "audio_10_cfg1", "music_10_cfg6", "reca_matched", "reca_mismatched")


def prepare(root, low_lr=False):
    train = LOW_LR_TRAIN if low_lr else TRAIN
    if not (train / "COMPLETE").is_file() or not (PRIOR / "COMPLETE").is_file():
        raise ValueError("training and prior evaluation must be complete")
    protocol = audit.load_protocol(PRIOR)
    if protocol["smoke"] or protocol["steps"] != 100 or len(protocol["audio"]) != 89:
        raise ValueError("unexpected reference protocol")
    coverage = json.loads((train / ("source_coverage.json" if low_lr else "coverage.json")).read_text())
    checkpoints = {"base": protocol["checkpoints"]["base"]}
    trajectory = {}
    if low_lr:
        if audit.sha(HIGH_LR_RESULTS) != HIGH_LR_SHA:
            raise ValueError("historical high-LR results digest mismatch")
        high_protocol = json.loads(HIGH_LR_RESULTS.read_text())["protocol"]
        if any(high_protocol[key] != protocol[key] for key in ("audio", "music", "seed", "steps", "dtype", "sample_rate")):
            raise ValueError("historical high-LR pairing protocol mismatch")
        if high_protocol["settings"] != {key: list(audit.SETTINGS[key]) for key in SETTINGS}:
            raise ValueError("historical generation settings mismatch")
        base_sha = (train / "inputs.sha256").read_text().splitlines()[0].split()[0]
        if not base_sha == checkpoints["base"]["sha256"] == high_protocol["checkpoints"]["base"]["sha256"]:
            raise ValueError("training and reference Base digest mismatch")
        coverage = {arm: coverage[arm] for arm in audit.ARMS[1:]}
        protocol["reference_git_commit"] = protocol.pop("git_commit")
        protocol["git_commit"] = dict(line.split("=", 1) for line in Path("DEPLOYED_COMMIT").read_text().splitlines())["commit"]
        protocol["high_lr_reference"] = dict(path=str(HIGH_LR_RESULTS), sha256=HIGH_LR_SHA)
    for arm in audit.ARMS[1:]:
        path = train / arm / "dit_exposure7072.pt"
        schedule = train / f"{arm}_schedule.jsonl"
        schedule_sha = audit.sha(schedule)
        if schedule_sha != coverage[arm]["schedule_sha256"]:
            raise ValueError(f"schedule digest mismatch: {arm}")
        contract = json.loads((train / arm / "contract.json").read_text())
        if low_lr:
            previous = high_protocol["checkpoints"][arm]
            expected = dict(previous["contract"], peak_lr=5e-6, schedule=str(schedule))
            if contract != expected or previous["schedule_sha256"] != schedule_sha:
                raise ValueError(f"historical training contract or schedule mismatch: {arm}")
        rows = audit.audio_io.read_jsonl(train / arm / "metrics.jsonl")
        if not any(r["kind"] == "checkpoint" and r["exposure"] == 7072
                   and r["optimizer_step"] == 1768 and r["reload_validated"] for r in rows):
            raise ValueError(f"unverified final checkpoint: {arm}")
        if path.stat().st_size < 10_000_000_000:
            raise ValueError(f"incomplete checkpoint: {path}")
        checkpoints[arm] = dict(path=str(path), sha256=audit.sha(path), bytes=path.stat().st_size,
                                exposure=7072, optimizer_step=1768, schedule_sha256=schedule_sha,
                                contract=contract)
        steps = [r for r in rows if r["kind"] == "optimizer_step"]
        diagnostics = [r for r in rows if r["kind"] == "diagnostic"]
        trajectory[arm] = dict(
            diagnostics=[{k: r[k] for k in ("exposure", "reca_loss", "t2a_loss", "paired_gradient_cosine")}
                         for r in diagnostics],
            first_last_50_loss={
                key: [statistics.mean(values) if values else None
                      for span in (steps[:50], steps[-50:])
                      for values in [[r[key] for r in span if r[key] is not None]]]
                for key in ("reca_loss", "t2a_loss")
            },
        )
    root.mkdir(parents=True)
    protocol.update(settings=audit.SETTINGS, checkpoints=checkpoints, coverage=coverage,
                    reused_base=str(PRIOR), training_trajectory=trajectory,
                    training_deployment=(train / "DEPLOYED_COMMIT").read_text().strip(),
                    deployed_commit=Path("DEPLOYED_COMMIT").read_text().strip())
    for setting in SETTINGS:
        previous = PRIOR / setting / "base"
        for i in range(89):
            record = previous / f"{i:03d}.json"
            if not record.is_file() or not Path(json.loads(record.read_text())["wav"]).is_file():
                raise ValueError(f"missing previous Base sample: {record}")
        folder = root / setting
        folder.mkdir()
        (folder / "base").symlink_to((PRIOR / setting / "base").resolve(), target_is_directory=True)
    (root / "reca_ref").symlink_to((PRIOR / "reca_ref").resolve(), target_is_directory=True)
    audit.write_json(root / "protocol.json", protocol)
    print(json.dumps(trajectory, indent=2), flush=True)


def low_lr_contrasts(payload, historical):
    comparisons = payload["paired_contrasts"]
    for setting, entries in payload["generation"].items():
        previous = historical["generation"][setting]
        records = entries["base"]["rows"]
        fields = ("sha", "prompt", "seed", "source_sha", "frames", "exploratory")
        identity = [tuple(row[key] for key in fields) for row in records]
        mask = np.array([not row["exploratory"] for row in records])
        kept = [row for row, keep in zip(records, mask) if keep]

        def scores(entry):
            rows = entry["rows"]
            if [tuple(row[key] for key in fields) for row in rows] != identity:
                raise ValueError(f"invalid historical pairing: {setting}")
            return np.array([row["clap"] for row in rows])[mask]

        low = {arm: scores(entries[arm]) for arm in audit.ARMS}
        high = {arm: scores(previous[arm]) for arm in audit.ARMS}
        for arm in audit.ARMS[1:]:
            contrasts = {"low_minus_high_lr": (low[arm], high[arm])}
            if arm != "t2a_7070":
                contrasts["low_lr_minus_low_lr_t2a"] = low[arm], low["t2a_7070"]
                contrasts["lr_gain_minus_t2a_lr_gain"] = low[arm] - high[arm], low["t2a_7070"] - high["t2a_7070"]
            for label, (left, right) in contrasts.items():
                comparisons.setdefault(label, {}).setdefault(setting, {})[arm] = audit.paired(left, right, kept)


def score(root, low_lr=False):
    audit.score(root)
    path = root / "results.json"
    payload = json.loads(path.read_text())
    if low_lr:
        if audit.sha(HIGH_LR_RESULTS) != HIGH_LR_SHA:
            raise ValueError("historical results changed since prepare")
        low_lr_contrasts(payload, json.loads(HIGH_LR_RESULTS.read_text()))
        audit.write_json(path, payload)
        return
    for setting, entries in payload["generation"].items():
        big, small = (entries[arm]["rows"] for arm in ("t2a_7070", "t2a_707"))
        pairs = [(a, b) for a, b in zip(big, small) if not a["exploratory"]]
        if any(a["sha"] != b["sha"] for a, b in pairs):
            raise ValueError(f"invalid paired comparison: {setting}")
        payload["paired_contrasts"].setdefault("t2a_large_minus_small", {})[setting] = audit.paired(
            np.array([a["clap"] for a, _ in pairs]),
            np.array([b["clap"] for _, b in pairs]), [a for a, _ in pairs])
    audit.write_json(path, payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "generate", "score"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--low-lr", action="store_true")
    args = parser.parse_args()
    audit.ARMS = ARMS[:-1] if args.low_lr else ARMS
    audit.SETTINGS = {key: audit.SETTINGS[key] for key in SETTINGS}
    if args.phase == "prepare":
        prepare(args.out, args.low_lr)
    elif args.phase == "generate":
        if not 0 <= args.shard < args.shards:
            parser.error("invalid shard")
        audit.generate(args.out, args.shard, args.shards, skip_base=True)
    else:
        score(args.out, args.low_lr)
