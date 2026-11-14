"""Evaluate the AudioCaps scale matrix on the preceding audit's paired inputs."""

import argparse
import json
from pathlib import Path
import statistics

import numpy as np

import diagnose_posttrain as audit

TRAIN = Path("runs/train/audiocaps_scale_matrix_v2")
PRIOR = Path("runs/audits/posttrain_four_arm_v1")
ARMS = ("base", "pure_7070", "t2a_7070", "mix_7070", "t2a_707")
SETTINGS = ("audio_10_cfg6", "audio_10_cfg1", "music_10_cfg6", "reca_matched", "reca_mismatched")


def prepare(root):
    if not (TRAIN / "COMPLETE").is_file() or not (PRIOR / "COMPLETE").is_file():
        raise ValueError("training and prior evaluation must be complete")
    protocol = audit.load_protocol(PRIOR)
    if protocol["smoke"] or protocol["steps"] != 100 or len(protocol["audio"]) != 89:
        raise ValueError("unexpected reference protocol")
    coverage = json.loads((TRAIN / "coverage.json").read_text())
    checkpoints = {"base": protocol["checkpoints"]["base"]}
    trajectory = {}
    for arm in ARMS[1:]:
        path = TRAIN / arm / "dit_exposure7072.pt"
        schedule = TRAIN / f"{arm}_schedule.jsonl"
        schedule_sha = audit.sha(schedule)
        if schedule_sha != coverage[arm]["schedule_sha256"]:
            raise ValueError(f"schedule digest mismatch: {arm}")
        rows = audit.audio_io.read_jsonl(TRAIN / arm / "metrics.jsonl")
        if not any(r["kind"] == "checkpoint" and r["exposure"] == 7072
                   and r["optimizer_step"] == 1768 and r["reload_validated"] for r in rows):
            raise ValueError(f"unverified final checkpoint: {arm}")
        if path.stat().st_size < 10_000_000_000:
            raise ValueError(f"incomplete checkpoint: {path}")
        checkpoints[arm] = dict(path=str(path), sha256=audit.sha(path), bytes=path.stat().st_size,
                                exposure=7072, optimizer_step=1768, schedule_sha256=schedule_sha,
                                contract=json.loads((TRAIN / arm / "contract.json").read_text()))
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
                    training_deployment=(TRAIN / "DEPLOYED_COMMIT").read_text().strip(),
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


def score(root):
    audit.score(root)
    path = root / "results.json"
    payload = json.loads(path.read_text())
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
    args = parser.parse_args()
    audit.ARMS = ARMS
    audit.SETTINGS = {key: audit.SETTINGS[key] for key in SETTINGS}
    if args.phase == "prepare":
        prepare(args.out)
    elif args.phase == "generate":
        if not 0 <= args.shard < args.shards:
            parser.error("invalid shard")
        audit.generate(args.out, args.shard, args.shards, skip_base=True)
    else:
        score(args.out)
