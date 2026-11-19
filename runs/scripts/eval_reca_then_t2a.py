"""Evaluate the sequential endpoint on the frozen low-LR paired protocol."""

import argparse
import json
from pathlib import Path

import numpy as np

import diagnose_posttrain as audit
from eval_audiocaps_scale import SETTINGS


TRAIN = Path("runs/train/audiocaps_reca_then_t2a_v1")
PRIOR = Path("runs/audits/audiocaps_low_lr_eval_v1")
PRIOR_SHA = "852c5c459ec89f8b0a4e8ead80fa53fa8903e227b84c62e0d883362c24369b20"
SCHEDULE_SHA = "dbc75946263411da77466edc9b31b1c3d945df864265588036d5d83344285e94"
ARM = "sequential_7070"
CONTROLS = ("base", "mix_7070", "pure_7070", "t2a_7070")


def prior_results():
    path = PRIOR / "results.json"
    if not (PRIOR / "COMPLETE").is_file() or audit.sha(path) != PRIOR_SHA:
        raise ValueError("Invalid historical low-LR reference")
    return json.loads(path.read_text())


def prepare(root):
    if root.exists() or not (TRAIN / "COMPLETE").is_file():
        raise ValueError("Require completed training and a new evaluation directory")
    protocol = prior_results()["protocol"]
    if (protocol["steps"] != 100 or protocol["smoke"] or protocol["dtype"] != "float16"
            or protocol["settings"] != {key: list(audit.SETTINGS[key]) for key in SETTINGS}):
        raise ValueError("Low-LR generation settings mismatch")
    checkpoint = TRAIN / ARM / "dit_exposure7072.pt"
    schedule = TRAIN / "schedule.jsonl"
    coverage = json.loads((TRAIN / "schedule.jsonl.audit.json").read_text())
    mix = protocol["checkpoints"]["mix_7070"]
    contract = json.loads((TRAIN / ARM / "contract.json").read_text())
    if (audit.sha(schedule) != SCHEDULE_SHA or coverage["schedule_sha256"] != SCHEDULE_SHA
            or coverage["source_sha256"] != mix["schedule_sha256"]
            or contract != dict(mix["contract"], schedule=str(schedule), task_mode="reca_then_t2a")
            or (TRAIN / "base_inputs.sha256").read_text().split()[0] != protocol["checkpoints"]["base"]["sha256"]):
        raise ValueError("Sequential training provenance mismatch")
    rows = audit.audio_io.read_jsonl(TRAIN / ARM / "metrics.jsonl")
    if not any(row["kind"] == "checkpoint" and row["reload_validated"] and row["optimizer_step"] == 1768
               and row["exposure"] == 7072 for row in rows) or checkpoint.stat().st_size < 10_000_000_000:
        raise ValueError("Missing verified final checkpoint")
    protocol["checkpoints"][ARM] = dict(path=str(checkpoint), bytes=checkpoint.stat().st_size,
                                        sha256=audit.sha(checkpoint), schedule_sha256=SCHEDULE_SHA,
                                        exposure=7072, optimizer_step=1768, contract=contract)
    protocol["coverage"][ARM] = coverage
    protocol["training_trajectory"][ARM] = dict(diagnostics=[
        {key: row[key] for key in ("exposure", "reca_loss", "t2a_loss", "paired_gradient_cosine")}
        for row in rows if row["kind"] == "diagnostic"])
    protocol["low_lr_reference"] = dict(results=str(PRIOR / "results.json"), sha256=PRIOR_SHA,
                                        git_commit=protocol["git_commit"], training_deployment=protocol["training_deployment"])
    protocol["training_deployment"] = (TRAIN / "DEPLOYED_COMMIT").read_text().strip()
    protocol["deployed_commit"] = Path("DEPLOYED_COMMIT").read_text().strip()
    protocol["git_commit"] = dict(line.split("=", 1) for line in protocol["deployed_commit"].splitlines())["commit"]
    root.mkdir(parents=True)
    for setting in SETTINGS:
        folder = root / setting
        folder.mkdir()
        (folder / "base").symlink_to((PRIOR / setting / "base").resolve(strict=True), target_is_directory=True)
    (root / "reca_ref").symlink_to((PRIOR / "reca_ref").resolve(strict=True), target_is_directory=True)
    audit.write_json(root / "protocol.json", protocol)
    print(f"Prepared {ARM}: {protocol['checkpoints'][ARM]['sha256']}", flush=True)


def add_controls(payload, prior):
    contrasts = payload["paired_contrasts"]
    identity = ("sha", "prompt", "seed", "frames", "source_sha", "exploratory")
    for setting, entries in payload["generation"].items():
        rows = entries[ARM]["rows"]
        def identities(items):
            return [tuple(row[field] for field in identity) for row in items]
        main = [idx for idx, row in enumerate(rows) if not row["exploratory"]]
        records = [rows[idx] for idx in main]
        if len(rows) != 89 or len(main) != (89 if setting == "music_10_cfg6" else 81):
            raise ValueError(f"Invalid number of paired items: {setting}")
        seq_values = np.array([row["clap"] for row in records])
        for arm in CONTROLS:
            ref = prior["generation"][setting][arm]
            if identities(ref["rows"]) != identities(rows):
                raise ValueError(f"Invalid paired identities: {arm}/{setting}")
            if arm == "base":
                if not np.allclose([row["clap"] for row in entries["base"]["rows"]],
                                   [row["clap"] for row in ref["rows"]], rtol=0, atol=1e-6):
                    raise ValueError(f"Base score differs from reference: {setting}")
            else:
                entries[arm] = ref
            ref_values = np.array([ref["rows"][idx]["clap"] for idx in main])
            contrasts.setdefault("sequential_minus_control", {}).setdefault(setting, {})[arm] = audit.paired(
                seq_values, ref_values, records)
    gap = "reca_matched_minus_mismatched"
    for arm in CONTROLS:
        contrasts[gap][arm] = prior["paired_contrasts"][gap][arm]
    matched = payload["generation"]["reca_matched"]
    mismatched = payload["generation"]["reca_mismatched"]
    primary = [idx for idx, row in enumerate(matched[ARM]["rows"]) if not row["exploratory"]]
    records = [matched[ARM]["rows"][idx] for idx in primary]
    def advantages(arm):
        return np.array([matched[arm]["rows"][idx]["clap"] - mismatched[arm]["rows"][idx]["clap"]
                         for idx in primary])
    contrasts["sequential_minus_control_matching_advantage"] = {
        arm: audit.paired(advantages(ARM), advantages(arm), records) for arm in CONTROLS}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "generate", "score"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    audit.ARMS = ("base", ARM)
    audit.SETTINGS = {key: audit.SETTINGS[key] for key in SETTINGS}
    if args.phase == "prepare":
        prepare(args.out)
    else:
        protocol = audit.load_protocol(args.out)
        if ((args.out / "COMPLETE").exists() or protocol.get("low_lr_reference", {}).get("sha256") != PRIOR_SHA
                or protocol.get("checkpoints", {}).get(ARM, {}).get("schedule_sha256") != SCHEDULE_SHA):
            raise ValueError("Refusing to modify an unrelated or completed evaluation")
        if args.phase == "generate":
            if not 0 <= args.shard < args.shards:
                parser.error("invalid shard")
            audit.generate(args.out, args.shard, args.shards, skip_base=True)
        else:
            previous = prior_results()
            audit.score(args.out)
            path = args.out / "results.json"
            results = json.loads(path.read_text())
            add_controls(results, previous)
            audit.write_json(path, results)
