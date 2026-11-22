"""Evaluate a single follow-up endpoint on the frozen low-LR paired protocol."""

import argparse
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np

import diagnose_posttrain as audit
from eval_audiocaps_scale import SETTINGS


PRIOR = Path("runs/audits/audiocaps_low_lr_eval_v1")
PRIOR_SHA = "852c5c459ec89f8b0a4e8ead80fa53fa8903e227b84c62e0d883362c24369b20"
CONTROLS = ("base", "mix_7070", "pure_7070", "t2a_7070")


class Endpoint(NamedTuple):
    train: Path
    arm: str
    schedule_sha: str


SEQUENTIAL = Endpoint(Path("runs/train/audiocaps_reca_then_t2a_v1"), "sequential_7070",
                      "dbc75946263411da77466edc9b31b1c3d945df864265588036d5d83344285e94")
TEXT_SLOTS = Endpoint(Path("runs/train/audiocaps_mix_text_slots_v1"), "text_slots_7070",
                      "bd5ca9ffa653c06af7d0c1eefb1e36e2fbf6ed09f8bf1b496afb9da1107126a0")
POOL4 = Endpoint(Path("runs/train/audiocaps_mix_pool4_v1"), "mix_pool4_7070", TEXT_SLOTS.schedule_sha)


def reference(endpoint):
    if endpoint == POOL4:
        return ("caption_budget_reference", Path("runs/audits/audiocaps_mix_text_slots_eval_v1"),
                "d33acc98ac6a7efd7716fbee0fa9247db48c5d6b0389576b2b7395f78862085c")
    return "low_lr_reference", PRIOR, PRIOR_SHA


def prior_results(endpoint=SEQUENTIAL):
    _, directory, digest = reference(endpoint)
    path = directory / "results.json"
    if not (directory / "COMPLETE").is_file() or audit.sha(path) != digest:
        raise ValueError(f"Invalid historical reference: {directory}")
    return json.loads(path.read_text())


def text_slot_coverage(protocol, contract, schedule, rows):
    mix = protocol["checkpoints"]["mix_7070"]
    source = protocol["coverage"]["mix_7070"]
    expected = dict(mix["contract"], schedule=str(schedule), task_mode="mix_text_slots",
                    exposure_unit="historical_mix_schedule_slot", scheduled_slots=7072,
                    active_training_exposures=3536, active_task_counts={"t2a": 3536, "reca": 0},
                    active_microbatches_per_update=2, loss_divisor=4,
                    inactive_slot_action="replay_frozen_prepare_only")
    del expected["equal_exposure_not_equal_updates"]
    if contract != expected or mix["schedule_sha256"] != TEXT_SLOTS.schedule_sha:
        raise ValueError("Caption control must use the historical Mix schedule and text weighting")
    slots = audit.audio_io.read_jsonl(schedule)
    updates = [row for row in rows if row["kind"] == "optimizer_step"]
    ref_path = Path(mix["path"]).with_name("metrics.jsonl")
    reference = [row for row in audit.audio_io.read_jsonl(ref_path) if row["kind"] == "optimizer_step"]
    if len(slots) != 7072 or len(updates) != 1768 or len(reference) != len(updates):
        raise ValueError("Incomplete control or Mix trajectory")
    for step, (row, ref) in enumerate(zip(updates, reference), start=1):
        end = step * 4
        positions = [slot["exposure"] for slot in slots[end - 4:end] if slot["task"] == "t2a"]
        if (row["optimizer_step"] != step or ref["optimizer_step"] != step
                or row["exposure"] != end or row["scheduled_slots"] != end or ref["exposure"] != end
                or len(positions) != 2 or row["active_slot_positions"] != positions
                or row["active_training_exposures"] != 2 * step
                or row["active_task_counts"] != {"reca": 0, "t2a": 2 * step} or row["loss_divisor"] != 4
                or row["lr"] != ref["lr"] or row["reca_loss"] is not None or not np.isfinite(row["t2a_loss"])):
            raise ValueError(f"Caption budget / LR mismatch at update {step}")
    return dict(manifest=source["manifest"], schedule_sha256=TEXT_SLOTS.schedule_sha,
                scheduled_slots=7072, exposures=expected["active_task_counts"],
                unique_audio=len({slot["sample_index"] for slot in slots if slot["task"] == "t2a"}),
                verified_updates=len(updates), lr_reference=str(ref_path))


def prepare(root, endpoint=SEQUENTIAL):
    train, arm, schedule_sha = endpoint
    if root.exists() or not (train / "COMPLETE").is_file():
        raise ValueError("Require completed training and a new evaluation directory")
    reference_key, prior_dir, prior_digest = reference(endpoint)
    protocol = prior_results(endpoint)["protocol"]
    if (protocol["steps"] != 100 or protocol["smoke"] or protocol["dtype"] != "float16"
            or protocol["settings"] != {key: list(audit.SETTINGS[key]) for key in SETTINGS}):
        raise ValueError("Low-LR generation settings mismatch")
    checkpoint = train / arm / "dit_exposure7072.pt"
    schedule = train / "schedule.jsonl"
    mix = protocol["checkpoints"]["mix_7070"]
    contract = json.loads((train / arm / "contract.json").read_text())
    if (audit.sha(schedule) != schedule_sha
            or (train / "base_inputs.sha256").read_text().split()[0] != protocol["checkpoints"]["base"]["sha256"]):
        raise ValueError("Training schedule or Base digest mismatch")
    rows = audit.audio_io.read_jsonl(train / arm / "metrics.jsonl")
    if endpoint == TEXT_SLOTS:
        coverage = text_slot_coverage(protocol, contract, schedule, rows)
    elif endpoint == POOL4:
        coverage = protocol["coverage"]["mix_7070"].copy()
        if (contract != dict(mix["contract"], schedule=str(schedule), reca_audio_pool_width=4)
                or coverage["schedule_sha256"] != schedule_sha or mix["schedule_sha256"] != schedule_sha):
            raise ValueError("Pooled Mix differs from the expected historical training contract")
        steps = [row for row in rows if row["kind"] == "optimizer_step"]
        historical = audit.audio_io.read_jsonl(Path(mix["path"]).with_name("metrics.jsonl"))
        ref_steps = [row for row in historical if row["kind"] == "optimizer_step"]
        if len(steps) != 1768 or len(ref_steps) != len(steps):
            raise ValueError("Incomplete pooled or historical Mix trajectory")
        for i, (step, ref) in enumerate(zip(steps, ref_steps), start=1):
            if (step["optimizer_step"] != i or step["exposure"] != 4 * i or step["lr"] != ref["lr"]
                    or not all(np.isfinite(step[k]) for k in ("t2a_loss", "reca_loss"))):
                raise ValueError(f"Pooled Mix exposure, LR or task loss mismatch at update {i}")
    else:
        coverage = json.loads((train / "schedule.jsonl.audit.json").read_text())
        if (coverage["schedule_sha256"] != schedule_sha or coverage["source_sha256"] != mix["schedule_sha256"]
                or contract != dict(mix["contract"], schedule=str(schedule), task_mode="reca_then_t2a")):
            raise ValueError("Sequential training provenance mismatch")
    if not any(row["kind"] == "checkpoint" and row["reload_validated"] and row["optimizer_step"] == 1768
               and row["exposure"] == 7072 for row in rows) or checkpoint.stat().st_size < 10_000_000_000:
        raise ValueError("Missing verified final checkpoint")
    protocol["checkpoints"][arm] = dict(path=str(checkpoint), bytes=checkpoint.stat().st_size,
                                        sha256=audit.sha(checkpoint), schedule_sha256=schedule_sha,
                                        exposure=7072, optimizer_step=1768, contract=contract)
    protocol["coverage"][arm] = coverage
    protocol["training_trajectory"][arm] = dict(diagnostics=[
        {key: row[key] for key in ("exposure", "reca_loss", "t2a_loss", "paired_gradient_cosine")}
        for row in rows if row["kind"] == "diagnostic"])
    protocol[reference_key] = dict(results=str(prior_dir / "results.json"), sha256=prior_digest,
                                   git_commit=protocol["git_commit"], training_deployment=protocol["training_deployment"])
    protocol["training_deployment"] = (train / "DEPLOYED_COMMIT").read_text().strip()
    protocol["deployed_commit"] = Path("DEPLOYED_COMMIT").read_text().strip()
    protocol["git_commit"] = dict(line.split("=", 1) for line in protocol["deployed_commit"].splitlines())["commit"]
    root.mkdir(parents=True)
    for setting in SETTINGS:
        folder = root / setting
        folder.mkdir()
        (folder / "base").symlink_to((prior_dir / setting / "base").resolve(strict=True), target_is_directory=True)
    (root / "reca_ref").symlink_to((prior_dir / "reca_ref").resolve(strict=True), target_is_directory=True)
    audit.write_json(root / "protocol.json", protocol)
    print(f"Prepared {arm}: {protocol['checkpoints'][arm]['sha256']}", flush=True)


def add_controls(payload, prior, endpoint=SEQUENTIAL):
    reference_first = endpoint == TEXT_SLOTS
    label = "reference_minus_text_slots" if reference_first else "sequential_minus_control"
    controls = CONTROLS
    if endpoint == POOL4:
        label = "pool4_minus_control"
        controls = (*CONTROLS, TEXT_SLOTS.arm)
    def compare(new, reference, records):
        return audit.paired(reference, new, records) if reference_first else audit.paired(new, reference, records)

    contrasts = payload["paired_contrasts"]
    identity = ("sha", "prompt", "seed", "frames", "source_sha", "exploratory")
    for setting, entries in payload["generation"].items():
        rows = entries[endpoint.arm]["rows"]
        def identities(items):
            return [tuple(row[field] for field in identity) for row in items]
        main = [idx for idx, row in enumerate(rows) if not row["exploratory"]]
        records = [rows[idx] for idx in main]
        if len(rows) != 89 or len(main) != (89 if setting == "music_10_cfg6" else 81):
            raise ValueError(f"Invalid number of paired items: {setting}")
        if endpoint == POOL4 and any(row.get("reca_audio_pool_width", 1) != (4 if setting.startswith("reca_") else 1)
                                     for row in rows):
            raise ValueError(f"Incorrect recorded conditioning intervention: {setting}")
        values = np.array([row["clap"] for row in records])
        for arm in controls:
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
            contrasts.setdefault(label, {}).setdefault(setting, {})[arm] = compare(values, ref_values, records)
    gap = "reca_matched_minus_mismatched"
    for arm in controls:
        contrasts[gap][arm] = prior["paired_contrasts"][gap][arm]
    matched = payload["generation"]["reca_matched"]
    mismatched = payload["generation"]["reca_mismatched"]
    primary = [idx for idx, row in enumerate(matched[endpoint.arm]["rows"]) if not row["exploratory"]]
    records = [matched[endpoint.arm]["rows"][idx] for idx in primary]
    def advantages(arm):
        return np.array([matched[arm]["rows"][idx]["clap"] - mismatched[arm]["rows"][idx]["clap"]
                         for idx in primary])
    contrasts[f"{label}_matching_advantage"] = {
        arm: compare(advantages(endpoint.arm), advantages(arm), records) for arm in controls}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "generate", "score"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    alternative = parser.add_mutually_exclusive_group()
    alternative.add_argument("--mix-text-slots", action="store_true")
    alternative.add_argument("--mix-pool4", action="store_true")
    args = parser.parse_args()
    endpoint = TEXT_SLOTS if args.mix_text_slots else SEQUENTIAL
    if args.mix_pool4:
        endpoint = POOL4
    audit.ARMS = ("base", endpoint.arm)
    audit.SETTINGS = {key: audit.SETTINGS[key] for key in SETTINGS}
    if args.phase == "prepare":
        prepare(args.out, endpoint)
    else:
        protocol = audit.load_protocol(args.out)
        reference_key, _, prior_digest = reference(endpoint)
        if ((args.out / "COMPLETE").exists() or protocol.get(reference_key, {}).get("sha256") != prior_digest
                or protocol.get("checkpoints", {}).get(endpoint.arm, {}).get("schedule_sha256") != endpoint.schedule_sha):
            raise ValueError("Refusing to modify an unrelated or completed evaluation")
        if args.phase == "generate":
            if not 0 <= args.shard < args.shards:
                parser.error("invalid shard")
            audit.generate(args.out, args.shard, args.shards, skip_base=True)
        else:
            previous = prior_results(endpoint)
            audit.score(args.out)
            path = args.out / "results.json"
            results = json.loads(path.read_text())
            add_controls(results, previous, endpoint)
            audit.write_json(path, results)
