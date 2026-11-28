"""Create a blinded edit rating form or score independently completed forms."""

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf


ARMS = ("base", "pure_7070", "t2a_7070", "mix_7070", "mix_pool4_7070", "text_slots_7070")
FIELDS = ("edit", "preserve", "naturalness", "wrong_object")
SEED = 551710


def copy_audio(source, target):
    wave, rate = sf.read(source, frames=441000, dtype="float32")
    if rate != 44100 or len(wave) != 441000:
        raise ValueError(f"Invalid listening audio {source}")
    # A fixed, shared gain keeps relative levels intact and leaves headroom.
    sf.write(target, wave * 0.4, rate, subtype="FLOAT")


def build(formal, destination):
    if not (formal / "GENERATION_COMPLETE").is_file():
        raise ValueError("Require complete generation")
    rows = [json.loads(line) for line in (formal / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    prepare_order = lambda row: hashlib.sha256(f"{SEED}:{row['group']}".encode()).digest()
    native = sorted([row for row in rows if row["kind"] == "add"], key=prepare_order)[:4]
    groups = {row["group"] for row in native}
    native += [row for row in rows if row["kind"] == "remove" and row["group"] in groups]
    music = sorted([row for row in rows if row["kind"] == "style"], key=prepare_order)[:4]
    picked = native + music
    public = destination / "form"
    wav_dir = public / "audio"
    wav_dir.mkdir(parents=True, exist_ok=False)
    rng = random.Random(SEED)
    rng.shuffle(picked)
    order = []
    key = {}
    for number, row in enumerate(picked, 1):
        src = f"source_{number:02d}.wav"
        copy_audio(row["source"], wav_dir / src)
        arms = list(ARMS)
        rng.shuffle(arms)
        for position, arm in enumerate(arms, 1):
            code = f"{number:02d}{chr(64+position)}"
            name = f"edit_{code}.wav"
            copy_audio(formal / "audio" / arm / f"{row['id']}.wav", wav_dir / name)
            order.append(dict(code=code, source=f"audio/{src}", output=f"audio/{name}",
                              instruction=row["instruction"]))
            key[code] = dict(arm=arm, id=row["id"], group=row["group"], kind=row["kind"])
    with (public / "ratings.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["code", "source", "output", "instruction", *FIELDS])
        writer.writeheader()
        writer.writerows(order)
    (destination / "private_key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    (public / "README.md").write_text(
        "# Audio editing listening form\n\n"
        "Open ratings.csv. Listen to source and output while reading the instruction. "
        "Score edit (1 = requested change achieved, 0 = absent or wrong); preserve "
        "(1 = recognizable content outside the requested change remains, 0 = lost); "
        "naturalness (1–5); wrong_object (1 = changed an unintended source, 0 = no). "
        "For Style Transfer judge whether the requested instrument changed and melody/rhythm remain. "
        "Rate edit and preservation independently. Save one completed copy per listener.\n",
        encoding="utf-8")


def summarize(key_path, ratings, out):
    key = json.loads(key_path.read_text(encoding="utf-8"))
    per_case = {code: [] for code in key}
    for path in ratings:
        with path.open(encoding="utf-8-sig") as handle:
            sheet = list(csv.DictReader(handle))
        if len(sheet) != len(key) or {row["code"] for row in sheet} != set(key):
            raise ValueError(f"Missing or duplicate ratings in {path}")
        for row in sheet:
            values = {field: int(row[field]) for field in FIELDS}
            if any(values[field] not in (0, 1) for field in ("edit", "preserve", "wrong_object")):
                raise ValueError(f"Invalid binary label in {path}")
            if not 1 <= values["naturalness"] <= 5:
                raise ValueError(f"Invalid naturalness in {path}")
            e, p = values["edit"], values["preserve"]
            values.update(joint=e*p, edit_only=e*(1-p), preserve_only=p*(1-e), neither=(1-e)*(1-p))
            per_case[row["code"]].append(values)
    result = {}
    for subset, kinds in (("native", {"add", "remove"}), ("style", {"style"})):
        scores = {arm: {} for arm in ARMS}
        for code, value in key.items():
            if value["kind"] in kinds:
                case = per_case[code]
                scores[value["arm"]][value["id"]] = dict(
                    group=value["group"], **{f: float(np.mean([x[f] for x in case])) for f in case[0]})
        comparisons = {}
        for arm in ARMS:
            ref = "text_slots_7070" if arm == "mix_7070" else "base"
            comparisons[arm] = {}
            for field in ("edit", "preserve", "joint", "edit_only", "preserve_only", "neither",
                          "naturalness", "wrong_object"):
                groups = {}
                rows = scores[arm]
                for item, row in rows.items():
                    groups.setdefault(row["group"], []).append(row[field] - scores[ref][item][field])
                values = np.array([np.mean(value) for value in groups.values()])
                draws = np.random.default_rng(SEED).choice(values, (10000, len(values))).mean(axis=1)
                comparisons[arm][field] = dict(mean=float(np.mean([row[field] for row in rows.values()])),
                                               reference=ref, delta=float(values.mean()),
                                               ci95=np.quantile(draws, [.025, .975]).tolist())
        result[subset] = comparisons
    out.write_text(json.dumps(dict(listeners=len(ratings), results=result), indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pack = sub.add_parser("build")
    pack.add_argument("--formal", type=Path, required=True)
    pack.add_argument("--out", type=Path, required=True)
    report = sub.add_parser("score")
    report.add_argument("--key", type=Path, required=True)
    report.add_argument("--out", type=Path, required=True)
    report.add_argument("ratings", nargs="+", type=Path)
    args = parser.parse_args()
    if args.command == "build":
        build(args.formal, args.out)
    else:
        if len(set(path.resolve() for path in args.ratings)) != len(args.ratings):
            parser.error("Each listener form should be provided once")
        summarize(args.key, args.ratings, args.out)
