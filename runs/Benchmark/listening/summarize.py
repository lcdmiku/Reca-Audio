"""Summarize completed blind listening sheets with their separately stored keys."""

import argparse
import itertools
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ARTIFACT_TAGS = {"失真 / 爆音", "断裂 / 跳变", "异常重复", "噪声 / 模糊", "不自然的音色"}


def load(path):
    with path.open(encoding="utf-8-sig") as handle:
        record = json.load(handle)
    if not isinstance(record, dict) or type(record.get("version")) is not int or record["version"] != 1:
        raise ValueError(f"{path}: unsupported schema")
    return record


def string(record, field):
    value = record[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing or empty {field}")
    return value.strip()


def indexed(rows):
    result = {}
    if not isinstance(rows, list) or not rows:
        raise ValueError("expected a nonempty list of assignments")
    for row in rows:
        ident = string(row, "id")
        if ident in result:
            raise ValueError(f"duplicate assignment: {ident}")
        result[ident] = row
    return result


def validate_keys(paths):
    forms, arms = {}, set()
    for path in paths:
        form = load(path)
        pack_id = string(form, "pack_id")
        if pack_id in forms:
            raise ValueError(f"duplicate key for {pack_id}")
        blocks = {}
        for ident, block in indexed(form["blocks"]).items():
            clips = {clip_id: string(clip, "arm") for clip_id, clip in indexed(block["clips"]).items()}
            if len(clips) != 4 or len(set(clips.values())) != 4:
                raise ValueError(f"{ident}: key must have four distinct arms")
            if arms and set(clips.values()) != arms:
                raise ValueError(f"{ident}: arms differ between blocks or forms")
            arms = set(clips.values())
            blocks[ident] = (string(block, "domain"), string(block, "sha"), clips)
        forms[pack_id] = blocks
    return forms, arms


def interval(draws):
    draws.sort()
    def quantile(fraction):
        pos = fraction * (len(draws) - 1)
        left = int(pos)
        return draws[left] + (pos - left) * (draws[min(left + 1, len(draws) - 1)] - draws[left])
    return [quantile(.025), quantile(.975)]


def estimate(values):
    ci = None
    if len(values) >= 2:
        rng = random.Random(42)
        ci = interval([sum(rng.choices(values, k=len(values))) / len(values) for _ in range(10000)])
    return {"mean": statistics.mean(values), "sha_ci95": ci}


def summarize(keys, files, *, alignment=False):
    forms, arms = validate_keys(keys)
    dimension = "alignment" if alignment else "quality"
    submissions = set()
    scores = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    tags = defaultdict(Counter)
    for path in files:
        rating = load(path)
        if rating.get("dimension", "quality") != dimension:
            raise ValueError(f"{path}: expected a {dimension} submission")
        pack = string(rating, "pack_id")
        rater = string(rating, "rater_id").casefold()
        if re.fullmatch(r"[a-z0-9_-]{3,32}", rater, flags=re.ASCII) is None:
            raise ValueError(f"{path}: invalid pseudonymous rater ID")
        if pack not in forms or (pack, rater) in submissions:
            raise ValueError(f"{path}: unknown pack or duplicate rater/pack")
        assignments = indexed(rating["blocks"])
        if assignments.keys() != forms[pack].keys():
            raise ValueError(f"{path}: missing or unknown blocks")
        for ident, (domain, sha, expected) in forms[pack].items():
            given = indexed(assignments[ident]["clips"])
            if given.keys() != expected.keys():
                raise ValueError(f"{path}: missing or unknown clips in {ident}")
            for clip_id, arm in expected.items():
                clip = given[clip_id]
                if ("quality" if alignment else "alignment") in clip:
                    raise ValueError(f"{path}: mixed rating dimensions in {ident}/{clip_id}")
                score = clip[dimension]
                if type(score) is not int or not 1 <= score <= 5 or clip.get("heard") is not True:
                    raise ValueError(f"{path}: incomplete score or playback for {ident}/{clip_id}")
                selected = clip.get("tags", []) if alignment else clip["tags"]
                if (not isinstance(selected, list) or (alignment and selected)
                        or not all(isinstance(tag, str) and tag in ARTIFACT_TAGS for tag in selected)
                        or len(selected) != len(set(selected))):
                    raise ValueError(f"{path}: invalid artifact tags for {ident}/{clip_id}")
                scores[domain][sha][arm][rater].append(score)
                tags[domain, arm].update(selected)
        submissions.add((pack, rater))
    labels = {"dimension": dimension} if alignment else {}
    if not submissions:
        return {**labels, "status": "awaiting_ratings", "submissions": 0, "raters": 0, "domains": {}}
    report = {
        **labels,
        "status": "scored", "submissions": len(submissions),
        "raters": len({rater for _, rater in submissions}),
        "method": "Average repeats within listener/SHA/arm, then listeners within SHA, then weight each SHA equally. "
                  "Marginal 95% percentile intervals resample SHA (10000 draws, seed 42), conditional on listeners.",
        "domains": {},
    }
    order = sorted(arms, key=lambda arm: (0 if arm.lower() == "base" else 1 if arm.lower().startswith("t2a") else 2, arm))
    for domain, items in sorted(scores.items()):
        by_arm, per_arm = {}, {}
        domain_raters = set()
        for arm in order:
            groups = [items[sha][arm] for sha in sorted(items)]
            by_arm[arm] = [statistics.mean(statistics.mean(repeats) for repeats in group.values()) for group in groups]
            domain_raters.update(rater for group in groups for rater in group)
            per_arm[arm] = dict(estimate(by_arm[arm]),
                                raw_ratings=sum(len(repeats) for group in groups for repeats in group.values()))
            if not alignment:
                per_arm[arm]["artifact_tag_counts"] = dict(tags[domain, arm])
        comparisons = {}
        for right, left in itertools.combinations(order, 2):
            deltas = [lhs - rhs for lhs, rhs in zip(by_arm[left], by_arm[right])]
            comparisons[f"{left}_minus_{right}"] = estimate(deltas)
        report["domains"][domain] = {"audio_sha": len(items), "raters": len(domain_raters),
                                      "arms": per_arm, f"paired_{dimension}_differences": comparisons}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", type=Path, action="append", required=True)
    parser.add_argument("ratings", type=Path, nargs="*")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--alignment", action="store_true", help="Read caption alignment ratings instead of quality.")
    args = parser.parse_args()
    try:
        result = summarize(args.key, args.ratings, alignment=args.alignment)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        parser.error(str(exc))
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
