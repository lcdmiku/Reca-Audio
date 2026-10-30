#!/usr/bin/env python
"""AudioEdit — 指令式音频编辑评测（add / remove / extract / style transfer）。

论文协议：在 AudioEdit test set 上报告四类任务平均的 AE_FAD↓、AE_LSD↓，
以及 add/extract/style 上的 CLAP↑（指令遵循）。

统一 jsonl 字段：
    {"id","task","source_audio","desc"?,"source_category"?,"target_category"?,"target_audio"?}
    task ∈ {Add, Remove, Extract, Style Transfer}

示例：
    python runs/Benchmark/AudioEdit/evaluate.py gen \
        --items runs/Benchmark/AudioEdit/items.jsonl \
        --out runs/Benchmark/_runs/audioedit --num-shards 4 --shard-index 0
    python runs/Benchmark/AudioEdit/evaluate.py score --out runs/Benchmark/_runs/audioedit
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import audio_io, generation, metrics  # noqa: E402


def _load_items(path: str, limit: int, shard_index: int, num_shards: int):
    rows = audio_io.read_jsonl(path)
    if limit:
        rows = rows[:limit]
    return rows[shard_index::num_shards]


def cmd_prepare(args: argparse.Namespace) -> int:
    from common import datasets

    datasets.try_export(
        [args.dataset] if args.dataset else ["HKUSTAudio/AudioEdit"],
        split=args.split,
        out=args.out_jsonl,
        mapping={"id": args.id_field, "task": args.task_field,
                 "desc": args.desc_field, "source_category": args.source_category_field,
                 "target_category": args.target_category_field},
        audio_dir=args.audio_dir,
        audio_field=args.source_audio_field,
        limit=args.limit,
        config=args.config,
    )
    return 0


def cmd_gen(args: argparse.Namespace) -> int:
    items = _load_items(args.items, args.limit, args.shard_index, args.num_shards)
    if not items:
        print("[skip] 本分片没有任务")
        return 0

    out_dir = Path(args.out)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    tag = f"shard{args.shard_index}of{args.num_shards}"

    model = generation.build_model(args)
    manifest = (out_dir / f"manifest.{tag}.jsonl").open("w", encoding="utf-8")
    print(f"[edit] {len(items)} 条 | {tag} steps={args.steps} cfg={args.cfg}", flush=True)
    t0 = time.time()
    for i, item in enumerate(items, 1):
        rec = {"id": item.get("id", f"{i:05d}"), "task": item["task"],
               "target_audio": item.get("target_audio"), "text": item.get("desc") or item.get("target_category")}
        t_start = time.time()
        try:
            kwargs = {}
            if item.get("desc"):
                kwargs["desc"] = item["desc"]
            if item.get("source_category"):
                kwargs["source_category"] = item["source_category"]
            if item.get("target_category"):
                kwargs["target_category"] = item["target_category"]
            audio = model.edit(item["task"], item["source_audio"], steps=args.steps,
                               cfg_scale=args.cfg, seed=args.seed + i, **kwargs)
            name = f"{rec['id']}.wav"
            audio_io.save_audio(str(audio_dir / name), audio, model.sample_rate)
            rec["out"] = str(audio_dir / name)
            print(f"[{i}/{len(items)}] {time.time() - t_start:.1f}s {rec['task']}", flush=True)
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
            print(f"[{i}/{len(items)}] FAIL: {e}", file=sys.stderr, flush=True)
        manifest.write(json.dumps(rec, ensure_ascii=False) + "\n")
        manifest.flush()
    manifest.close()
    print(f"[done] {tag} {(time.time() - t0) / 60:.1f} min")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    rows = audio_io.read_jsonl(str(Path(args.out) / "manifest.jsonl"))
    rows = [r for r in rows if r.get("out") and not r.get("error")]
    if not rows:
        raise SystemExit("manifest 里没有可用的生成音频，先跑 gen")

    gen = [r["out"] for r in rows]
    per_task = {}
    for task in sorted({r["task"] for r in rows}):
        sub = [r for r in rows if r["task"] == task]
        gen_t = [r["out"] for r in sub]
        ref_t = [r["target_audio"] for r in sub if r.get("target_audio")]
        entry = {"n": len(sub)}
        if ref_t and len(ref_t) == len(gen_t):
            entry["fad_clap"] = metrics.fad_clap(gen_t, ref_t, args.device)
            entry["lsd"] = metrics.lsd(gen_t, ref_t)
        texts = [r["text"] for r in sub if r.get("text")]
        if texts and len(texts) == len(gen_t):
            entry["clap"] = metrics.clap_text_audio_score(gen_t, texts, args.device)
        per_task[task] = entry

    result = {"benchmark": "AudioEdit", "n": len(rows), "per_task": per_task}
    # 论文口径：四任务平均
    for key in ("fad_clap", "lsd", "clap"):
        vals = [v[key] for v in per_task.values() if key in v]
        if vals:
            result[f"avg_{key}"] = sum(vals) / len(vals)

    out_json = Path(args.out) / "results.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--dataset", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--id-field", default="id")
    p.add_argument("--task-field", default="task")
    p.add_argument("--desc-field", default="desc")
    p.add_argument("--source-category-field", default="source_category")
    p.add_argument("--target-category-field", default="target_category")
    p.add_argument("--source-audio-field", default="source_audio")
    p.add_argument("--audio-dir", default="runs/Benchmark/_data/audioedit")
    p.add_argument("--out-jsonl", default="runs/Benchmark/AudioEdit/items.jsonl")
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(func=cmd_prepare)

    g = sub.add_parser("gen")
    generation.add_model_args(g)
    g.add_argument("--items", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--limit", type=int, default=0)
    g.add_argument("--steps", type=int, default=100)
    g.add_argument("--cfg", type=float, default=6.0)
    g.add_argument("--seed", type=int, default=1234)
    g.add_argument("--shard-index", type=int, default=0)
    g.add_argument("--num-shards", type=int, default=1)
    g.set_defaults(func=cmd_gen)

    m = sub.add_parser("merge")
    m.add_argument("--out", required=True)
    m.set_defaults(func=lambda a: generation.merge_manifests(a.out) and 0)

    s = sub.add_parser("score")
    s.add_argument("--out", required=True)
    s.add_argument("--device", default="cuda:0")
    s.set_defaults(func=cmd_score)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
