#!/usr/bin/env python
"""MMSU — 大规模口语理解与推理评测（47 项任务）。

论文协议：报告 MMSU 准确率，作为理解能力的指标。

流程：
    prepare  从 HF 导出统一 jsonl（question/answer/choices/audio）
    eval     逐条调用模型理解并判分（支持 --num-shards 多卡并行）
    merge    合并分片结果，输出总准确率

示例：
    python runs/Benchmark/MMSU/evaluate.py prepare --out-jsonl runs/Benchmark/MMSU/items.jsonl
    python runs/Benchmark/MMSU/evaluate.py eval --items runs/Benchmark/MMSU/items.jsonl \
        --out runs/Benchmark/_runs/mmsu --num-shards 2 --shard-index 0
    python runs/Benchmark/MMSU/evaluate.py merge --out runs/Benchmark/_runs/mmsu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import understanding  # noqa: E402


def cmd_prepare(args: argparse.Namespace) -> int:
    from common import datasets

    datasets.try_export(
        [args.dataset] if args.dataset else ["m-a-p/MMSU", "MMSU-Benchmark/MMSU"],
        split=args.split,
        out=args.out_jsonl,
        mapping={"id": args.id_field, "question": args.question_field,
                 "answer": args.answer_field, "choices": args.choices_field},
        audio_dir=args.audio_dir,
        audio_field=args.audio_field,
        limit=args.limit,
        config=args.config,
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--dataset", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--id-field", default="id")
    p.add_argument("--question-field", default="question")
    p.add_argument("--answer-field", default="answer")
    p.add_argument("--choices-field", default="choices")
    p.add_argument("--audio-field", default="audio")
    p.add_argument("--audio-dir", default="runs/Benchmark/_data/mmsu_audio")
    p.add_argument("--out-jsonl", default="runs/Benchmark/MMSU/items.jsonl")
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(func=cmd_prepare)

    e = sub.add_parser("eval")
    understanding.add_args(e)
    e.set_defaults(func=lambda a: understanding.run(a, "MMSU"))

    m = sub.add_parser("merge")
    m.add_argument("--out", required=True)
    m.set_defaults(func=lambda a: understanding.merge_and_report(a.out, "MMSU"))

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
