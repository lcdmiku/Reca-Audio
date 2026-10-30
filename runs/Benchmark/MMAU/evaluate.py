#!/usr/bin/env python
"""MMAU — 多模态音频理解与推理评测（27 项技能，覆盖 sound / music / speech）。

论文协议：报告 MMAU 准确率。流程与 MMSU 相同。

示例：
    python runs/Benchmark/MMAU/evaluate.py prepare --out-jsonl runs/Benchmark/MMAU/items.jsonl
    python runs/Benchmark/MMAU/evaluate.py eval --items runs/Benchmark/MMAU/items.jsonl \
        --out runs/Benchmark/_runs/mmau
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
        [args.dataset] if args.dataset else ["gamma-lab-ucla/MMAU", "MMAU/MMAU"],
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
    p.add_argument("--audio-dir", default="runs/Benchmark/_data/mmau_audio")
    p.add_argument("--out-jsonl", default="runs/Benchmark/MMAU/items.jsonl")
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(func=cmd_prepare)

    e = sub.add_parser("eval")
    understanding.add_args(e)
    e.set_defaults(func=lambda a: understanding.run(a, "MMAU"))

    m = sub.add_parser("merge")
    m.add_argument("--out", required=True)
    m.set_defaults(func=lambda a: understanding.merge_and_report(a.out, "MMAU"))

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
