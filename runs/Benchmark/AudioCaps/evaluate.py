#!/usr/bin/env python
"""AudioCaps — 文本到音频（T2A）评测。

论文协议：AudioCaps test，报告 KL / IS / FAD / FD（PANNs 嵌入），100 步 ODE、CFG=6.0。

四个阶段：
    prepare  从 HF 导出 prompts.jsonl（或用 --prompts 自己给）
    gen      批量生成 wav（支持 --num-shards/--shard-index 多卡并行）
    merge    合并分片 manifest
    score    对生成的 wav 算指标（需要 --ref 指向 AudioCaps test 的真实音频目录）

示例：
    python runs/Benchmark/AudioCaps/evaluate.py gen \
        --prompts runs/Benchmark/AudioCaps/prompts.jsonl \
        --out runs/Benchmark/_runs/audiocaps --num-shards 8 --shard-index 0
    python runs/Benchmark/AudioCaps/evaluate.py score \
        --out runs/Benchmark/_runs/audiocaps --ref /data/audiocaps_test_audio
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import audio_io, generation, metrics  # noqa: E402


def cmd_prepare(args: argparse.Namespace) -> int:
    from common import datasets

    datasets.try_export(
        [args.dataset] if args.dataset else [
            "d0rj/audiocaps",
            "sanchit-gandhi/audiocaps",
            "google/audiocaps",
        ],
        split=args.split,
        out=args.out_jsonl,
        mapping={"id": None, "prompt": args.prompt_field},
        limit=args.limit,
    )
    return 0


def cmd_gen(args: argparse.Namespace) -> int:
    return generation.run_generation(args, task="T2A")


def cmd_merge(args: argparse.Namespace) -> int:
    generation.merge_manifests(args.out)
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    rows = audio_io.read_jsonl(str(Path(args.out) / "manifest.jsonl"))
    gen = [r["out"] for r in rows if r.get("out") and not r.get("error")]
    if not gen:
        raise SystemExit("manifest 里没有可用的生成音频，先跑 gen")

    ref_files = audio_io.list_audio(args.ref)
    if not ref_files:
        raise SystemExit(f"--ref 目录里没有音频：{args.ref}")
    print(f"[score] gen={len(gen)} ref={len(ref_files)}")

    result = {"benchmark": "AudioCaps", "n_gen": len(gen), "n_ref": len(ref_files)}
    result["fad_clap"] = metrics.fad_clap(gen, [str(p) for p in ref_files], args.device)
    if args.vggish:
        result["fad_vggish"] = metrics.fad_vggish(gen, [str(p) for p in ref_files])
    if args.panns:
        result.update(metrics.panns_scores(gen, [str(p) for p in ref_files], args.device))
    result["clip"] = metrics.clap_text_audio_score(gen, [r["prompt"] for r in rows if r.get("out")], args.device)

    out_json = Path(args.out) / "results.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[score] -> {out_json}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--dataset", default=None, help="HF dataset id；不填则依次尝试内置候选")
    p.add_argument("--split", default="test")
    p.add_argument("--prompt-field", default="caption")
    p.add_argument("--out-jsonl", default="runs/Benchmark/AudioCaps/prompts.jsonl")
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(func=cmd_prepare)

    g = sub.add_parser("gen")
    generation.add_model_args(g)
    generation.add_gen_args(g)
    g.set_defaults(func=cmd_gen)

    m = sub.add_parser("merge")
    m.add_argument("--out", required=True)
    m.set_defaults(func=cmd_merge)

    s = sub.add_parser("score")
    s.add_argument("--out", required=True)
    s.add_argument("--ref", required=True, help="AudioCaps test 真实音频目录")
    s.add_argument("--device", default="cuda:0")
    s.add_argument("--vggish", action="store_true", help="额外算 VGGish-FAD（需 fadtk）")
    s.add_argument("--panns", action="store_true", help="额外算 KL/IS/FD（需 panns-inference）")
    s.set_defaults(func=cmd_score)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
