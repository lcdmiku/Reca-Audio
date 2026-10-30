#!/usr/bin/env python
"""MusicCaps — 文本到音乐（T2M）评测。

论文协议：MusicCaps，报告 KL / IS / FAD / FD（PANNs 嵌入），100 步 ODE、CFG=6.0。
流程与 AudioCaps 完全一致，只是 task=T2M。

示例：
    python runs/Benchmark/MusicCaps/evaluate.py gen \
        --prompts runs/Benchmark/MusicCaps/prompts.jsonl \
        --out runs/Benchmark/_runs/musiccaps --num-shards 4 --shard-index 0
    python runs/Benchmark/MusicCaps/evaluate.py score \
        --out runs/Benchmark/_runs/musiccaps --ref /data/musiccaps_audio
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
            "google/MusicCaps",
            "sanchit-gandhi/musiccaps",
        ],
        split=args.split,
        out=args.out_jsonl,
        mapping={"id": None, "prompt": args.prompt_field},
        limit=args.limit,
    )
    return 0


def cmd_gen(args: argparse.Namespace) -> int:
    return generation.run_generation(args, task="T2M")


def cmd_score(args: argparse.Namespace) -> int:
    rows = audio_io.read_jsonl(str(Path(args.out) / "manifest.jsonl"))
    pairs = [(r["out"], r["prompt"]) for r in rows if r.get("out") and not r.get("error")]
    if not pairs:
        raise SystemExit("manifest 里没有可用的生成音频，先跑 gen")
    gen = [p[0] for p in pairs]
    prompts = [p[1] for p in pairs]

    ref_files = [str(p) for p in audio_io.list_audio(args.ref)]
    if not ref_files:
        raise SystemExit(f"--ref 目录里没有音频：{args.ref}")
    print(f"[score] gen={len(gen)} ref={len(ref_files)}")

    result = {"benchmark": "MusicCaps", "n_gen": len(gen), "n_ref": len(ref_files)}
    result["fad_clap"] = metrics.fad_clap(gen, ref_files, args.device)
    if args.vggish:
        result["fad_vggish"] = metrics.fad_vggish(gen, ref_files)
    if args.panns:
        result.update(metrics.panns_scores(gen, ref_files, args.device))
    result["clip"] = metrics.clap_text_audio_score(gen, prompts, args.device)

    out_json = Path(args.out) / "results.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[score] -> {out_json}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--dataset", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--prompt-field", default="caption")
    p.add_argument("--out-jsonl", default="runs/Benchmark/MusicCaps/prompts.jsonl")
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(func=cmd_prepare)

    g = sub.add_parser("gen")
    generation.add_model_args(g)
    generation.add_gen_args(g)
    g.set_defaults(func=cmd_gen)

    s = sub.add_parser("score")
    s.add_argument("--out", required=True)
    s.add_argument("--ref", required=True, help="MusicCaps 真实音频目录")
    s.add_argument("--device", default="cuda:0")
    s.add_argument("--vggish", action="store_true")
    s.add_argument("--panns", action="store_true")
    s.set_defaults(func=cmd_score)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
