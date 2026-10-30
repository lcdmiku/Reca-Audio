#!/usr/bin/env python
"""TTS — Seed-TTS en 评测（论文 Table 3 的 TTS_WER）。

协议：给定文本（可选参考音色），生成语音后用 Whisper 转写，与参考文本算 WER。
论文报告 Seed-TTS en 的 WER = 1.77。

统一 jsonl：
    {"id": "00001", "text": "Hello, welcome to Audio-Omni.", "ref_audio": "...", "ref_text": "..."}

示例：
    python runs/Benchmark/TTS/evaluate.py gen \
        --items runs/Benchmark/TTS/items.jsonl --out runs/Benchmark/_runs/tts
    python runs/Benchmark/TTS/evaluate.py score --out runs/Benchmark/_runs/tts
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import audio_io, generation, metrics  # noqa: E402


def cmd_gen(args: argparse.Namespace) -> int:
    rows = audio_io.read_jsonl(args.items)
    if args.limit:
        rows = rows[: args.limit]
    rows = rows[args.shard_index :: args.num_shards]
    if not rows:
        print("[skip] 本分片没有任务")
        return 0

    out_dir = Path(args.out)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    tag = f"shard{args.shard_index}of{args.num_shards}"

    model = generation.build_model(args)
    manifest = (out_dir / f"manifest.{tag}.jsonl").open("w", encoding="utf-8")
    print(f"[tts] {len(rows)} 条 | {tag}", flush=True)
    for i, item in enumerate(rows, 1):
        rec = {"id": item.get("id", f"{i:05d}"), "text": item["text"]}
        try:
            kwargs = {}
            if item.get("ref_audio"):
                kwargs["voice_prompt_path"] = item["ref_audio"]
            if item.get("ref_text"):
                kwargs["voice_ref_text"] = item["ref_text"]
            audio = model.generate("TTS", prompt=item["text"], steps=args.steps,
                                   cfg_scale=args.cfg, seconds_total=args.seconds,
                                   seed=args.seed + i, **kwargs)
            name = f"{rec['id']}.wav"
            audio_io.save_audio(str(audio_dir / name), audio, model.sample_rate)
            rec["out"] = str(audio_dir / name)
            print(f"[{i}/{len(rows)}] ok", flush=True)
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
            print(f"[{i}/{len(rows)}] FAIL: {e}", file=sys.stderr, flush=True)
        manifest.write(json.dumps(rec, ensure_ascii=False) + "\n")
        manifest.flush()
    manifest.close()
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    rows = [r for r in audio_io.read_jsonl(str(Path(args.out) / "manifest.jsonl"))
            if r.get("out") and not r.get("error")]
    if not rows:
        raise SystemExit("manifest 里没有可用的生成音频，先跑 gen")
    score = metrics.wer([r["out"] for r in rows], [r["text"] for r in rows],
                        model_name=args.asr_model, device=args.device)
    result = {"benchmark": "TTS", "n": len(rows), "wer": score, "asr": args.asr_model}
    (Path(args.out) / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen")
    generation.add_model_args(g)
    g.add_argument("--items", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--limit", type=int, default=0)
    g.add_argument("--steps", type=int, default=100)
    g.add_argument("--cfg", type=float, default=6.0)
    g.add_argument("--seconds", type=int, default=10)
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
    s.add_argument("--asr-model", default="openai/whisper-small",
                   help="论文用 Whisper-large-v3 级别；显存/速度受限时可用 small")
    s.set_defaults(func=cmd_score)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
