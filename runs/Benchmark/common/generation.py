"""分片批量生成：所有生成类 benchmark 共用。

用法（由各 benchmark 的 evaluate.py 调用）：
    gen 阶段：按 --num-shards 把 prompt 列表切片，每个进程只跑自己那片，
              输出 wav + manifest.<shard>.jsonl
    merge 阶段：把所有分片 manifest 合并成 manifest.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

from .audio_io import save_audio


def add_model_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--model-config", default="Audio-Omni/model/Audio-Omni.json")
    ap.add_argument("--ckpt", default="Audio-Omni/model/model.ckpt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cond-device", default="cuda:0",
                    help="Qwen 理解塔所在卡；默认与 --device 同卡。用 run_shards.sh 分片时每进程只见一张卡，必须留 cuda:0")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--no-build", action="store_true", help="只写 manifest，不实际生成（调试用）")


def add_gen_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--prompts", required=True, help="jsonl（含 id/prompt）或 txt（每行一条）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--steps", type=int, default=100, help="论文设置：100 步 ODE")
    ap.add_argument("--cfg", type=float, default=6.0, help="论文设置：CFG=6.0")
    ap.add_argument("--seconds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)


def load_prompts(path: str) -> List[dict]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(
            f"[fatal] 提示词文件不存在：{path}\n"
            "  仓库不自带 benchmark 提示词，三种来源：\n"
            "    1) 先用样本打通生成链路：\n"
            "         --prompts runs/Benchmark/AudioCaps/prompts.sample.jsonl\n"
            "         --prompts runs/Benchmark/MusicCaps/prompts.sample.jsonl\n"
            "    2) 从 HF 数据集导出完整测试集：\n"
            "         python runs/Benchmark/AudioCaps/evaluate.py prepare --out-jsonl <path>\n"
            "    3) 自己写 jsonl，每行 {\"id\": \"00001\", \"prompt\": \"...\"}"
        )
    if p.suffix == ".jsonl":
        rows = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows
    rows = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append({"id": f"{i:05d}", "prompt": line})
    return rows


def build_model(args):
    root = Path(__file__).resolve().parents[3]
    pkg = root / "Audio-Omni"
    if pkg.is_dir() and str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))

    from audio_omni.api import AudioOmni

    return AudioOmni(
        args.model_config,
        args.ckpt,
        device=args.device,
        conditioner_device=args.cond_device,
        dtype=args.dtype,
    )


def run_generation(args, task: str = "T2A") -> int:
    prompts = load_prompts(args.prompts)
    if args.limit:
        prompts = prompts[: args.limit]
    prompts = prompts[args.shard_index :: args.num_shards]
    if not prompts:
        print(f"[skip] shard {args.shard_index}/{args.num_shards} 无任务")
        return 0

    out_dir = Path(args.out)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    tag = f"shard{args.shard_index}of{args.num_shards}"

    print(f"[gen] task={task} {tag} n={len(prompts)} steps={args.steps} cfg={args.cfg}", flush=True)
    model = build_model(args)

    manifest_path = out_dir / f"manifest.{tag}.jsonl"
    t0 = time.time()
    with manifest_path.open("w", encoding="utf-8") as f:
        for i, item in enumerate(prompts, 1):
            rec = {
                "id": item.get("id", f"{i:05d}"),
                "prompt": item["prompt"],
                "task": task,
                # 让输出自解释：拿到一个 wav 目录就能知道是用什么设置生成的
                "steps": args.steps,
                "cfg": args.cfg,
                "seconds": args.seconds,
                "dtype": args.dtype,
                "ckpt": Path(args.ckpt).name,
            }
            t_start = time.time()
            try:
                if args.no_build:
                    rec["out"] = ""
                else:
                    audio = model.generate(
                        task,
                        prompt=item["prompt"],
                        steps=args.steps,
                        cfg_scale=args.cfg,
                        seconds_total=args.seconds,
                        seed=args.seed + i,
                    )
                    name = f"{rec['id']}.wav"
                    save_audio(str(audio_dir / name), audio, model.sample_rate)
                    rec["out"] = str(audio_dir / name)
                rec["sec"] = round(time.time() - t_start, 2)
                print(f"[{i}/{len(prompts)}] {rec['sec']}s {rec['prompt'][:40]}", flush=True)
            except Exception as e:  # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {e}"
                print(f"[{i}/{len(prompts)}] FAIL {rec['prompt'][:40]}: {e}", file=sys.stderr, flush=True)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

    print(f"[done] {tag} {(time.time() - t0) / 60:.1f} min -> {manifest_path}")
    return 0


def merge_manifests(out_dir: str) -> int:
    root = Path(out_dir)
    rows, seen = [], set()
    for path in sorted(root.glob("manifest.shard*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["id"] in seen:
                continue
            seen.add(rec["id"])
            rows.append(rec)
    rows.sort(key=lambda r: r["id"])
    merged = root / "manifest.jsonl"
    with merged.open("w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[merge] {len(rows)} 条 -> {merged}")
    return len(rows)
