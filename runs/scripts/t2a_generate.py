#!/usr/bin/env python
"""批量 T2A / T2M 生成，带分片与显存控制。

两种多卡用法：

1) 吞吐模式（8 进程 × 每卡一个完整模型）
       NUM_SHARDS=8 bash runs/scripts/run_t2a_shards.sh \
           --config Audio-Omni/model/Audio-Omni.json \
           --ckpt   Audio-Omni/model/model.ckpt \
           --prompts prompts.txt --out runs/samples/t2a --steps 100

2) 显存模式（一个进程用两张卡：理解塔与生成器分开）
       CUDA_VISIBLE_DEVICES=0,1 python runs/scripts/t2a_generate.py \
           --config ... --ckpt ... --prompts prompts.txt --out runs/samples/t2a \
           --device cuda:0 --cond-device cuda:1 --steps 100

单卡 OOM 时优先试 `--cond-device cuda:1`（省下 3B Qwen 的 12 GB），
再不行加 `--dtype float16`（DiT 权重再省一半）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def bootstrap() -> None:
    root = Path(__file__).resolve().parents[2]
    pkg = root / "Audio-Omni"
    if pkg.is_dir() and str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))


def load_prompts(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        hint = (
            "\n仓库自带示例：\n"
            "  runs/prompts/t2a_smoke.txt   （音效，配 --task T2A）\n"
            "  runs/prompts/t2m_smoke.txt   （音乐，配 --task T2M）"
        )
        raise SystemExit(f"[fatal] 提示词文件不存在：{path}{hint}")
    if p.suffix == ".jsonl":
        out = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line)["prompt"])
        return out
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="Audio-Omni/model/Audio-Omni.json")
    ap.add_argument("--ckpt", default="Audio-Omni/model/model.ckpt")
    ap.add_argument("--prompts", default="runs/prompts/t2a_smoke.txt",
                    help="txt（每行一条，允许 # 注释）或 jsonl（含 prompt 字段）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", default="T2A", choices=["T2A", "T2M"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--seconds", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cond-device", default=None, help="把 Qwen 理解塔放到另一张卡，如 cuda:1")
    ap.add_argument("--dtype", default=None, choices=[None, "float16", "float32"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    bootstrap()
    import soundfile as sf
    import torch

    from audio_omni.api import AudioOmni

    prompts = load_prompts(args.prompts)
    if args.limit:
        prompts = prompts[: args.limit]
    # 交错分片，保证每片内容分布接近
    prompts = prompts[args.shard_index :: args.num_shards]
    if not prompts:
        print(f"[skip] shard {args.shard_index}/{args.num_shards} 没有任务", file=sys.stderr)
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"shard{args.shard_index}of{args.num_shards}"

    print(f"[info] {tag}: {len(prompts)} 条 | step={args.steps} cfg={args.cfg} dtype={args.dtype}", flush=True)
    model = AudioOmni(
        args.config,
        args.ckpt,
        device=args.device,
        conditioner_device=args.cond_device,
        dtype=args.dtype,
    )

    manifest = (out_dir / f"manifest.{tag}.jsonl").open("w", encoding="utf-8")
    t0 = time.time()
    for i, prompt in enumerate(prompts, 1):
        t_start = time.time()
        rec = {"index": i, "prompt": prompt, "task": args.task}
        try:
            audio = model.generate(
                args.task,
                prompt=prompt,
                steps=args.steps,
                cfg_scale=args.cfg,
                seconds_total=args.seconds,
                seed=args.seed + i,
            )
            name = f"{tag}_{i:04d}.wav"
            sf.write(str(out_dir / name), audio.numpy().T, model.sample_rate, subtype="PCM_16")
            rec.update({"out": str(out_dir / name), "sec": round(time.time() - t_start, 2)})
            print(f"[{i}/{len(prompts)}] {rec['sec']}s  {prompt[:48]}", flush=True)
        except torch.cuda.OutOfMemoryError as e:  # noqa: PERF203
            rec["error"] = f"OOM: {e}"
            print(f"[{i}/{len(prompts)}] OOM，建议 --cond-device cuda:1 或 --dtype float16", file=sys.stderr)
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
            print(f"[{i}/{len(prompts)}] FAIL: {e}", file=sys.stderr)
        manifest.write(json.dumps(rec, ensure_ascii=False) + "\n")
        manifest.flush()

    manifest.close()
    print(f"[done] {tag}: {(time.time() - t0) / 60:.1f} min -> {out_dir}")
    model.memory_report("final")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
