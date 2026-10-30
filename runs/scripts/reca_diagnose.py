#!/usr/bin/env python
"""RecA 诊断脚本：用音频自身的理解特征重建同一段音频。

这是 RecA 的 E1 实验（对应原论文 Fig.5）——先确认"失配真实存在"：
模型能从自己的理解特征里恢复内容，但声学细节（音色/空间/质感）会崩。

用法（服务器，仓库根目录）：
    python runs/scripts/reca_diagnose.py \
        --config Audio-Omni/model/Audio-Omni.json \
        --ckpt   Audio-Omni/model/model.ckpt \
        --input  "$DATA_ROOT/processed/fsd50k" \
        --out    "$RUN_ROOT/samples/reca_diag" \
        --limit 100

输入可以是：目录（递归找 wav/flac/mp3）、单个文件、或 jsonl（每行含 "path"）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg")


def bootstrap() -> None:
    """允许未 pip install 时直接从仓库根目录运行。"""
    root = Path(__file__).resolve().parents[2]
    pkg = root / "Audio-Omni"
    if pkg.is_dir() and str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))


def iter_inputs(src: str) -> list[Path]:
    p = Path(src)
    if p.is_dir():
        files: list[Path] = []
        for ext in AUDIO_EXTS:
            files.extend(sorted(p.rglob(f"*{ext}")))
        return files
    if p.suffix == ".jsonl":
        out = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out.append(Path(rec.get("path") or rec["src"]))
        return out
    return [p]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="Audio-Omni/model/Audio-Omni.json")
    ap.add_argument("--ckpt", default="Audio-Omni/model/model.ckpt")
    ap.add_argument("--input", required=True, help="目录 / 文件 / jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--seconds", type=int, default=10)
    ap.add_argument("--template", default="", help="留空用默认 RECONSTRUCT_USER")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1234, help="固定种子，便于 A/B")
    args = ap.parse_args()

    bootstrap()
    import soundfile as sf
    import torch

    from audio_omni.api import AudioOmni

    files = iter_inputs(args.input)
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"[fatal] 没有找到音频：{args.input}", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] {len(files)} 条 -> {out_dir}")
    print(f"[info] dtype = {torch.get_autocast_dtype('cuda') if torch.cuda.is_available() else 'cpu'}")

    model = AudioOmni(args.config, args.ckpt, device=args.device)
    sr = model.sample_rate

    manifest = (out_dir / "manifest.jsonl").open("w", encoding="utf-8")
    t_start = time.time()
    for i, path in enumerate(files, 1):
        t0 = time.time()
        rec = {"index": i, "src": str(path)}
        try:
            audio = model.reconstruct(
                str(path),
                template=args.template,
                steps=args.steps,
                cfg_scale=args.cfg,
                seconds_total=args.seconds,
                seed=args.seed,
            )
            name = f"{i:05d}_{path.stem}.wav"
            dst = out_dir / name
            sf.write(str(dst), audio.numpy().T, sr, subtype="PCM_16")
            rec.update({"out": str(dst), "sec": round(time.time() - t0, 2)})
            print(f"[{i}/{len(files)}] {name}  {rec['sec']}s", flush=True)
        except Exception as e:  # noqa: BLE001
            rec.update({"error": f"{type(e).__name__}: {e}"})
            print(f"[{i}/{len(files)}] FAIL {path}: {e}", file=sys.stderr, flush=True)
        manifest.write(json.dumps(rec, ensure_ascii=False) + "\n")
        manifest.flush()
        torch.cuda.empty_cache()

    manifest.close()
    print(f"[done] {(time.time() - t_start) / 60:.1f} min, 输出 {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
