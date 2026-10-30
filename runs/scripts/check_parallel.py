#!/usr/bin/env python
"""验证「组件并行 + 精度」是否真的生效（不启动 Gradio，30 秒出结论）。

用法（在 Audio-Omni/ 目录或仓库根目录均可）：
    python runs/scripts/check_parallel.py \
        --config Audio-Omni/model/Audio-Omni.json \
        --ckpt   Audio-Omni/model/model.ckpt \
        --device cuda:0 --cond-device cuda:1 --dtype float16

判读：
    DiT  device=cuda:0  dtype=torch.float16     ← fp16 生效
    Qwen device=cuda:1  dtype=torch.float32     ← 组件并发生效
    cuda:0 ≈ 8 GB, cuda:1 ≈ 12 GB
如果两行 device 都是 cuda:0，说明参数没传进来或 CUDA_VISIBLE_DEVICES 不含第二张卡。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def bootstrap() -> None:
    root = Path(__file__).resolve().parents[2]
    pkg = root / "Audio-Omni"
    if pkg.is_dir() and str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))


def first_param(module):
    try:
        p = next(module.parameters())
        return p.device, p.dtype
    except StopIteration:
        return None, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="Audio-Omni/model/Audio-Omni.json")
    ap.add_argument("--ckpt", default="Audio-Omni/model/model.ckpt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cond-device", default=None)
    ap.add_argument("--dtype", default=None, choices=[None, "float16", "float32"])
    args = ap.parse_args()

    bootstrap()
    import torch

    from audio_omni.loader import load_model_bundle

    print("=" * 60)
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
    print("torch.cuda.device_count() =", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"  cuda:{i} = {torch.cuda.get_device_name(i)}")
    print("=" * 60)

    with open(args.config) as f:
        config = json.load(f)

    model, _, sr, ss = load_model_bundle(
        config,
        args.ckpt,
        device=args.device,
        conditioner_device=args.cond_device,
        dtype=args.dtype,
    )

    print("-" * 60)
    dit_dev, dit_dtype = first_param(model.model)
    print(f"DiT        device={dit_dev} dtype={dit_dtype}")
    try:
        qwen = model.conditioner.conditioners["omni_prompt"].model
        q_dev, q_dtype = first_param(qwen)
        print(f"Qwen       device={q_dev} dtype={q_dtype}")
    except Exception as e:  # noqa: BLE001
        print(f"Qwen       读取失败: {e}")
    if getattr(model, "pretransform", None) is not None:
        v_dev, v_dtype = first_param(model.pretransform)
        print(f"VAE        device={v_dev} dtype={v_dtype}")
    print(f"记录值      model.conditioner_device={getattr(model, 'conditioner_device', None)}")
    print(f"采样率      {sr}, sample_size={ss}")

    print("-" * 60)
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        print(f"cuda:{i} allocated={alloc:.2f}GB reserved={reserved:.2f}GB")

    ok = True
    if args.dtype == "float16" and dit_dtype != torch.float16:
        print("[FAIL] dtype 未生效：DiT 仍是", dit_dtype)
        ok = False
    if args.cond_device is not None and str(q_dev) != str(torch.device(args.cond_device)):
        print(f"[FAIL] 组件并行未生效：Qwen 在 {q_dev}，期望 {args.cond_device}")
        ok = False
    print("[OK] 配置与预期一致" if ok else "[FAIL] 见上面的提示")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
