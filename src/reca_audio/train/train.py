"""ReCA 后训练主循环。

只训练 DiT（生成头）；VAE 与理解塔（Qwen2.5-Omni thinker）全部冻结。
- ReCA 模式：条件 = 同一段音频经理解塔得到的 embedding，目标 = 重建该音频
- T2A 模式：条件 = caption，目标 = 对应音频（用于混合训练，默认权重 0）
- mix   模式：按 lambda_reca / lambda_t2a 逐样本采样（论文 λ_RecA=1, λ_T2A=0）

用法（服务器）：
    bash runs/scripts/train_reca.sh
或
    python -m reca_audio.train.train --manifest ... --steps 2000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from audio_omni.loader import load_model_bundle

from .conditioning import build_reca_metadata, build_t2a_metadata
from .data import AudioManifestDataset, collate_clips
from .flow import rectified_flow_loss
from .scheduling import choose_task_mode

try:  # 允许直接 `python src/reca_audio/train/train.py`
    from . import __name__ as _pkg  # noqa: F401
except Exception:  # pragma: no cover
    pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("ReCA post-training for Audio-Omni")
    p.add_argument("--manifest", required=True, help="jsonl：{audio, caption?}")
    p.add_argument("--out", required=True, help="checkpoint / 日志目录")
    p.add_argument("--model-config", default="Audio-Omni/model/Audio-Omni.json")
    p.add_argument("--ckpt", default="Audio-Omni/model/model.ckpt")

    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--seconds", type=float, default=11.0, help="必须对齐 sample_size=485100（11 s）")
    p.add_argument("--sample-rate", type=int, default=44100)
    p.add_argument("--limit", type=int, default=0, help="只用前 N 条（冒烟用）")
    p.add_argument("--num-workers", type=int, default=2)

    p.add_argument("--task-mode", choices=["reca", "t2a", "mix"], default="reca")
    p.add_argument("--lambda-reca", type=float, default=1.0)
    p.add_argument("--lambda-t2a", type=float, default=0.0)
    p.add_argument("--cfg-dropout", type=float, default=0.0,
                   help="ReCA 的音频条件不参与 CFG，默认不丢弃")

    p.add_argument("--device", default="cuda:0")
    p.add_argument("--conditioner-device", default="cuda:1")
    p.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    p.add_argument("--amp", action="store_true", help="DiT 前向用 fp16 autocast")
    p.add_argument("--optim", choices=["adamw", "adafactor"], default="adafactor",
                   help="adafactor 只存 factored 状态，比 AdamW 省 ~2/3 优化器显存")
    p.add_argument("--vae-device", default="", help="把冻结的 VAE 放到别的设备（如 cpu）以省显存")

    p.add_argument("--omni-downsample", type=int, default=1,
                   help="理解塔输入低通倍率（>1 制造信息瓶颈，对应 ReCA 原文降分辨率）")
    p.add_argument("--omni-seconds", type=float, default=0.0,
                   help="理解塔只看这么长的随机片段（0=整段），进一步压缩条件信息")

    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--resume", default="")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def lr_at(step: int, args: argparse.Namespace) -> float:
    if step < args.warmup:
        return args.lr * (step + 1) / max(1, args.warmup)
    progress = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def move_cond(tensors: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in tensors.items():
        if isinstance(v, (tuple, list)):
            out[k] = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in v)
        elif isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def build_metadata(
    batch: Dict[str, Any], idx: int, mode: str, args, device, prompt_rng=None
):
    clip = type("Clip", (), {
        "audio_omni": batch["audio_omni"][idx],
        "caption": batch["caption"][idx],
    })()
    if mode == "t2a":
        return build_t2a_metadata(clip, device=device, seconds=args.seconds, sample_rate=args.sample_rate)
    return build_reca_metadata(
        clip, device=device, seconds=args.seconds, sample_rate=args.sample_rate,
        omni_downsample=args.omni_downsample, omni_seconds=args.omni_seconds,
        prompt_rng=prompt_rng,
    )


def main() -> int:
    args = parse_args()
    choose_task_mode(args.task_mode, args.lambda_reca, args.lambda_t2a, 0.0)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_generator = torch.Generator()
    data_generator.manual_seed(args.seed + 11)
    task_generator = torch.Generator()
    task_generator.manual_seed(args.seed + 12)
    prompt_rng = random.Random(args.seed + 13)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")

    device = torch.device(args.device)
    cond_device = torch.device(args.conditioner_device)
    flow_generator = torch.Generator(device=device)
    flow_generator.manual_seed(args.seed + 14)

    print(f"[build] DiT/VAE -> {device} | conditioner -> {cond_device} | dtype={args.dtype}", flush=True)
    model, model_config, sample_rate, sample_size = load_model_bundle(
        json.loads(Path(args.model_config).read_text()),
        args.ckpt,
        device=str(device),
        conditioner_device=str(cond_device),
        dtype=args.dtype,
    )
    model.conditioner_device = cond_device
    vae_device = torch.device(args.vae_device) if args.vae_device else device
    if vae_device != device:
        model.pretransform.to(vae_device)
        print(f"[build] VAE -> {vae_device}（编码后再搬回 {device}）", flush=True)

    dit = model.model
    dit.train()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in dit.parameters():
        p.requires_grad_(True)
    n_train = sum(p.numel() for p in dit.parameters() if p.requires_grad)
    print(f"[build] trainable DiT params: {n_train/1e6:.1f}M", flush=True)

    dataset = AudioManifestDataset(
        args.manifest, seconds=args.seconds, sr=args.sample_rate, limit=args.limit
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_clips,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        generator=data_generator,
    )
    print(f"[data] {len(dataset)} clips | batch={args.batch_size} | accum={args.grad_accum}", flush=True)

    params = [p for p in dit.parameters() if p.requires_grad]
    if args.optim == "adafactor":
        from transformers.optimization import Adafactor

        optim = Adafactor(
            params, lr=args.lr, scale_parameter=False, relative_step=False,
            warmup_init=False, weight_decay=args.weight_decay,
        )
    else:
        optim = torch.optim.AdamW(
            params, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay,
        )
    # fp16 参数会产生 fp16 梯度，GradScaler 无法 unscale；只在 fp32 主权重时启用
    use_scaler = args.amp and args.dtype == "float32"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    start_step = 0
    task_counts = {"reca": 0, "t2a": 0}
    if args.resume and Path(args.resume).exists():
        state = torch.load(args.resume, map_location="cpu")
        dit.load_state_dict(state["dit"])
        optim.load_state_dict(state["optim"])
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        start_step = int(state.get("step", 0))
        task_counts.update(state.get("task_counts", {}))
        rng = state.get("rng", {})
        if "python" in rng:
            random.setstate(rng["python"])
        if "prompt" in rng:
            prompt_rng.setstate(rng["prompt"])
        if "torch" in rng:
            torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng.get("cuda"):
            torch.cuda.set_rng_state_all(rng["cuda"])
        if "data_generator" in rng:
            data_generator.set_state(rng["data_generator"])
        if "task_generator" in rng:
            task_generator.set_state(rng["task_generator"])
        if "flow_generator" in rng:
            flow_generator.set_state(rng["flow_generator"])
        print(f"[resume] from {args.resume} @ step {start_step}", flush=True)

    log_path = out / "train_log.jsonl"
    log_f = log_path.open("a", encoding="utf-8")

    def save(step: int) -> None:
        payload = {
            "dit": {k: v.detach().float().cpu() for k, v in dit.state_dict().items()},
            "optim": optim.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "task_counts": dict(task_counts),
            "rng": {
                "python": random.getstate(),
                "prompt": prompt_rng.getstate(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "data_generator": data_generator.get_state(),
                "task_generator": task_generator.get_state(),
                "flow_generator": flow_generator.get_state(),
            },
            "args": vars(args),
        }
        path = out / f"dit_step{step:06d}.pt"
        torch.save(payload, path)
        # latest.pt 用软链，避免每步再复制一份 5.7 GB
        link = out / "latest.pt"
        try:
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(path.name)
        except OSError:
            pass
        print(f"[save] {path}", flush=True)

    it = iter(loader)
    t0 = time.time()
    running: list[float] = []
    mode_running: dict[str, list[float]] = {"reca": [], "t2a": []}
    step = start_step
    while step < args.steps:
        optim.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(args.grad_accum):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)

            audio = batch["audio"].to(device, non_blocking=True)
            with torch.no_grad():
                latents = model.pretransform.encode(audio.to(vae_device)).detach().to(device)
                # 混合训练按 batch 统一选任务，避免同一 batch 里 ReCA/T2A 混排
                # （conditioner 会各自过滤 None 音频，混排可能破坏 batch 对齐）
                draw = float(torch.rand((), generator=task_generator).item())
                batch_mode = choose_task_mode(
                    args.task_mode, args.lambda_reca, args.lambda_t2a, draw
                )
                # 条件在 conditioner 所在卡上构造
                metas = [
                    build_metadata(
                        batch, i, batch_mode, args, cond_device, prompt_rng=prompt_rng
                    )
                    for i in range(len(batch["caption"]))
                ]
                cond_tensors = model.conditioner(metas, cond_device)
                cond_tensors = move_cond(cond_tensors, device)
                cond_inputs = model.get_conditioning_inputs(cond_tensors)

            with torch.amp.autocast("cuda", enabled=args.amp, dtype=torch.float16):
                loss, stats = rectified_flow_loss(
                    dit, latents, cond_inputs,
                    cfg_dropout_prob=args.cfg_dropout,
                    generator=flow_generator,
                )
            scaler.scale(loss / args.grad_accum).backward()
            accum_loss += stats["loss"]
            task_counts[batch_mode] += 1
            mode_running[batch_mode].append(stats["loss"])

        for g in optim.param_groups:
            g["lr"] = lr_at(step, args)
        if args.grad_clip > 0:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_([p for p in dit.parameters() if p.requires_grad], args.grad_clip)
        scaler.step(optim)
        scaler.update()

        step += 1
        running.append(accum_loss / args.grad_accum)
        if step % args.log_every == 0:
            recent = sum(running[-args.log_every:]) / len(running[-args.log_every:])
            dt = (time.time() - t0) / max(1, step - start_step)
            rec = {
                "step": step,
                "loss": round(recent, 5),
                "lr": optim.param_groups[0]["lr"],
                "sec_per_step": round(dt, 3),
                "reca_count": task_counts["reca"],
                "t2a_count": task_counts["t2a"],
                "reca_loss": (
                    round(sum(mode_running["reca"][-args.log_every:]) / len(mode_running["reca"][-args.log_every:]), 5)
                    if mode_running["reca"]
                    else None
                ),
                "t2a_loss": (
                    round(sum(mode_running["t2a"][-args.log_every:]) / len(mode_running["t2a"][-args.log_every:]), 5)
                    if mode_running["t2a"]
                    else None
                ),
            }
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            print(
                f"[step {step}/{args.steps}] loss={recent:.5f} lr={rec['lr']:.2e} "
                f"modes=R{task_counts['reca']}/T{task_counts['t2a']} {dt:.2f}s/step",
                flush=True,
            )

        if args.save_every and step % args.save_every == 0:
            save(step)

    save(step)
    log_f.close()
    print(f"[done] {step} steps -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
