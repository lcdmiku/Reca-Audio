"""多进程训练：每进程占 2 张卡——偶数卡放 DiT，奇数卡放冻结的理解塔 + VAE。

和推理时的组件并行完全一致，因此不需要预计算，也不会爆显存：
    rank i:  DiT  -> cuda:(2i)      参数 12 GB + 梯度 12 GB + 激活
             理解塔/VAE -> cuda:(2i+1)   18.9 GB + ~1 GB

4 个进程 = 8 张 V100。

    torchrun --nproc_per_node=4 -m reca_audio.train.train_ddp \
        --mode live --steps 2000 --grad-accum 1

也支持 `--mode cached`（配合 precompute.py 的纯 DiT 数据并行）。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from audio_omni.loader import load_model_bundle

from .conditioning import build_reca_metadata
from .data import AudioManifestDataset, collate_clips
from .flow import rectified_flow_loss


class CachedClips(Dataset):
    def __init__(self, cache: str, limit: int = 0) -> None:
        self.files = sorted(Path(cache).glob("*.pt"))
        if limit:
            self.files = self.files[:limit]
        if not self.files:
            raise ValueError(f"{cache} 里没有缓存样本")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        blob = torch.load(self.files[idx], map_location="cpu")
        return {
            "latent": blob["latent"].float(),
            "cond": blob["cond"].float(),
            "mask": blob["mask"].bool(),
        }


def collate_cached(items):
    return {
        "latent": torch.stack([i["latent"] for i in items], 0),
        "cond": torch.stack([i["cond"] for i in items], 0),
        "mask": torch.stack([i["mask"] for i in items], 0),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["live", "cached"], default="live")
    p.add_argument("--manifest", default="runs/Benchmark/_data/audiocaps/train_manifest.jsonl")
    p.add_argument("--cache", default="runs/train/cache_reca")
    p.add_argument("--out", required=True)
    p.add_argument("--model-config", default="Audio-Omni/model/Audio-Omni.json")
    p.add_argument("--ckpt", default="Audio-Omni/model/model.ckpt")

    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--cfg-dropout", type=float, default=0.1)

    p.add_argument("--seconds", type=float, default=5.0, help="训练片段长度；5 s 可稳住在 32 GB")
    p.add_argument("--gpus-per-rank", type=int, default=2)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--dtype", choices=["float32", "float16"], default="float32",
                   help="DiT 权重精度；fp16 把 12 GB 参数 + 12 GB 梯度压到 12 GB")
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def lr_at(step: int, args) -> float:
    if step < args.warmup:
        return args.lr * (step + 1) / max(1, args.warmup)
    progress = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def move_cond(tensors, device):
    out = {}
    for k, v in tensors.items():
        if isinstance(v, (tuple, list)):
            out[k] = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in v)
        elif isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def main() -> int:
    args = parse_args()
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    per = args.gpus_per_rank
    local = int(os.environ.get("LOCAL_RANK", 0))

    dit_idx = local * per
    aux_idx = dit_idx + 1 if per > 1 else dit_idx
    torch.cuda.set_device(dit_idx)
    if world > 1:
        dist.init_process_group(backend="nccl")
    dit_device = torch.device(f"cuda:{dit_idx}")
    aux_device = torch.device(f"cuda:{aux_idx}")

    out = Path(args.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "args.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False))
    torch.manual_seed(args.seed + rank)

    if args.mode == "cached":
        print(f"[rank {rank}] 加载完整 bundle（CPU）后只保留 DiT", flush=True)
        model, _, _, _ = load_model_bundle(
            json.loads(Path(args.model_config).read_text()), args.ckpt,
            device="cpu", conditioner_device="cpu", dtype="float32",
        )
        dit = model.model
        import gc

        del model.conditioner, model.pretransform
        gc.collect()
        dit = dit.to(dit_device)
        conditioner = None
        vae = None
    else:
        print(f"[rank {rank}] DiT -> {dit_device} | 理解塔/VAE -> {aux_device}", flush=True)
        model, _, sample_rate, _ = load_model_bundle(
            json.loads(Path(args.model_config).read_text()), args.ckpt,
            device=str(dit_device), conditioner_device=str(aux_device), dtype=args.dtype,
        )
        model.conditioner_device = aux_device
        dit = model.model
        conditioner = model.conditioner
        vae = model.pretransform.to(aux_device).float()  # VAE 始终 fp32

    dit.train()
    for p in dit.parameters():
        p.requires_grad_(True)
    if world > 1:
        dit = DDP(
            dit,
            device_ids=[dit_idx],
            find_unused_parameters=False,
            gradient_as_bucket_view=True,   # 梯度桶复用 grad 存储，省一大块显存
            static_graph=True,              # 图固定，省去每步重建
        )

    n_train = sum(p.numel() for p in dit.parameters() if p.requires_grad)
    if rank == 0:
        print(f"[ddp] world={world} gpus/rank={per} trainable={n_train/1e6:.1f}M", flush=True)

    if args.mode == "cached":
        dataset = CachedClips(args.cache, args.limit)
        collate = collate_cached
    else:
        dataset = AudioManifestDataset(args.manifest, seconds=args.seconds, limit=args.limit)
        collate = collate_clips
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True) if world > 1 else None
    loader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=sampler is None, sampler=sampler,
        num_workers=args.num_workers, collate_fn=collate,
        drop_last=True, pin_memory=True,
    )

    from transformers.optimization import Adafactor

    optim = Adafactor(
        [p for p in dit.parameters() if p.requires_grad],
        lr=args.lr, scale_parameter=False, relative_step=False,
        warmup_init=False, weight_decay=args.weight_decay,
    )
    # 参数是 fp16 时梯度也是 fp16，GradScaler 无法 unscale，直接关掉
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and args.dtype == "float32")

    def save(step: int) -> None:
        if rank != 0:
            return
        inner = dit.module if world > 1 else dit
        payload = {
            "dit": {k: v.detach().half().cpu() for k, v in inner.state_dict().items()},
            "optim": optim.state_dict(),
            "step": step,
            "args": vars(args),
        }
        path = out / f"dit_step{step:06d}.pt"
        torch.save(payload, path)
        link = out / "latest.pt"
        try:
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(path.name)
        except OSError:
            pass
        print(f"[save] {path}", flush=True)

    log_f = (out / "train_log.jsonl").open("a", encoding="utf-8") if rank == 0 else None
    epoch = 0
    if sampler:
        sampler.set_epoch(epoch)
    it = iter(loader)
    step, t0 = 0, time.time()
    running = []

    while step < args.steps:
        optim.zero_grad(set_to_none=True)
        acc = 0.0
        for _ in range(args.grad_accum):
            try:
                batch = next(it)
            except StopIteration:
                epoch += 1
                if sampler:
                    sampler.set_epoch(epoch)
                it = iter(loader)
                batch = next(it)

            if args.mode == "cached":
                latent = batch["latent"].to(dit_device, non_blocking=True)
                cond = batch["cond"].to(dit_device, non_blocking=True)
                mask = batch["mask"].to(dit_device, non_blocking=True)
                cond_inputs = {"cross_attn_cond": cond, "cross_attn_mask": mask}
            else:
                audio = batch["audio"].to(dit_device, non_blocking=True)
                with torch.no_grad():
                    # VAE 与理解塔都在 aux 卡，条件算完只把几十 MB 搬回 DiT 卡
                    latent = vae.encode(audio.to(aux_device)).detach().to(dit_device)
                    metas = [
                        build_reca_metadata(
                            type("C", (), {"audio_omni": batch["audio_omni"][i], "caption": batch["caption"][i]})(),
                            device=aux_device, seconds=args.seconds, sample_rate=sample_rate,
                        )
                        for i in range(len(batch["caption"]))
                    ]
                    cond_tensors = conditioner(metas, aux_device)
                    cond_inputs = model.get_conditioning_inputs(move_cond(cond_tensors, dit_device))

            with torch.amp.autocast("cuda", enabled=args.amp, dtype=torch.float16):
                loss, stats = rectified_flow_loss(
                    dit, latent, cond_inputs, cfg_dropout_prob=args.cfg_dropout
                )
            scaler.scale(loss / args.grad_accum).backward()
            acc += stats["loss"]

        for g in optim.param_groups:
            g["lr"] = lr_at(step, args)
        if args.grad_clip > 0:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_([p for p in dit.parameters() if p.requires_grad], args.grad_clip)
        scaler.step(optim)
        scaler.update()

        step += 1
        running.append(acc / args.grad_accum)
        if rank == 0 and step % args.log_every == 0:
            recent = sum(running[-args.log_every:]) / len(running[-args.log_every:])
            dt = (time.time() - t0) / step
            rec = {"step": step, "loss": round(recent, 5), "lr": optim.param_groups[0]["lr"], "s_per_step": round(dt, 3)}
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            print(f"[step {step}/{args.steps}] loss={recent:.5f} lr={rec['lr']:.2e} {dt:.2f}s/step", flush=True)
        if args.save_every and step % args.save_every == 0:
            save(step)

    save(step)
    if log_f:
        log_f.close()
    if world > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
