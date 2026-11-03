"""预计算 ReCA 的训练条件与 VAE latent。

动机：每张 V100 放不下“18.9 GB 冻结理解塔 + 3B DiT + 梯度”，所以把
条件（理解塔输出）和 latent（VAE 输出）离线算好，训练阶段只跑 DiT 的 DDP。

用法（8 卡并行，每卡一个分片）：
    bash runs/scripts/precompute_reca.sh
单卡调试：
    python -m reca_audio.train.precompute --manifest ... --out ... --shard-index 0 --num-shards 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from audio_omni.loader import load_model_bundle

from .conditioning import build_reca_metadata
from .data import AudioManifestDataset


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-config", default="Audio-Omni/model/Audio-Omni.json")
    ap.add_argument("--ckpt", default="Audio-Omni/model/model.ckpt")
    ap.add_argument("--seconds", type=float, default=11.0)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--print-every", type=int, default=20)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"[precompute] shard {args.shard_index}/{args.num_shards} on {device}", flush=True)
    model, model_config, sample_rate, sample_size = load_model_bundle(
        json.loads(Path(args.model_config).read_text()),
        args.ckpt,
        device=str(device),
        conditioner_device=str(device),
        dtype="float32",
    )
    model.pretransform.to(device)
    print(f"[precompute] sample_rate={sample_rate} sample_size={sample_size}", flush=True)

    ds = AudioManifestDataset(args.manifest, seconds=args.seconds, sr=sample_rate, limit=args.limit)
    indices = list(range(args.shard_index, len(ds), args.num_shards))
    print(f"[precompute] {len(ds)} clips, 本分片 {len(indices)} 条", flush=True)

    for n, idx in enumerate(indices):
        dst = out / f"{idx:06d}.pt"
        if dst.exists():
            continue
        clip = ds[idx]
        audio = clip.audio.unsqueeze(0).to(device)
        with torch.no_grad():
            latent = model.pretransform.encode(audio).half().cpu()[0]
            meta = build_reca_metadata(clip, device=device, seconds=args.seconds, sample_rate=sample_rate)
            cond = model.conditioner([meta], device)
            cond_inputs = model.get_conditioning_inputs(cond)
            c = cond_inputs["cross_attn_cond"][0].half().cpu()
            m = cond_inputs["cross_attn_mask"][0].bool().cpu()

        torch.save(
            {
                "latent": latent,
                "cond": c,
                "mask": m,
                "prompt": meta["omni_prompt"]["text_prompt"],
                "path": clip.path,
            },
            dst,
        )
        if (n + 1) % args.print_every == 0:
            print(f"[precompute] {n + 1}/{len(indices)} -> {dst.name} cond={tuple(c.shape)} latent={tuple(latent.shape)}", flush=True)

    print(f"[precompute] shard {args.shard_index} done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
