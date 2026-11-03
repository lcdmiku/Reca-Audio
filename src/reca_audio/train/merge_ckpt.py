"""把训练得到的 DiT 权重合并回完整 model.ckpt，供现有推理/评测脚本直接使用。

完整 ckpt 结构：{"state_dict": {...}}，DiT 的键形如
    model.model.transformer.layers.0.pre_norm.beta
而我们保存的是 DiTWrapper 的 state_dict，键形如
    model.transformer.layers.0.pre_norm.beta
所以合并时统一加前缀 "model."。

    python -m reca_audio.train.merge_ckpt \
        --ckpt Audio-Omni/model/model.ckpt \
        --dit runs/train/reca_v1/latest.pt \
        --out Audio-Omni/model/model_reca.ckpt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="原始 21 GB 完整 ckpt")
    ap.add_argument("--dit", required=True, help="训练产出的 dit_step*.pt / latest.pt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print("[merge] 读取原始 ckpt（约 21 GB，需要足够内存）", flush=True)
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = blob["state_dict"] if "state_dict" in blob else blob

    payload = torch.load(args.dit, map_location="cpu", weights_only=False)
    dit = payload["dit"] if "dit" in payload else payload
    print(f"[merge] DiT 张量 {len(dit)} 个，step={payload.get('step', '?')}", flush=True)

    updated, missing = 0, []
    for key, value in dit.items():
        full = "model." + key
        if full in sd:
            sd[full] = value.to(sd[full].dtype)
            updated += 1
        else:
            missing.append(full)

    if updated == 0:
        # 兜底：按后缀匹配（改动过 wrapper 命名时仍能合上）
        suffix_map = {k.split("transformer.", 1)[-1]: k for k in sd if ".transformer." in k}
        for key, value in dit.items():
            tail = key.split("transformer.", 1)[-1] if "transformer." in key else key
            full = suffix_map.get(tail)
            if full:
                sd[full] = value.to(sd[full].dtype)
                updated += 1
            else:
                missing.append(key)

    print(f"[merge] 已更新 {updated}/{len(dit)}", flush=True)
    if missing:
        print(f"[merge] 未匹配 {len(missing)} 个，例如 {missing[:3]}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": sd}, out)
    print(f"[merge] -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
