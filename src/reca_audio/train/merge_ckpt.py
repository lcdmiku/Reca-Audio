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


def prepare_dit_replacements(full_state, dit_state, *, strict: bool = True):
    """Validate and return full-checkpoint replacements without mutating it."""
    expected_full = {key for key in full_state if key.startswith("model.")}
    if not expected_full:
        raise KeyError("base checkpoint contains no model.* DiT keys")

    replacements = {}
    missing = []
    shape_mismatches = []
    for key, value in dit_state.items():
        full = "model." + key
        if full not in full_state:
            missing.append(full)
            continue
        source_shape = tuple(getattr(value, "shape", ()))
        target_shape = tuple(getattr(full_state[full], "shape", ()))
        if source_shape != target_shape:
            shape_mismatches.append((full, source_shape, target_shape))
            continue
        replacements[full] = value

    uncovered = sorted(expected_full - set(replacements))
    if strict:
        if shape_mismatches:
            full, source_shape, target_shape = shape_mismatches[0]
            raise ValueError(
                f"shape mismatch for {full}: trained={source_shape} base={target_shape}"
            )
        if missing:
            raise KeyError(f"unmatched trained DiT keys: {missing[:3]}")
        if uncovered:
            raise KeyError(f"uncovered base DiT keys: {uncovered[:3]}")
    return replacements


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="原始 21 GB 完整 ckpt")
    ap.add_argument("--dit", required=True, help="训练产出的 dit_step*.pt / latest.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="allow missing or shape-mismatched DiT keys (debugging only)",
    )
    args = ap.parse_args()

    print("[merge] 读取原始 ckpt（约 21 GB，需要足够内存）", flush=True)
    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = blob["state_dict"] if "state_dict" in blob else blob

    payload = torch.load(args.dit, map_location="cpu", weights_only=False)
    dit = payload["dit"] if "dit" in payload else payload
    print(f"[merge] DiT 张量 {len(dit)} 个，step={payload.get('step', '?')}", flush=True)

    replacements = prepare_dit_replacements(sd, dit, strict=not args.allow_partial)
    for full, value in replacements.items():
        sd[full] = value.to(sd[full].dtype)
    print(f"[merge] 已更新 {len(replacements)}/{len(dit)}", flush=True)
    if args.allow_partial and len(replacements) != len(dit):
        print("[merge] WARNING: partial merge explicitly allowed", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": sd}, out)
    print(f"[merge] -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
