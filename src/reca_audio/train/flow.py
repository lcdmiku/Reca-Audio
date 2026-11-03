"""Rectified flow 训练损失。

与 `audio_omni.inference.sampling.sample_rf` 的前向约定一致：
    t ~ U(0,1)，x_t = (1-t) * noise + t * x1，模型回归 v = x1 - noise。
"""
from __future__ import annotations

import typing as tp

import torch
import torch.nn.functional as F


def rectified_flow_loss(
    dit,
    latents: torch.Tensor,
    cond_inputs: tp.Dict[str, tp.Any],
    cfg_dropout_prob: float = 0.1,
    generator: tp.Optional[torch.Generator] = None,
) -> tp.Tuple[torch.Tensor, tp.Dict[str, float]]:
    """返回 (loss, 统计量)。latents: [B, C, T]，cond_inputs 来自 get_conditioning_inputs。"""
    b = latents.shape[0]
    noise = torch.randn(latents.shape, device=latents.device, dtype=latents.dtype, generator=generator)
    t = torch.rand(b, device=latents.device, dtype=latents.dtype, generator=generator)

    shape = (b, 1, 1)
    x_t = (1 - t.view(shape)) * noise + t.view(shape) * latents
    target = latents - noise

    pred = dit(x_t, t, **cond_inputs, cfg_dropout_prob=cfg_dropout_prob)
    if isinstance(pred, tuple):
        pred = pred[0]

    loss = F.mse_loss(pred.float(), target.float())
    stats = {
        "loss": float(loss.detach()),
        "t_mean": float(t.mean()),
        "pred_std": float(pred.detach().float().std()),
        "target_std": float(target.detach().float().std()),
    }
    return loss, stats
