"""统一的模型加载入口。

历史上 API（`AudioOmni`）与 Gradio（`interface/gradio.load_model`）各自实现了一遍
加载逻辑，改一处忘一处就会漂移。这里收敛成一个函数，两边都调用它。

职责：
- 构建模型并从 ckpt 加载（strict 失败时降级 strict=False，与 Gradio 原行为一致）
- 可选加载 pretransform ckpt
- 搬到目标设备、冻结参数
- 精度策略：`dtype` 只调生成头；`model_half` 保留旧语义（整模型半精度，VAE 除外）
- 组件并行：把理解塔（Qwen）放到另一张卡，并在模型上记录设备，供采样时取默认值
"""
from __future__ import annotations

import typing as tp

import torch

from .models.factory import create_model_from_config
from .models.utils import load_ckpt_state_dict

_HALF_ALIASES = ("float16", "fp16", "half")


def _normalize_dtype(dtype: tp.Optional[str]) -> tp.Optional[torch.dtype]:
    if dtype is None:
        return None
    if dtype in _HALF_ALIASES:
        return torch.float16
    if dtype in ("float32", "fp32"):
        return torch.float32
    raise ValueError(f"unsupported dtype: {dtype!r}（可选 float16 / float32 / None）")


def _check_cuda_device(dev: torch.device, name: str) -> None:
    """把 "invalid device ordinal" 换成能直接照做的报错。"""
    if dev.type != "cuda":
        return
    n = torch.cuda.device_count()
    if dev.index is not None and dev.index >= n:
        raise ValueError(
            f"{name}={dev} 不可用：当前进程只看到 {n} 张 GPU（cuda:0..cuda:{n - 1}）。"
            " 多半是 CUDA_VISIBLE_DEVICES 限制了可见卡数，例如写成 CUDA_VISIBLE_DEVICES=0 时 cuda:1 就不存在。"
        )


def apply_precision_and_parallelism(
    model: torch.nn.Module,
    *,
    device: tp.Union[str, torch.device] = "cuda",
    conditioner_device: tp.Optional[tp.Union[str, torch.device]] = None,
    dtype: tp.Optional[str] = None,
    model_half: bool = False,
) -> torch.device:
    """把「精度」与「组件并行」两件事收敛到一处，供所有加载路径复用。

    - `dtype`：只改生成头（DiT），理解塔与 VAE 精度不变
    - `model_half`：旧语义，整模型半精度但 VAE 保持 fp32
    - `conditioner_device`：理解塔放到另一张卡；结果记在 `model.conditioner_device`
    """
    device = torch.device(device)
    if conditioner_device is not None:
        conditioner_device = torch.device(conditioner_device)
    else:
        conditioner_device = device
    _check_cuda_device(device, "device")
    _check_cuda_device(conditioner_device, "conditioner_device")

    # VAE 始终 fp32，保证解码精度
    if getattr(model, "pretransform", None) is not None:
        model.pretransform = model.pretransform.to(dtype=torch.float32).eval()

    target_dtype = _normalize_dtype(dtype)
    if target_dtype is not None:
        model.model.to(target_dtype)
    elif model_half:
        pretransform_backup = getattr(model, "pretransform", None)
        if pretransform_backup is not None:
            model.pretransform = None
        model.to(torch.float16)
        if pretransform_backup is not None:
            model.pretransform = pretransform_backup.to(dtype=torch.float32).eval()

    if conditioner_device != device:
        model.conditioner.to(conditioner_device)
    model.conditioner_device = conditioner_device
    return conditioner_device


def load_model_bundle(
    model_config: tp.Dict[str, tp.Any],
    ckpt_path: tp.Optional[str] = None,
    *,
    pretrained_name: tp.Optional[str] = None,
    pretransform_ckpt_path: tp.Optional[str] = None,
    device: tp.Union[str, torch.device] = "cuda",
    conditioner_device: tp.Optional[tp.Union[str, torch.device]] = None,
    dtype: tp.Optional[str] = None,
    model_half: bool = False,
    verbose: bool = True,
) -> tp.Tuple[torch.nn.Module, tp.Dict[str, tp.Any], int, int]:
    """返回 (model, model_config, sample_rate, sample_size)。"""
    if pretrained_name is not None:
        from .models.pretrained import get_pretrained_model

        if verbose:
            print(f"[loader] 加载预训练模型 {pretrained_name}")
        model, model_config = get_pretrained_model(pretrained_name)
    else:
        model = create_model_from_config(model_config)
        if ckpt_path is not None:
            if verbose:
                print(f"[loader] 加载 checkpoint：{ckpt_path}")
            state_dict = load_ckpt_state_dict(ckpt_path)
            try:
                model.load_state_dict(state_dict, strict=True)
            except RuntimeError as e:
                if verbose:
                    print(f"[loader] strict=True 失败，降级 strict=False：{e}")
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if verbose and missing:
                    print(f"[loader] missing keys（将用默认初始化）：{missing[:10]}...")
                if verbose and unexpected:
                    print(f"[loader] unexpected keys（忽略）：{unexpected[:10]}...")

    if pretransform_ckpt_path is not None:
        if verbose:
            print(f"[loader] 加载 pretransform：{pretransform_ckpt_path}")
        pretransform_state_dict = load_ckpt_state_dict(pretransform_ckpt_path)
        if getattr(model, "pretransform", None) is not None:
            model.pretransform.load_state_dict(pretransform_state_dict, strict=False)

    device = torch.device(device)
    model.to(device).eval().requires_grad_(False)

    apply_precision_and_parallelism(
        model,
        device=device,
        conditioner_device=conditioner_device,
        dtype=dtype,
        model_half=model_half,
    )

    if verbose and torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / 1e9
            if alloc > 0.05:
                print(f"[loader] cuda:{i} allocated={alloc:.2f}GB", flush=True)

    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    return model, model_config, sample_rate, sample_size
