"""统一的模型加载入口。

历史上 API（`AudioOmni`）与 Gradio（`interface/gradio.load_model`）各自实现了一遍
加载逻辑，改一处忘一处就会漂移。这里收敛成一组函数，两边都调用。

职责拆分：
- `move_model()`：只负责“放哪张卡”，支持把理解塔单独放到另一张卡（组件并行）
- `apply_precision()`：只负责精度（dtype / model_half）
- `apply_precision_and_parallelism()`：两者的组合，兼容旧调用
- `load_model_bundle()`：构建 + 加载 ckpt + 搬设备 + 精度，返回四元组

组件并行的关键细节：搬模型时先把 conditioner 摘出来，避免 12 GB 的 Qwen
在 `.to(device)` 中途先占满生成卡、再被搬到另一张卡——那会让“两卡”在加载阶段就 OOM。
"""
from __future__ import annotations

import os
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


def move_model(
    model: torch.nn.Module,
    device: tp.Union[str, torch.device] = "cuda",
    conditioner_device: tp.Optional[tp.Union[str, torch.device]] = None,
) -> torch.device:
    """把模型搬到 `device`；理解塔可单独放到 `conditioner_device`。

    先摘出 conditioner 再搬主干，避免它在生成卡上占满显存后被搬走。
    结果记录在 `model.conditioner_device`，采样函数默认读取。
    """
    device = torch.device(device)
    if conditioner_device is not None:
        conditioner_device = torch.device(conditioner_device)
    else:
        conditioner_device = device
    _check_cuda_device(device, "device")
    _check_cuda_device(conditioner_device, "conditioner_device")

    if conditioner_device != device:
        conditioner = model.conditioner
        model.conditioner = torch.nn.Identity()  # 临时占位，._apply 时跳过
        model.to(device).eval().requires_grad_(False)
        model.conditioner = conditioner
        # 刚才被跳过，这里补上 eval/冻结，保持与整体一致
        conditioner.to(conditioner_device).eval().requires_grad_(False)
    else:
        model.to(device).eval().requires_grad_(False)

    model.conditioner_device = conditioner_device
    return conditioner_device


def apply_precision(
    model: torch.nn.Module,
    dtype: tp.Optional[str] = None,
    model_half: bool = False,
) -> None:
    """精度策略：`dtype` 只改生成头；`model_half` 保留旧语义（VAE 始终 fp32）。"""
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


def apply_precision_and_parallelism(
    model: torch.nn.Module,
    *,
    device: tp.Union[str, torch.device] = "cuda",
    conditioner_device: tp.Optional[tp.Union[str, torch.device]] = None,
    dtype: tp.Optional[str] = None,
    model_half: bool = False,
) -> torch.device:
    """搬设备 + 定精度。保留这个名字，方便旧调用方一行接入。"""
    cond_device = move_model(model, device=device, conditioner_device=conditioner_device)
    apply_precision(model, dtype=dtype, model_half=model_half)
    return cond_device


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

    if verbose:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
        print(
            f"[loader] CUDA_VISIBLE_DEVICES={visible} | visible_gpus={torch.cuda.device_count()} | "
            f"device={torch.device(device)} | conditioner_device={conditioner_device or device} | "
            f"dtype={dtype or ('float16' if model_half else 'float32')}",
            flush=True,
        )

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
