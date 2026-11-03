"""条件构造：ReCA（音频 → 理解塔）与 T2A（文本 → 理解塔）。

Audio-Omni 的 conditioner 期望 batch_metadata 是 **list[dict]**，键与
config 里的 conditioning.configs[].id 对应：

    omni_prompt        qwen_omni   —— 这里放 audio_prompt（ReCA）或 text_prompt（T2A）
    speech_prompt      tts-f5      —— TTS 用，ReCA/T2A 传 None
    sync_feature       sync_feature—— 视频同步特征，ReCA/T2A 传零
    audio_input_prompt mel_spec    —— 编辑/TTS 用源音频；ReCA 必须传零，否则泄漏重建目标

`text_prompt` 是普通元数据字段（tts-f5 读取），不参与 cross-attn。
"""
from __future__ import annotations

import importlib.util
import random
import typing as tp
from pathlib import Path

import torch

from audio_omni.prompts import build_prompt

EMPTY_SYNC_DIM = (1, 240, 768)

# 兜底指令；正式训练用 runs/scripts/const.py 里的多样化提示词
RECA_INSTRUCTION = "Describe the audio."

# audio_input_prompt 的 mel 投影层在 config 里写死 1723 帧 = 10.0 s @44.1kHz/hop256。
# 它和扩散头的 sample_size（485100 采样 = 11 s）不是同一个长度，必须分开处理。
MEL_SECONDS = 10.0

_PROMPTS: tp.Optional[tp.List[str]] = None


def recon_prompts(const_path: str = "") -> tp.List[str]:
    """加载 `runs/scripts/const.py` 里的音频重建提示词列表（带缓存）。"""
    global _PROMPTS
    if _PROMPTS is not None:
        return _PROMPTS

    candidates = []
    if const_path:
        candidates.append(Path(const_path))
    here = Path(__file__).resolve()
    candidates.append(here.parents[3] / "runs" / "scripts" / "const.py")  # repo root/runs/scripts

    for path in candidates:
        if not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location("_reca_const", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001
            print(f"[conditioning] 读取 {path} 失败：{exc}")
            continue
        getter = getattr(module, "get_audio_recon_prompt_list", None)
        if callable(getter):
            prompts = [p for p in getter() if isinstance(p, str) and p.strip()]
            if prompts:
                _PROMPTS = prompts
                print(f"[conditioning] 载入 {len(prompts)} 条重建提示词 <- {path}")
                return _PROMPTS

    _PROMPTS = [RECA_INSTRUCTION]
    print("[conditioning] 未找到 const.py，退回固定指令")
    return _PROMPTS


def sample_recon_prompt() -> str:
    return random.choice(recon_prompts())



def _zero_sync(device: tp.Union[str, torch.device]) -> torch.Tensor:
    return torch.zeros(*EMPTY_SYNC_DIM, dtype=torch.float32, device=device)


def build_reca_metadata(
    clip,
    device: tp.Union[str, torch.device] = "cuda",
    seconds: float = 10.0,
    sample_rate: int = 44100,
) -> tp.Dict[str, tp.Any]:
    """ReCA：用**同一段音频**的理解塔表征做条件，重建这段音频。

    audio_input_prompt 故意传全零音频——它是 mel 条件的入口，
    若把原音频放进去，模型就能“抄”目标，ReCA 就没意义了。
    """
    zeros_wav = torch.zeros(int(sample_rate * MEL_SECONDS), dtype=torch.float32, device=device)
    return {
        "omni_prompt": {
            # Qwen2_5OmniProcessor 不接受纯音频输入，必须带 text；
            # 每条样本随机抽一条重建指令（对应 ReCA 原文“多样化提示词防僵化”）。
            "text_prompt": sample_recon_prompt(),
            "video_prompt": None,
            "audio_prompt": clip.audio_omni.float().cpu().numpy(),
        },
        "text_prompt": "",
        "audio_input_prompt": {"audio_input_wav": zeros_wav, "TASK_TTS": False},
        "sync_feature": _zero_sync(device),
        "speech_prompt": None,
    }


def build_t2a_metadata(
    clip,
    device: tp.Union[str, torch.device] = "cuda",
    seconds: float = 10.0,
    sample_rate: int = 44100,
    task: str = "T2A",
) -> tp.Dict[str, tp.Any]:
    """T2A/T2M：文本条件 + 重建同一段音频（给混合训练用）。"""
    caption = clip.caption.strip() or "audio"
    zeros_wav = torch.zeros(int(sample_rate * MEL_SECONDS), dtype=torch.float32, device=device)
    return {
        "omni_prompt": {
            "text_prompt": build_prompt(task, caption),
            "video_prompt": None,
            "audio_prompt": None,
        },
        "text_prompt": caption,
        "audio_input_prompt": {"audio_input_wav": zeros_wav, "TASK_TTS": False},
        "sync_feature": _zero_sync(device),
        "speech_prompt": None,
    }
