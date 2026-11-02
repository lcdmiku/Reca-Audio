"""UltraEval-Audio adapter for Audio-Omni.

Registers `--model audio-omni`, wrapping `audio_omni.api.AudioOmni`:
- prompt 里的 audio 取第一个，交给 understand(question, audio=...)
- 文本部分拼成问题

对应 registry/model/audio_omni.yaml。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

from audio_evals.base import PromptStruct
from audio_evals.models.model import Model


class AudioOmniModel(Model):
    def __init__(
        self,
        is_chat: bool = True,
        sample_params: Dict[str, any] = None,
        model_config: str = "",
        ckpt: str = "",
        device: str = "cuda:0",
        conditioner_device: str = None,
        dtype: str = None,
        repo_root: str = "",
    ):
        super().__init__(True, sample_params)

        if repo_root:
            root = Path(repo_root)
        else:
            # <repo>/runs/Benchmark/UltraEval-Audio/audio_evals/models/audio_omni.py
            root = Path(__file__).resolve().parents[5]
        pkg = root / "Audio-Omni"
        if pkg.is_dir() and str(pkg) not in sys.path:
            sys.path.insert(0, str(pkg))

        from audio_omni.api import AudioOmni

        self.model = AudioOmni(
            model_config,
            ckpt,
            device=device,
            conditioner_device=conditioner_device,
            dtype=dtype,
        )

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        texts, audios = [], []
        for line in prompt:
            for item in line.get("contents", []):
                if item.get("type") == "audio":
                    audios.append(item["value"])
                elif item.get("type") == "text":
                    texts.append(item["value"])
        question = "\n".join(texts).strip()
        audio = audios[0] if audios else None
        return self.model.understand(question, audio=audio)
