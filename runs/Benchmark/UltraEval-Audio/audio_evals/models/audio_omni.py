"""UltraEval-Audio adapter for Audio-Omni.

Registers `--model audio-omni`, wrapping `audio_omni.api.AudioOmni`.

UltraEval 的 PromptStruct 有几种形态（见 audio_evals/base.py）：
- 纯字符串
- {"audio": path, "text": ...}（simple-asr 这类简化模板）
- [{"role": "user", "contents": [{"type": "audio"|"text", "value": ...}]}]

这里统一抽成 (question, audio_path)。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from audio_evals.base import PromptStruct
from audio_evals.models.model import Model

DEFAULT_QUESTION = "Transcribe the audio content into text."


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

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_audio(path: str) -> str:
        """数据集里的音频常是相对路径（raw/...），相对 cwd 解析一次。"""
        p = Path(path)
        if p.is_absolute() or p.exists():
            return str(p)
        alt = Path.cwd() / p
        return str(alt) if alt.exists() else str(p)

    def _parse(self, prompt: PromptStruct) -> Tuple[str, Optional[str]]:
        texts: List[str] = []
        audios: List[str] = []

        def walk(node) -> None:
            if isinstance(node, str):
                texts.append(node)
            elif isinstance(node, dict):
                # 标准对话结构：{"type": "audio"|"text", "value": ...}
                node_type = node.get("type")
                if node_type == "audio" and isinstance(node.get("value"), str):
                    audios.append(self._resolve_audio(node["value"]))
                    return
                if node_type == "text" and isinstance(node.get("value"), str):
                    texts.append(node["value"])
                    return
                for key, value in node.items():
                    if key in ("audio", "audio_path", "wav", "WavPath"):
                        if isinstance(value, str):
                            audios.append(self._resolve_audio(value))
                    elif key in ("text", "content", "question", "instruction"):
                        walk(value)
                    elif key in ("contents", "value"):
                        walk(value)
            elif isinstance(node, (list, tuple)):
                for item in node:
                    walk(item)

        walk(prompt)
        question = "\n".join(t for t in texts if t and not t.startswith("raw/")).strip()
        return question or DEFAULT_QUESTION, (audios[0] if audios else None)

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        question, audio = self._parse(prompt)
        return self.model.understand(question, audio=audio)
