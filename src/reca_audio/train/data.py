"""音频数据集：manifest.jsonl 每行 {"audio": path, "caption": str?}。

统一裁到固定时长（默认 10 s），同时给出
- 44.1 kHz 双声道：VAE 的重建目标
- 16 kHz 单声道：理解塔（Qwen2.5-Omni）的条件输入
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torchaudio

_TARGET_SR = 44100
_OMNI_SR = 16000


@dataclass
class AudioClip:
    audio: torch.Tensor        # [2, T] @ 44.1 kHz
    audio_omni: torch.Tensor   # [T16] @ 16 kHz，喂给理解塔
    caption: str
    path: str


def _read_audio(path: str, target_sr: int, channels: int, seconds: float) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if channels == 2:
        wav = wav.repeat(2, 1)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    want = int(target_sr * seconds)
    if wav.shape[-1] < want:
        pad = want - wav.shape[-1]
        wav = torch.nn.functional.pad(wav, (0, pad))
    elif wav.shape[-1] > want:
        start = random.randint(0, wav.shape[-1] - want)
        wav = wav[..., start : start + want]
    return wav.float()


class AudioManifestDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        manifest: str,
        seconds: float = 10.0,
        sr: int = _TARGET_SR,
        channels: int = 2,
        limit: int = 0,
    ) -> None:
        self.rows: List[Dict[str, Any]] = []
        with open(manifest, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        if limit:
            self.rows = self.rows[:limit]
        if not self.rows:
            raise ValueError(f"manifest 为空：{manifest}")
        self.seconds = seconds
        self.sr = sr
        self.channels = channels

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> AudioClip:
        row = self.rows[idx]
        path = row.get("audio") or row.get("path") or row.get("wav")
        if not path:
            raise KeyError(f"第 {idx} 行没有 audio/path 字段")
        if not Path(path).is_absolute():
            path = str(Path(path).resolve())

        audio = _read_audio(path, self.sr, self.channels, self.seconds)
        omni = audio.mean(0)
        if self.sr != _OMNI_SR:
            omni = torchaudio.functional.resample(omni, self.sr, _OMNI_SR)

        return AudioClip(
            audio=audio,
            audio_omni=omni.contiguous(),
            caption=str(row.get("caption") or row.get("prompt") or ""),
            path=path,
        )


def collate_clips(clips: Sequence[AudioClip]) -> Dict[str, Any]:
    return {
        "audio": torch.stack([c.audio for c in clips], dim=0),
        "audio_omni": [c.audio_omni for c in clips],
        "caption": [c.caption for c in clips],
        "path": [c.path for c in clips],
    }
