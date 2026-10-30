"""音频读写与小工具。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg")


def load_audio(path: str, sr: int = 16000, mono: bool = True):
    """返回 (torch.FloatTensor[channels, samples], sr)。"""
    import torch
    import torchaudio

    wav, orig_sr = torchaudio.load(str(path))
    if mono and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)
    return wav, sr


def save_audio(path: str, wav, sr: int) -> None:
    """接受 (channels, samples) 的 float / int16 tensor。"""
    import numpy as np
    import soundfile as sf

    arr = wav.detach().cpu().numpy() if hasattr(wav, "detach") else np.asarray(wav)
    if arr.ndim == 1:
        arr = arr[:, None]
    else:
        arr = arr.T  # (C, N) -> (N, C)
    if arr.dtype != np.int16:
        arr = np.clip(arr, -1.0, 1.0)
    sf.write(str(path), arr, sr, subtype="PCM_16")


def list_audio(directory: str) -> List[Path]:
    root = Path(directory)
    files: List[Path] = []
    for ext in AUDIO_EXTS:
        files.extend(sorted(root.rglob(f"*{ext}")))
    return files


def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: Iterable[dict]) -> int:
    n = 0
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n
