"""把公开数据集整理成统一的 jsonl（评测脚本只吃 jsonl）。

统一格式：
- 生成：{"id": str, "prompt": str}
- 理解：{"id": str, "question": str, "answer": str, "choices": [..]?, "audio": str}
- 编辑：{"id": str, "task": "Add|Remove|Extract|Style Transfer", "source_audio": str,
         "desc": str?, "target_audio": str?}

HF 数据集里的 Audio 字段既可能是 {"array","sampling_rate"}，也可能是本地路径，
`_save_audio_value()` 两种都处理。
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional


def _save_audio_value(value: Any, dst: Path, default_sr: int = 16000) -> Optional[str]:
    import numpy as np
    import soundfile as sf

    dst.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, dict) and value.get("array") is not None:
        arr = np.asarray(value["array"])
        sr = int(value.get("sampling_rate") or default_sr)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        sf.write(str(dst), arr, sr, subtype="PCM_16")
        return str(dst)
    if isinstance(value, (str, os.PathLike)) and os.path.exists(str(value)):
        shutil.copy(str(value), dst)
        return str(dst)
    if hasattr(value, "get_wav"):  # torchcodec / datasets 的 AudioDecoder
        arr, sr = value.get_wav()
        sf.write(str(dst), arr, int(sr), subtype="PCM_16")
        return str(dst)
    return None


def export_from_hf(
    dataset_id: str,
    split: str,
    out: str,
    mapping: Dict[str, Optional[str]],
    audio_dir: Optional[str] = None,
    audio_field: Optional[str] = None,
    limit: int = 0,
    config: Optional[str] = None,
) -> List[dict]:
    """从 HF 数据集导出统一 jsonl。

    mapping: 输出字段 -> 数据集字段名（None 表示跳过），例如
        {"id": "id", "prompt": "caption"}
    音频字段单独用 audio_field 指定，会落盘并在 jsonl 里写路径。
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_id, config, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    rows: List[dict] = []
    for i, row in enumerate(ds):
        item = {}
        for dst_key, src_key in mapping.items():
            if src_key is None:
                continue
            item[dst_key] = row.get(src_key)
        item.setdefault("id", f"{i:05d}")

        if audio_field and audio_dir:
            path = _save_audio_value(row.get(audio_field), Path(audio_dir) / f"{item['id']}.wav")
            if path is None:
                print(f"[warn] 第 {i} 条没有可用音频，跳过")
                continue
            item[audio_field] = path
        rows.append(item)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    import json

    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[export] {len(rows)} 条 -> {out}")
    return rows


def try_export(candidates: List[str], **kwargs) -> List[dict]:
    """依次尝试候选 dataset id，第一个成功即返回。"""
    last = None
    for dataset_id in candidates:
        try:
            print(f"[export] 尝试 {dataset_id} ...")
            return export_from_hf(dataset_id=dataset_id, **kwargs)
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"[export] {dataset_id} 失败：{type(e).__name__}: {e}")
    raise SystemExit(
        "所有候选数据集都不可用，请显式指定 --dataset，或手工准备统一格式的 jsonl。\n"
        f"最后一个错误：{last}"
    )


def export_from_csv(
    csv_path: str,
    out: str,
    prompt_col: str = "caption",
    id_col: Optional[str] = None,
    id_suffix_cols: Optional[List[str]] = None,
    limit: int = 0,
) -> List[dict]:
    """从官方发布的 csv 导出统一 jsonl（AudioCaps / MusicCaps 等）。

    AudioCaps 的 test.csv 列：audiocap_id, youtube_id, start_time, caption
    MusicCaps 的 musiccaps-public.csv 列：ytid, start_s, end_s, caption, ...
    """
    import csv
    import json

    keep_cols = ("youtube_id", "ytid", "start_time", "start_s", "end_s")
    rows: List[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if prompt_col not in (reader.fieldnames or []):
            raise SystemExit(
                f"[fatal] csv 里没有列 {prompt_col!r}；实际列：{reader.fieldnames}\n"
                "  用 --prompt-col 指定正确的列名。"
            )
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            base = row.get(id_col) if id_col else f"{i:05d}"
            suffix = "_".join(str(row.get(c, "")).replace(".", "p") for c in (id_suffix_cols or []))
            rid = f"{base}_{suffix}".strip("_") if suffix else str(base)
            item = {"id": rid, "prompt": row[prompt_col]}
            for col in keep_cols:
                if col in row:
                    item[col] = row[col]
            rows.append(item)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[export] {len(rows)} 条 -> {out}")
    return rows
