"""把原始音频统一成 24 kHz 单声道 PCM16，并生成 index.jsonl。

设计取舍：
- **不做切片**。训练时再随机裁 10 s，这样短音效不会被人为补成长空白。
- 只做格式归一 + 响度归一（-20 dBFS RMS，带峰值保护）+ 去重 + 质量过滤。

用法（在仓库根目录，已 `pip install -e .`）：
    python -m reca_audio.data.preprocess \
        --input  "$DATA_ROOT/raw/fsd50k/FSD50K.dev_audio" \
        --out    "$DATA_ROOT/processed/fsd50k" \
        --workers 32

先小跑验证：
    python -m reca_audio.data.preprocess --input ... --out ... --limit 200
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Optional

import numpy as np

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac", ".wma"}

TARGET_SR = 24000
TARGET_RMS_DB = -20.0
SILENCE_DB = -55.0
MIN_SECONDS = 0.5


# ---------------------------------------------------------------- 工具

def cheap_key(path: Path, nbytes: int = 256 * 1024) -> str:
    """内容指纹：文件大小 + 头部哈希。足够抓重复，又不读整个文件。"""
    h = hashlib.md5()
    size = path.stat().st_size
    h.update(str(size).encode())
    with path.open("rb") as f:
        h.update(f.read(nbytes))
    return h.hexdigest()


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """优先 soundfile；失败再退回 librosa（m4a 之类需要 ffmpeg）。"""
    import soundfile as sf

    try:
        y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception:
        import librosa

        y, sr = librosa.load(str(path), sr=None, mono=True)
        return np.asarray(y, dtype=np.float32).reshape(-1), int(sr)

    if y.ndim > 1:
        y = y.mean(axis=1)
    return np.asarray(y, dtype=np.float32).reshape(-1), int(sr)


def normalize(y: np.ndarray) -> tuple[np.ndarray, float, float]:
    """RMS 归一到 TARGET_RMS_DB，并保证峰值不过 0.99。返回 (y, 原rms_db, 实际增益db)。"""
    rms = float(np.sqrt(np.mean(np.square(y), dtype=np.float64) + 1e-12))
    rms_db = 20.0 * np.log10(rms + 1e-12)
    gain_db = TARGET_RMS_DB - rms_db
    peak = float(np.max(np.abs(y)) + 1e-9)
    if peak * (10.0 ** (gain_db / 20.0)) > 0.99:
        gain_db = 20.0 * np.log10(0.99 / peak)
    y = np.clip(y * (10.0 ** (gain_db / 20.0)), -1.0, 1.0)
    return y.astype(np.float32), rms_db, float(gain_db)


# ---------------------------------------------------------------- worker

def process_one(job: tuple[str, str, bool]) -> dict[str, Any]:
    src_str, out_dir_str, overwrite = job
    src = Path(src_str)
    out_dir = Path(out_dir_str)
    try:
        key = cheap_key(src)
        uid = key[:16]
        out_path = out_dir / f"{uid}.wav"
        if out_path.exists() and not overwrite:
            return {"status": "exists", "uid": uid, "src": str(src)}

        import soundfile as sf

        y, sr = load_audio(src)
        dur = len(y) / float(sr)
        if dur < MIN_SECONDS:
            return {"status": "short", "src": str(src), "dur": dur}

        if sr != TARGET_SR:
            import librosa

            y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR, res_type="soxr_hq")

        y, rms_db, gain_db = normalize(y)
        if rms_db < SILENCE_DB:
            return {"status": "silent", "src": str(src), "rms_db": rms_db}

        sf.write(str(out_path), y, TARGET_SR, subtype="PCM_16")
        return {
            "status": "ok",
            "uid": uid,
            "src": str(src),
            "path": str(out_path),
            "dur": round(len(y) / TARGET_SR, 3),
            "sr": TARGET_SR,
            "rms_db": round(rms_db, 2),
            "gain_db": round(gain_db, 2),
        }
    except Exception as e:  # noqa: BLE001
        return {
            "status": "error",
            "src": str(src),
            "err": f"{type(e).__name__}: {e}",
            "tb": traceback.format_exc(limit=2),
        }


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="原始音频目录（递归）")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个（冒烟用）")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.out)
    if not in_dir.is_dir():
        print(f"[fatal] 输入目录不存在: {in_dir}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in in_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTS)
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"[fatal] 目录里没有音频文件: {in_dir}", file=sys.stderr)
        return 2

    print(f"[info] 待处理 {len(files)} 个文件 -> {out_dir}")
    jobs = [(str(p), str(out_dir), args.overwrite) for p in files]

    stats = {"ok": 0, "exists": 0, "short": 0, "silent": 0, "error": 0}
    total_dur = 0.0
    index_path = out_dir / "index.jsonl"
    seen_keys: set[str] = set()
    dup = 0

    with index_path.open("w", encoding="utf-8") as fout, Pool(args.workers) as pool:
        for i, rec in enumerate(pool.imap_unordered(process_one, jobs, chunksize=4), 1):
            status = rec["status"]
            stats[status] = stats.get(status, 0) + 1

            if status == "ok":
                if rec["uid"] in seen_keys:
                    dup += 1
                else:
                    seen_keys.add(rec["uid"])
                    total_dur += rec["dur"]
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            elif status == "error" and stats["error"] <= 5:
                print(f"[warn] {rec['src']}: {rec['err']}", file=sys.stderr)

            if i % 200 == 0 or i == len(files):
                print(
                    f"[{i}/{len(files)}] ok={stats['ok']} dup={dup} "
                    f"short={stats['short']} silent={stats['silent']} err={stats['error']} "
                    f"hours={total_dur / 3600:.1f}",
                    flush=True,
                )

    manifest = {
        "input": str(in_dir),
        "output": str(out_dir),
        "n_files": len(files),
        "n_ok": stats["ok"],
        "n_dup": dup,
        "n_short": stats["short"],
        "n_silent": stats["silent"],
        "n_error": stats["error"],
        "hours": round(total_dur / 3600, 2),
        "target_sr": TARGET_SR,
        "target_rms_db": TARGET_RMS_DB,
        "index": str(index_path),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
