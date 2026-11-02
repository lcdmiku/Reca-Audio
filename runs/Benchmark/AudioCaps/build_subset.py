#!/usr/bin/env python
"""从 AudioCaps parquet 分片构建 T2A 评测输入。

数据来源：HF `TwinkStart/AudioCaps` 的分片（audio 内嵌在 parquet 里）。
产物：
    runs/Benchmark/AudioCaps/prompts.jsonl   # {"id","prompt"}
    runs/Benchmark/_data/audiocaps/ref/*.wav # 参考音频（FAD 的真实分布）

用法：
    python runs/Benchmark/AudioCaps/build_subset.py --shards-dir ... --max-items 100
"""
from __future__ import annotations

import argparse
import glob
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf


def valid(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"PAR1"
    except OSError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards-dir", default="runs/Benchmark/_data/audiocaps")
    ap.add_argument("--pattern", default="shard*.parquet", help="分片文件名通配")
    ap.add_argument("--ref-dir", default="runs/Benchmark/_data/audiocaps/ref")
    ap.add_argument("--out-jsonl", default="runs/Benchmark/AudioCaps/prompts.jsonl")
    ap.add_argument("--max-items", type=int, default=0)
    args = ap.parse_args()

    shards = sorted(p for p in glob.glob(str(Path(args.shards_dir) / args.pattern)) if valid(p))
    if not shards:
        raise SystemExit("[fatal] 没有可用的分片")

    ref_dir = Path(args.ref_dir)
    ref_dir.mkdir(parents=True, exist_ok=True)

    rows, seen = [], set()
    for path in shards:
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=["audiocap_id", "caption", "audio"])
            for i in range(table.num_rows):
                cid = str(table.column("audiocap_id")[i].as_py())
                if cid in seen:
                    continue
                cap = table.column("caption")[i].as_py()
                audio = table.column("audio")[i].as_py() or {}
                data = audio.get("bytes")
                if not data or not cap:
                    continue
                dst = ref_dir / f"{cid}.wav"
                if not dst.exists():
                    try:
                        arr, sr = sf.read(io.BytesIO(data))
                        sf.write(str(dst), arr, int(sr), subtype="PCM_16")
                    except Exception as e:  # noqa: BLE001
                        print(f"[skip] {cid}: {type(e).__name__}: {e}")
                        continue
                seen.add(cid)
                rows.append({"id": cid, "prompt": cap})
                if args.max_items and len(rows) >= args.max_items:
                    break
            if args.max_items and len(rows) >= args.max_items:
                break
        print(f"{Path(path).name}: total={len(rows)}")

    out = Path(args.out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[done] {len(rows)} 条 -> {out}；参考音频 {len(list(ref_dir.glob('*.wav')))} 个")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
