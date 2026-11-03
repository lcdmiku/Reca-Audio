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
    ap.add_argument("--ids-csv", default="", help="官方 test.csv；只保留其中的 audiocap_id")
    ap.add_argument("--id-cols", default="audiocap_id", help="逗号分隔的 id 列，拼成唯一 id")
    args = ap.parse_args()

    id_cols = [c.strip() for c in args.id_cols.split(",") if c.strip()]

    allow = None
    if args.ids_csv:
        import csv

        with open(args.ids_csv, newline="", encoding="utf-8") as f:
            allow = {str(r["audiocap_id"]).strip() for r in csv.DictReader(f)}
        print(f"[ids] 官方 test.csv: {len(allow)} 个 id")

    shards = sorted(p for p in glob.glob(str(Path(args.shards_dir) / args.pattern)) if valid(p))
    if not shards:
        raise SystemExit("[fatal] 没有可用的分片")

    ref_dir = Path(args.ref_dir)
    ref_dir.mkdir(parents=True, exist_ok=True)

    # 第一遍：只读 id 列，定位命中分片（全量 41 GB，全读太慢）
    if allow is not None:
        hits = {}
        for path in shards:
            try:
                table = pq.read_table(path, columns=id_cols)
            except Exception:  # noqa: BLE001
                continue
            ids = {
                "_".join(str(table.column(c)[i].as_py()) for c in id_cols)
                for i in range(table.num_rows)
            }
            common = ids & allow
            if common:
                hits[path] = common
        print(f"[scan] {len(hits)}/{len(shards)} 个分片命中，共 {sum(len(v) for v in hits.values())} 条")
        shards = sorted(hits)
    else:
        hits = {p: None for p in shards}

    rows, seen = [], set()
    for path in shards:
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=id_cols + ["caption", "audio"])
            for i in range(table.num_rows):
                cid = "_".join(str(table.column(c)[i].as_py()) for c in id_cols)
                if cid in seen:
                    continue
                if allow is not None and cid not in allow:
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
