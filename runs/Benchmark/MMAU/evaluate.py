#!/usr/bin/env python
"""MMAU — 官方协议版评测。

官方仓库：https://github.com/Sakshi113/MMAU
数据：HF `gamma-lab-umd/MMAU-test-mini`（1000 条，带答案；完整 test 的答案不公开）
官方流程：在原始 benchmark 文件里加一个 `model_output` 字段，然后
         `python evaluation.py --input INPUT_JSON_PATH`。

判分规则（官方 evaluation.py）：**token 集合匹配**，不是选项字母——
    条件1：答案的所有 token 都出现在预测里
    条件2：预测里不含任何「错误选项独有」的 token
因此提示词必须让模型输出**选项原文**而不是字母。

本脚本：
    prepare  下载 test-mini -> items.jsonl（官方字段 + 本地音频）
    eval     逐条理解，写 model_output
    merge    合并成官方要求的 JSON 列表（model_pred.json）
    score    复刻官方 token 匹配 + task/difficulty/sub-category 准确率

用法：
    python runs/Benchmark/MMAU/evaluate.py prepare --limit 50
    bash runs/Benchmark/run_shards.sh --benchmark MMAU
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import audio_io, generation  # noqa: E402


def build_prompt(item: dict) -> str:
    """官方判分按 token 匹配，答案里带 "(A)"，所以要求模型把选项标签一起复述。"""
    choices = item.get("choices") or []
    lines = [
        "Listen to the audio and answer the question.",
        "Reply with the option exactly as listed, including its letter, e.g. (A) Man.",
        "",
        f"Question: {item.get('question', '')}",
        "Options:",
    ]
    for choice in choices:
        lines.append(f"- {choice}")
    lines.extend(["", "Answer:"])
    return "\n".join(lines)


# ----------------------------------------------------------------------
# prepare
# ----------------------------------------------------------------------
def _to_choices(value):
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except Exception:  # noqa: BLE001
            pass
        return [v.strip() for v in re.split(r"\n|\|", value) if v.strip()]
    return []


def _write_wav_from_bytes(data: bytes, dst: Path) -> bool:
    import io

    import soundfile as sf

    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if data[:4] == b"RIFF":
            dst.write_bytes(data)
            return True
        arr, sr = sf.read(io.BytesIO(data))
        sf.write(str(dst), arr, int(sr), subtype="PCM_16")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[prepare] 写音频失败 {dst.name}: {type(e).__name__}: {e}")
        return False


def _prepare_from_parquet(args: argparse.Namespace) -> int:
    """test_mini.parquet 的字段：context{bytes,path} / instruction / choices / answer / other_attributes。"""
    import glob

    import pyarrow as pa
    import pyarrow.parquet as pq

    files = []
    for pattern in args.parquet.split(","):
        files.extend(sorted(glob.glob(pattern.strip())))
    if not files:
        raise SystemExit(f"[fatal] 没找到 parquet：{args.parquet}")

    audio_dir = Path(args.audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    want = args.limit * args.stride if args.limit else 10**9

    rows = []
    cols = ["instruction", "choices", "answer", "other_attributes", "context"]
    for path in files:
        if len(rows) >= want:
            break
        pf = pq.ParquetFile(path)
        remaining = want - len(rows)
        chunks, got = [], 0
        for rg in range(pf.num_row_groups):
            chunk = pf.read_row_group(rg, columns=cols)
            chunks.append(chunk)
            got += chunk.num_rows
            if got >= remaining:
                break
        table = pa.concat_tables(chunks)
        take = min(table.num_rows, remaining)
        print(f"[prepare] {Path(path).name}: {pf.metadata.num_rows} 行 / {pf.num_row_groups} 组，取 {take}")
        for i in range(take):
            try:
                meta = json.loads(table.column("other_attributes")[i].as_py() or "{}")
            except Exception:  # noqa: BLE001
                meta = {}
            item_id = str(meta.get("id") or f"{len(rows):05d}")
            audio = table.column("context")[i].as_py() or {}
            data = audio.get("bytes") if isinstance(audio, dict) else None
            if not data:
                print(f"[prepare] {item_id} 无音频 bytes，跳过")
                continue
            dst = audio_dir / f"{item_id}.wav"
            if not _write_wav_from_bytes(data, dst):
                continue
            rows.append({
                "id": item_id,
                "question": table.column("instruction")[i].as_py(),
                "choices": table.column("choices")[i].as_py(),
                "answer": table.column("answer")[i].as_py(),
                "task": meta.get("task", ""),
                "difficulty": meta.get("difficulty", ""),
                "sub-category": meta.get("sub-category", ""),
                "audio": str(dst),
            })

    if args.stride > 1:
        rows = rows[:: args.stride]
        if args.limit:
            rows = rows[: args.limit]

    out = Path(args.out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[prepare] {len(rows)} 条 -> {out}（来源：本地 parquet）")
    if rows:
        print(f"[prepare] 示例：task={rows[0]['task']} difficulty={rows[0]['difficulty']} "
              f"answer={rows[0]['answer']!r}")
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    if args.parquet:
        return _prepare_from_parquet(args)

    from datasets import load_dataset

    from common.datasets import _save_audio_value

    ds = None
    for split in [args.split] + [s for s in ("test", "train") if s != args.split]:
        try:
            ds = load_dataset(args.dataset, split=split)
            print(f"[prepare] {args.dataset} split={split}，共 {len(ds)} 条")
            break
        except Exception as e:  # noqa: BLE001
            print(f"[prepare] split={split} 失败：{type(e).__name__}: {e}")
    if ds is None:
        raise SystemExit(f"[fatal] 无法加载 {args.dataset}")

    audio_dir = Path(args.audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    n = len(ds) if not args.limit else min(args.limit, len(ds))

    rows = []
    for i in range(n):
        row = ds[i]
        item_id = str(row.get(args.id_field) or f"{i:05d}")
        audio_path = _save_audio_value(row.get(args.audio_field), audio_dir / f"{item_id}.wav")
        if audio_path is None:
            print(f"[prepare] 第 {i} 条无音频，跳过")
            continue
        rows.append({
            "id": item_id,
            "question": row.get(args.question_field, ""),
            "choices": _to_choices(row.get("choices", [])),
            "answer": row.get("answer", ""),
            "task": row.get("task", ""),
            "difficulty": row.get("difficulty", ""),
            "sub-category": row.get("sub-category", ""),
            "audio": audio_path,
        })

    out = Path(args.out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[prepare] {len(rows)} 条 -> {out}")
    if rows:
        print(f"[prepare] 首条：task={rows[0]['task']} choices={rows[0]['choices']} answer={rows[0]['answer']!r}")
    return 0


# ----------------------------------------------------------------------
# eval
# ----------------------------------------------------------------------
def cmd_eval(args: argparse.Namespace) -> int:
    items = audio_io.read_jsonl(args.items)
    if args.limit:
        items = items[: args.limit]
    items = items[args.shard_index :: args.num_shards]
    if not items:
        print("[skip] 本分片没有任务")
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"shard{args.shard_index}of{args.num_shards}"

    model = generation.build_model(args)
    path = out_dir / f"preds.{tag}.jsonl"
    print(f"[mmau] {len(items)} 条 | {tag}", flush=True)
    t0 = time.time()
    with path.open("w", encoding="utf-8") as f:
        for i, item in enumerate(items, 1):
            rec = dict(item)
            try:
                rec["model_output"] = model.understand(build_prompt(item), audio=item.get("audio"))
                print(f"[{i}/{len(items)}] {rec['model_output'][:60]!r}", flush=True)
            except Exception as e:  # noqa: BLE001
                rec["model_output"] = ""
                rec["error"] = f"{type(e).__name__}: {e}"
                print(f"[{i}/{len(items)}] FAIL: {e}", file=sys.stderr, flush=True)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
    print(f"[mmau] 完成 {(time.time() - t0) / 60:.1f} min -> {path}")
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    rows, seen = [], set()
    for path in sorted(out_dir.glob("preds.shard*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["id"] in seen:
                continue
            seen.add(rec["id"])
            rows.append(rec)
    rows.sort(key=lambda r: r["id"])
    merged = out_dir / "model_pred.json"
    # 官方 evaluation.py 读的是 JSON 列表，不是 jsonl
    merged.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[merge] {len(rows)} 条 -> {merged}")
    return 0


# ----------------------------------------------------------------------
# score（复刻官方 evaluation.py 的 token 匹配）
# ----------------------------------------------------------------------
def _tokenize(text: str) -> set:
    return set(re.findall(r"\b\w+\b", (text or "").lower()))


def string_match(answer: str, prediction: str, choices) -> bool:
    pred_tokens = _tokenize(prediction)
    answer_tokens = _tokenize(answer)
    if not pred_tokens:
        return False
    incorrect = set()
    for choice in choices or []:
        choice_tokens = _tokenize(choice)
        if choice_tokens != answer_tokens:
            incorrect |= choice_tokens - answer_tokens
    return answer_tokens.issubset(pred_tokens) and pred_tokens.isdisjoint(incorrect)


def score_records(records) -> dict:
    task_m = defaultdict(lambda: [0, 0])
    diff_m = defaultdict(lambda: [0, 0])
    sub_m = defaultdict(lambda: [0, 0])
    corr = total = no_pred = 0

    for rec in records:
        if "model_output" not in rec:
            continue
        if not rec["model_output"]:
            no_pred += 1
        ok = string_match(rec.get("answer", ""), rec.get("model_output", ""), rec.get("choices", []))
        task_m[rec.get("task", "")][0] += int(ok)
        task_m[rec.get("task", "")][1] += 1
        diff_m[rec.get("difficulty", "")][0] += int(ok)
        diff_m[rec.get("difficulty", "")][1] += 1
        sub = rec.get("sub-category")
        if sub is not None:
            sub_m[sub][0] += int(ok)
            sub_m[sub][1] += 1
        corr += int(ok)
        total += 1

    def as_dict(m):
        return {k: {"accuracy": (c / t if t else 0.0), "n": t} for k, (c, t) in m.items()}

    return {
        "benchmark": "MMAU",
        "overall_accuracy": corr / total if total else 0.0,
        "n": total,
        "no_prediction": no_pred,
        "task": as_dict(task_m),
        "difficulty": as_dict(diff_m),
        "sub_category": as_dict(sub_m),
    }


def cmd_score(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    path = out_dir / "model_pred.json"
    if not path.exists():
        raise SystemExit(f"[fatal] 找不到 {path}，先跑 merge")
    records = json.loads(path.read_text(encoding="utf-8"))
    result = score_records(records)
    (out_dir / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("task", "difficulty", "sub_category")},
                     ensure_ascii=False, indent=2))
    for group in ("task", "difficulty"):
        for k, v in result[group].items():
            print(f"  {group:<10s} {k:<8s} acc={v['accuracy']:.4f} n={v['n']}")
    print(f"[score] -> {out_dir / 'results.json'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--parquet", default=None,
                   help="本地 parquet（test_mini.parquet），优先于 --dataset")
    p.add_argument("--stride", type=int, default=1, help="按步长抽样")
    p.add_argument("--dataset", default="gamma-lab-umd/MMAU-test-mini")
    p.add_argument("--split", default="test")
    p.add_argument("--id-field", default="id")
    p.add_argument("--question-field", default="question")
    p.add_argument("--audio-field", default="audio")
    p.add_argument("--audio-dir", default="runs/Benchmark/_data/mmau_audio")
    p.add_argument("--out-jsonl", default="runs/Benchmark/MMAU/items.jsonl")
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(func=cmd_prepare)

    e = sub.add_parser("eval")
    generation.add_model_args(e)
    e.add_argument("--items", default="runs/Benchmark/MMAU/items.jsonl")
    e.add_argument("--out", default="runs/Benchmark/output/MMAU")
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--shard-index", type=int, default=0)
    e.add_argument("--num-shards", type=int, default=1)
    e.set_defaults(func=cmd_eval)

    m = sub.add_parser("merge")
    m.add_argument("--out", default="runs/Benchmark/output/MMAU")
    m.set_defaults(func=cmd_merge)

    s = sub.add_parser("score")
    s.add_argument("--out", default="runs/Benchmark/output/MMAU")
    s.set_defaults(func=cmd_score)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
