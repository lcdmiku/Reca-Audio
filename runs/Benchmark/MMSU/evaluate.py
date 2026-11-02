#!/usr/bin/env python
"""MMSU — 官方协议版评测。

官方仓库：https://github.com/dingdongwang/MMSU
数据：HF `ddwang2000/MMSU`（5000 条 audio-question-answer，47 个子任务）
官方流程：`mmsu_inference.py`（需自己接模型）产出 `model_pred.jsonl`，
         再 `mmsu_evaluation.py` 打分。

本脚本做同样的事，但把「接模型」这一步接到了 Audio-Omni：

    prepare  下载 HF 数据 -> items.jsonl（官方字段 + 本地音频路径）
    eval     逐条调用理解模型，把回复写进 response，输出官方格式 jsonl
    merge    合并分片
    score    复刻官方判分（category / sub-category / overall 准确率）

字段（与官方 mmsu_evaluation.py 一致）：
    category, sub-category, answer_gt, choice_a..choice_d, response

用法：
    python runs/Benchmark/MMSU/evaluate.py prepare --limit 50
    bash runs/Benchmark/run_shards.sh --benchmark MMSU
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import audio_io, generation  # noqa: E402

LETTERS = ("A", "B", "C", "D")


def build_prompt(item: dict) -> str:
    """官方 inference 脚本的提示词未随仓库给出，这里用标准四选一格式。

    判分只读 `response`；换提示词会影响分数，报告里要注明用的是哪一版。
    """
    lines = [
        "Listen to the audio and answer the multiple-choice question.",
        "Reply with only the option letter (A, B, C or D).",
        "",
        f"Question: {item.get('question', '')}",
    ]
    for letter, key in zip(LETTERS, ("choice_a", "choice_b", "choice_c", "choice_d")):
        text = item.get(key)
        if text:
            lines.append(f"{letter}. {text}")
    lines.extend(["", "Answer:"])
    return "\n".join(lines)


# ----------------------------------------------------------------------
# prepare
# ----------------------------------------------------------------------
def _write_wav_from_bytes(data: bytes, dst: Path) -> bool:
    """parquet 里的 audio 是 struct{bytes,path}；bytes 既可能是原始 wav，也可能是编码音频。"""
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
    """直接读本地 parquet，比走 datasets 的网络流式加载快得多。"""
    import glob

    import pyarrow.parquet as pq

    files = []
    for pattern in args.parquet.split(","):
        files.extend(sorted(glob.glob(pattern.strip())))
    if not files:
        raise SystemExit(f"[fatal] 没找到 parquet：{args.parquet}")

    audio_dir = Path(args.audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    want = args.limit if args.limit else 10**9

    rows = []
    for path in files:
        if len(rows) >= want:
            break
        pf = pq.ParquetFile(path)
        cols = [args.id_field, args.question_field, "choice_a", "choice_b", "choice_c",
                "choice_d", args.answer_field, "category", "sub-category", args.audio_field]
        # 一个 shard 可能有多个 row group（实测每组 100 行），逐个读直到凑够
        import pyarrow as pa

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
            rec = {c: table.column(c)[i].as_py() for c in cols}
            item_id = str(rec.get(args.id_field) or f"{len(rows):05d}")
            audio = rec.get(args.audio_field) or {}
            data = audio.get("bytes") if isinstance(audio, dict) else None
            if not data:
                print(f"[prepare] {item_id} 无音频 bytes，跳过")
                continue
            dst = audio_dir / f"{item_id}.wav"
            if not _write_wav_from_bytes(data, dst):
                continue
            rows.append({
                "id": item_id,
                "category": rec.get("category", ""),
                "sub-category": rec.get("sub-category", ""),
                "question": rec.get(args.question_field, ""),
                "choice_a": rec.get("choice_a", ""),
                "choice_b": rec.get("choice_b", ""),
                "choice_c": rec.get("choice_c", ""),
                "choice_d": rec.get("choice_d", ""),
                "answer_gt": rec.get(args.answer_field, ""),
                "audio": str(dst),
            })

    out = Path(args.out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[prepare] {len(rows)} 条 -> {out}（来源：本地 parquet）")
    if rows:
        print(f"[prepare] 示例：{rows[0]['category']} / {rows[0]['question'][:70]!r}")
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    if args.parquet:
        return _prepare_from_parquet(args)

    from datasets import load_dataset

    from common.datasets import _save_audio_value

    ds, used_split = None, None
    for split in [args.split] + [s for s in ("test", "train", "validation") if s != args.split]:
        try:
            ds = load_dataset(args.dataset, split=split)
            used_split = split
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
            "category": row.get("category", ""),
            "sub-category": row.get("sub-category", ""),
            "question": row.get(args.question_field, ""),
            "choice_a": row.get("choice_a", ""),
            "choice_b": row.get("choice_b", ""),
            "choice_c": row.get("choice_c", ""),
            "choice_d": row.get("choice_d", ""),
            "answer_gt": row.get(args.answer_field, ""),
            "audio": audio_path,
        })

    out = Path(args.out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[prepare] {len(rows)} 条 -> {out}（split={used_split}）")
    if rows:
        print(f"[prepare] 首条字段：{list(rows[0].keys())}")
        print(f"[prepare] 示例：{rows[0]['question'][:80]!r} / 答案 {rows[0]['answer_gt']!r}")
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
    print(f"[mmsu] {len(items)} 条 | {tag}", flush=True)
    t0 = time.time()
    with path.open("w", encoding="utf-8") as f:
        for i, item in enumerate(items, 1):
            rec = dict(item)
            try:
                rec["response"] = model.understand(build_prompt(item), audio=item.get("audio"))
                print(f"[{i}/{len(items)}] {rec['response'][:60]!r}", flush=True)
            except Exception as e:  # noqa: BLE001
                rec["response"] = ""
                rec["error"] = f"{type(e).__name__}: {e}"
                print(f"[{i}/{len(items)}] FAIL: {e}", file=sys.stderr, flush=True)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
    print(f"[mmsu] 完成 {(time.time() - t0) / 60:.1f} min -> {path}")
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
    merged = out_dir / "model_pred.jsonl"
    with merged.open("w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[merge] {len(rows)} 条 -> {merged}")
    return 0


# ----------------------------------------------------------------------
# score（复刻官方 mmsu_evaluation.py）
# ----------------------------------------------------------------------
def extract_letter(response: str):
    """官方规则：首字符是 A-D 就取它；否则看倒数第二个字符；都不行返回 None。"""
    text = (response or "").strip().replace("\n", "")
    if not text or text == "None":
        return None
    if text[0] in LETTERS:
        return text[0]
    if len(text) > 1 and text[-2] in LETTERS:
        return text[-2]
    return None


def score_records(records) -> dict:
    per_task_cat = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
    total_correct = total_count = 0
    skipped = 0

    for rec in records:
        letter = extract_letter(rec.get("response", ""))
        if letter is None:
            skipped += 1
            continue  # 与官方一致：无法解析的回复不计入分母
        task = rec.get("category", "")
        sub = rec.get("sub-category", "")
        if rec.get(f"choice_{letter.lower()}", None) == rec.get("answer_gt", None):
            per_task_cat[task][sub]["correct"] += 1
            total_correct += 1
        per_task_cat[task][sub]["total"] += 1
        total_count += 1

    per_task = {}
    for task, subs in per_task_cat.items():
        c = sum(v["correct"] for v in subs.values())
        t = sum(v["total"] for v in subs.values())
        per_task[task] = {
            "accuracy": c / t if t else 0.0,
            "n": t,
            "sub": {k: (v["correct"] / v["total"] if v["total"] else 0.0) for k, v in subs.items()},
        }

    return {
        "benchmark": "MMSU",
        "overall_accuracy": total_correct / total_count if total_count else 0.0,
        "n_scored": total_count,
        "n_skipped_unparseable": skipped,
        "per_task": per_task,
    }


def cmd_score(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    path = out_dir / "model_pred.jsonl"
    if not path.exists():
        raise SystemExit(f"[fatal] 找不到 {path}，先跑 merge")
    result = score_records(audio_io.read_jsonl(str(path)))
    (out_dir / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_task"}, ensure_ascii=False, indent=2))
    for task, v in sorted(result["per_task"].items(), key=lambda kv: -kv[1]["n"]):
        print(f"  {task:<40s} acc={v['accuracy']:.4f}  n={v['n']}")
    print(f"[score] -> {out_dir / 'results.json'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--parquet", default=None,
                   help="本地 parquet（可逗号分隔/通配），优先于 --dataset；音频内嵌在 bytes 里")
    p.add_argument("--dataset", default="ddwang2000/MMSU")
    p.add_argument("--split", default="test")
    p.add_argument("--id-field", default="id")
    p.add_argument("--question-field", default="question")
    p.add_argument("--answer-field", default="answer_gt")
    p.add_argument("--audio-field", default="audio")
    p.add_argument("--audio-dir", default="runs/Benchmark/_data/mmsu_audio")
    p.add_argument("--out-jsonl", default="runs/Benchmark/MMSU/items.jsonl")
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(func=cmd_prepare)

    e = sub.add_parser("eval")
    generation.add_model_args(e)
    e.add_argument("--items", default="runs/Benchmark/MMSU/items.jsonl")
    e.add_argument("--out", default="runs/Benchmark/output/MMSU")
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--shard-index", type=int, default=0)
    e.add_argument("--num-shards", type=int, default=1)
    e.set_defaults(func=cmd_eval)

    m = sub.add_parser("merge")
    m.add_argument("--out", default="runs/Benchmark/output/MMSU")
    m.set_defaults(func=cmd_merge)

    s = sub.add_parser("score")
    s.add_argument("--out", default="runs/Benchmark/output/MMSU")
    s.set_defaults(func=cmd_score)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
