"""理解类 benchmark（MMSU / MMAU）的共用流程。

流程：读统一 jsonl -> 逐条调用 `AudioOmni.understand()` -> 存预测 -> 算准确率。
注意：这里只做“参考答案匹配”的自动判分；论文若用 LLM judge，需要另接判分器。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import List

from .audio_io import write_jsonl


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--items", required=True, help="统一 jsonl：id/question/answer/choices?/audio?")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--model-config", default="Audio-Omni/model/Audio-Omni.json")
    ap.add_argument("--ckpt", default="Audio-Omni/model/model.ckpt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cond-device", default="cuda:1")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])


def _normalize(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[\s\u3000]+", " ", text)
    text = re.sub(r"[^\w\s\u4e00-\u9fff]", "", text)
    return text


def is_correct(pred: str, answer: str, choices: List[str] | None = None) -> bool:
    """宽松匹配：完全一致 / 参考答案包含在预测里 / 选项字母一致。"""
    p, a = _normalize(pred), _normalize(answer)
    if not a:
        return False
    if p == a or a in p:
        return True
    # 选项题：模型可能只回字母
    if choices:
        for i, choice in enumerate(choices):
            letter = chr(ord("A") + i)
            if _normalize(choice) == a and p.strip() == letter.lower():
                return True
    return False


def run(args: argparse.Namespace, benchmark: str) -> int:
    rows = []
    for line in Path(args.items).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]
    rows = rows[args.shard_index :: args.num_shards]
    if not rows:
        print("[skip] 本分片没有任务")
        return 0

    root = Path(__file__).resolve().parents[3]
    pkg = root / "Audio-Omni"
    if pkg.is_dir() and str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))
    from audio_omni.api import AudioOmni

    print(f"[{benchmark}] {len(rows)} 条 | shard {args.shard_index}/{args.num_shards}", flush=True)
    model = AudioOmni(
        args.model_config,
        args.ckpt,
        device=args.device,
        conditioner_device=args.cond_device,
        dtype=args.dtype,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"shard{args.shard_index}of{args.num_shards}"
    preds = []
    t0 = time.time()
    for i, item in enumerate(rows, 1):
        rec = {"id": item.get("id", f"{i:05d}"), "question": item.get("question", ""),
               "answer": item.get("answer", ""), "choices": item.get("choices")}
        t_start = time.time()
        try:
            pred = model.understand(rec["question"], audio=item.get("audio"))
            rec["pred"] = pred
            rec["correct"] = is_correct(pred, rec["answer"], rec["choices"])
            print(f"[{i}/{len(rows)}] {time.time() - t_start:.1f}s correct={rec['correct']}", flush=True)
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
            print(f"[{i}/{len(rows)}] FAIL: {e}", file=sys.stderr, flush=True)
        preds.append(rec)

    write_jsonl(str(out_dir / f"preds.{tag}.jsonl"), preds)
    ok = [p for p in preds if "correct" in p]
    acc = sum(p["correct"] for p in ok) / len(ok) if ok else float("nan")
    summary = {"benchmark": benchmark, "shard": tag, "n": len(preds), "n_ok": len(ok),
               "accuracy": acc, "minutes": round((time.time() - t0) / 60, 1)}
    (out_dir / f"summary.{tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def merge_and_report(out_dir: str, benchmark: str) -> int:
    """合并所有分片的 preds 并重算总准确率。"""
    root = Path(out_dir)
    preds, seen = [], set()
    for path in sorted(root.glob("preds.shard*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["id"] in seen:
                continue
            seen.add(rec["id"])
            preds.append(rec)
    ok = [p for p in preds if "correct" in p]
    acc = sum(p["correct"] for p in ok) / len(ok) if ok else float("nan")
    result = {"benchmark": benchmark, "n": len(preds), "n_ok": len(ok), "accuracy": acc}
    (root / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0
