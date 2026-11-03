"""把评测用的 prompts.jsonl（{id, prompt}）+ 音频目录合成训练 manifest。

    python -m reca_audio.train.make_manifest \
        --prompts runs/Benchmark/_data/audiocaps/full2/prompts.jsonl \
        --audio-dir runs/Benchmark/_data/audiocaps/full2/ref \
        --out runs/Benchmark/_data/audiocaps/train_manifest.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    audio_dir = Path(args.audio_dir)
    rows = []
    with open(args.prompts, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            wav = audio_dir / f"{row['id']}.wav"
            if not wav.exists():
                continue
            rows.append({"audio": str(wav), "caption": row.get("prompt", "")})
            if args.limit and len(rows) >= args.limit:
                break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[manifest] {len(rows)} 条 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
