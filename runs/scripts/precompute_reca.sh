#!/usr/bin/env bash
# 预计算 ReCA 条件 + latent（每卡一个分片）
#
#   bash runs/scripts/precompute_reca.sh            # 8 卡
#   NUM_SHARDS=2 bash runs/scripts/precompute_reca.sh
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
ROOT="$(pwd)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate audio-omni || exit 1

export QWEN_OMNI_MODEL_PATH="${QWEN_OMNI_MODEL_PATH:-$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-3B/snapshots/*/ 2>/dev/null | head -1)}"
export PYTHONPATH="${ROOT}/src:${ROOT}/Audio-Omni:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MANIFEST="${MANIFEST:-runs/Benchmark/_data/audiocaps/train_manifest.jsonl}"
OUT="${OUT:-runs/train/cache_reca}"
NUM_SHARDS="${NUM_SHARDS:-8}"
LIMIT="${LIMIT:-0}"

mkdir -p "${OUT}"
pids=()
for ((i=0; i<NUM_SHARDS; i++)); do
  CUDA_VISIBLE_DEVICES="${i}" python -m reca_audio.train.precompute \
    --manifest "${MANIFEST}" --out "${OUT}" \
    --shard-index "${i}" --num-shards "${NUM_SHARDS}" \
    --device cuda:0 --limit "${LIMIT}" \
    > "${OUT}/precompute_${i}.log" 2>&1 &
  pids+=($!)
done

for pid in "${pids[@]}"; do wait "${pid}"; done
echo "[precompute] all shards done -> ${OUT}"
ls "${OUT}"/*.pt 2>/dev/null | wc -l
