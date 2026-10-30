#!/usr/bin/env bash
# 通用多卡启动器：把 <script> 的 gen 阶段按分片铺到多张卡上。
#
# 用法：
#   bash runs/Benchmark/run_shards.sh <script> <num_shards> [gen 阶段的其它参数...]
#
# 例：
#   bash runs/Benchmark/run_shards.sh runs/Benchmark/AudioCaps/evaluate.py 8 \
#     --prompts runs/Benchmark/AudioCaps/prompts.jsonl \
#     --out runs/Benchmark/_runs/audiocaps \
#     --model-config Audio-Omni/model/Audio-Omni.json --ckpt Audio-Omni/model/model.ckpt \
#     --cond-device cuda:1 --dtype float16
set -euo pipefail

SCRIPT="${1:?用法: run_shards.sh <script> <num_shards> [args...]}"
NUM_SHARDS="${2:?缺少 num_shards}"
shift 2

GPU_IDS="${GPU_IDS:-}"
if [[ -z "${GPU_IDS}" ]]; then
  GPU_IDS=$(seq -s, 0 $((NUM_SHARDS - 1)))
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "[run] script=${SCRIPT} shards=${NUM_SHARDS} gpus=${GPU_IDS}"

pids=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu=$(echo "${GPU_IDS}" | cut -d, -f $((i + 1)))
  # 每个进程只看到属于自己的那张卡，脚本内部的 cuda:0 即该卡
  CUDA_VISIBLE_DEVICES="${gpu}" python "${SCRIPT}" gen \
    --shard-index "${i}" --num-shards "${NUM_SHARDS}" \
    "$@" &
  pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
echo "[done] status=${status}"
exit "${status}"
