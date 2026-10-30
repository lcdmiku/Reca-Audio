#!/usr/bin/env bash
# 多卡吞吐模式：每个 GPU 起一个进程，各自加载一份模型，提示词按片分发。
#
# 要求：单卡能放下 完整模型（Qwen fp32 12 GB + DiT ~8 GB + VAE）。
# 若单卡放不下，改用组件并行（两个进程共享两张卡见 README）：
#   CUDA_VISIBLE_DEVICES=0,1 python runs/scripts/t2a_generate.py ... --cond-device cuda:1
#
# 用法：
#   NUM_SHARDS=8 bash runs/scripts/run_t2a_shards.sh \
#     --config Audio-Omni/model/Audio-Omni.json \
#     --ckpt   Audio-Omni/model/model.ckpt \
#     --prompts prompts.txt --out runs/samples/t2a --steps 100
set -euo pipefail

NUM_SHARDS="${NUM_SHARDS:-8}"
GPU_IDS="${GPU_IDS:-}"

if [[ -z "${GPU_IDS}" ]]; then
  GPU_IDS=$(seq -s, 0 $((NUM_SHARDS - 1)))
fi

echo "[info] shards=${NUM_SHARDS} gpus=${GPU_IDS}"

pids=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu=$(echo "${GPU_IDS}" | cut -d, -f $((i + 1)))
  CUDA_VISIBLE_DEVICES="${gpu}" python runs/scripts/t2a_generate.py \
    --device cuda:0 \
    --shard-index "${i}" --num-shards "${NUM_SHARDS}" \
    "$@" &
  pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

echo "[done] all shards finished (status=${status})"
exit "${status}"
