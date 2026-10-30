#!/usr/bin/env bash
# 通用多卡启动器：把 <script> 的 gen 阶段按分片铺到多张卡上。
#
# 两种模式：
#   GPUS_PER_PROC=1（默认，吞吐优先）  每进程 1 卡，整模型压在一张卡上
#                                        fp16 约 22–25 GB，32 GB 卡勉强够，fp32 会 OOM
#   GPUS_PER_PROC=2（显存优先）        每进程 2 卡：主干在 cuda:0、理解塔在 cuda:1
#                                        约 8 GB + 12 GB，安全
#
# 用法：
#   bash runs/Benchmark/run_shards.sh <script> <num_shards> [gen 阶段的其它参数...]
#
# 例（8 卡 → 4 片 × 2 卡）：
#   GPUS_PER_PROC=2 bash runs/Benchmark/run_shards.sh runs/Benchmark/AudioCaps/evaluate.py 4 \
#     --prompts runs/Benchmark/AudioCaps/prompts.jsonl \
#     --out runs/Benchmark/_runs/audiocaps \
#     --model-config Audio-Omni/model/Audio-Omni.json --ckpt Audio-Omni/model/model.ckpt \
#     --dtype float16
set -euo pipefail

SCRIPT="${1:?用法: run_shards.sh <script> <num_shards> [args...]}"
NUM_SHARDS="${2:?缺少 num_shards}"
shift 2

GPUS_PER_PROC="${GPUS_PER_PROC:-1}"
NEEDED=$((NUM_SHARDS * GPUS_PER_PROC))

GPU_IDS="${GPU_IDS:-}"
if [[ -z "${GPU_IDS}" ]]; then
  GPU_IDS=$(seq -s, 0 $((NEEDED - 1)))
fi

AVAILABLE=$(echo "${GPU_IDS}" | tr ',' '\n' | wc -l)
if (( AVAILABLE < NEEDED )); then
  echo "[fatal] 需要 ${NEEDED} 张卡（${NUM_SHARDS} 片 × ${GPUS_PER_PROC}），但只给了 ${AVAILABLE} 张：${GPU_IDS}" >&2
  exit 2
fi

if (( GPUS_PER_PROC == 1 )); then
  echo "[warn] 每进程只有 1 张卡：整模型会压在同一张卡上，fp32 必 OOM；显存不够请用 GPUS_PER_PROC=2"
  COND_DEVICE="cuda:0"
else
  COND_DEVICE="cuda:1"
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "[run] script=${SCRIPT} shards=${NUM_SHARDS} gpus_per_proc=${GPUS_PER_PROC} gpus=${GPU_IDS} cond=${COND_DEVICE}"

pids=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  start=$((i * GPUS_PER_PROC))
  gpus=$(echo "${GPU_IDS}" | cut -d, -f $((start + 1))-$((start + GPUS_PER_PROC)))
  # 参数放在最后，确保 --device/--cond-device 不被前面的同名参数覆盖
  CUDA_VISIBLE_DEVICES="${gpus}" python "${SCRIPT}" gen \
    --shard-index "${i}" --num-shards "${NUM_SHARDS}" \
    "$@" \
    --device cuda:0 --cond-device "${COND_DEVICE}" &
  pids+=($!)
  echo "  shard ${i} -> GPUs ${gpus}"
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
echo "[done] status=${status}"
exit "${status}"
