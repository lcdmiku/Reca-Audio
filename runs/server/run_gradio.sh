#!/usr/bin/env bash
# 一键启动 Gradio（避免手写多行命令时漏续行符导致参数没传进去）。
#
# 用法：
#   bash runs/server/run_gradio.sh                 # 默认双卡 + fp16
#   COND_DEVICE=cuda:0 bash runs/server/run_gradio.sh   # 单卡
#   bash runs/server/run_gradio.sh --share         # 额外参数原样透传
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}/Audio-Omni"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[run] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[run] cond=${COND_DEVICE:-cuda:1} dtype=${DTYPE:-float16} port=${PORT:-7777}"

exec python3 run_gradio.py \
  --server-port "${PORT:-7777}" \
  --model-config "${MODEL_CONFIG:-model/Audio-Omni.json}" \
  --ckpt-path "${CKPT:-model/model.ckpt}" \
  --cond-device "${COND_DEVICE:-cuda:1}" \
  --dtype "${DTYPE:-float16}" \
  "$@"
