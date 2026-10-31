#!/usr/bin/env bash
# 评测一键入口：只指定 benchmark 名，其余全部用默认值。
#
#   bash runs/Benchmark/run_shards.sh --benchmark AudioCaps
#
# 默认行为：
#   - 模型   Audio-Omni/model/Audio-Omni.json + model.ckpt
#   - 精度   fp16；每进程 2 卡（主干 cuda:0、理解塔 cuda:1），按卡数自动分片
#   - 采样   100 步、CFG=6.0（论文设置）
#   - 输入   优先 <benchmark>/prompts.jsonl（或 items.jsonl），没有则退回 prompts.sample.jsonl
#   - 输出   runs/Benchmark/output/<benchmark>/
#   - 收尾   自动 merge；AudioEdit/TTS 自动打分，AudioCaps/MusicCaps 需 REF
#
# 覆盖方式（环境变量）：
#   GPUS_PER_PROC=1 STEPS=30 LIMIT=2 DTYPE=float32 \
#   bash runs/Benchmark/run_shards.sh --benchmark AudioCaps
#
#   REF=/data/audiocaps_test_audio   # AudioCaps/MusicCaps 打分用
#   NUM_SHARDS=4                     # 手动指定分片数
#   INPUT=/path/to/prompts.jsonl     # 手动指定输入
#
# 也兼容旧用法（直接给脚本路径 + 分片数）：
#   bash runs/Benchmark/run_shards.sh runs/Benchmark/AudioCaps/evaluate.py 8 --prompts ...
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH_ROOT="${ROOT}/runs/Benchmark"

# ---------------------------------------------------------------- 参数
if [[ "${1:-}" == "--benchmark" ]]; then
  BENCH="${2:?用法: run_shards.sh --benchmark <AudioCaps|MusicCaps|MMSU|MMAU|AudioEdit|TTS>}"
  shift 2
  EXTRA=("$@")
  SCRIPT="${BENCH_ROOT}/${BENCH}/evaluate.py"
  if [[ ! -f "${SCRIPT}" ]]; then
    echo "[fatal] 没有这个 benchmark：${BENCH}" >&2
    echo "        可用：" >&2
    for d in "${BENCH_ROOT}"/*/evaluate.py; do
      if [[ -f "$d" ]]; then
        echo "          $(basename "$(dirname "$d")")" >&2
      fi
    done
    exit 2
  fi
else
  # 旧用法：<script> <num_shards> [args...]
  SCRIPT="${1:?用法: run_shards.sh --benchmark <name>}"
  BENCH="$(basename "$(dirname "${SCRIPT}")")"
  NUM_SHARDS="${2:-1}"
  shift 2 2>/dev/null || shift 1
  EXTRA=("$@")
fi

OUT="${OUT:-${BENCH_ROOT}/output/${BENCH}}"
MODEL_CONFIG="${MODEL_CONFIG:-${ROOT}/Audio-Omni/model/Audio-Omni.json}"
CKPT="${CKPT:-${ROOT}/Audio-Omni/model/model.ckpt}"
DTYPE="${DTYPE:-float16}"
STEPS="${STEPS:-100}"
CFG="${CFG:-6.0}"
LIMIT="${LIMIT:-0}"
mkdir -p "${OUT}"

# ---------------------------------------------------------------- 卡与分片
if command -v nvidia-smi >/dev/null 2>&1; then
  NGPU=$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)
else
  NGPU=$(python -c "import torch;print(torch.cuda.device_count())" 2>/dev/null || echo 1)
fi
if (( NGPU < 1 )); then NGPU=1; fi

GPUS_PER_PROC="${GPUS_PER_PROC:-2}"
if (( GPUS_PER_PROC > NGPU )); then GPUS_PER_PROC="${NGPU}"; fi
NUM_SHARDS="${NUM_SHARDS:-$((NGPU / GPUS_PER_PROC))}"
if (( NUM_SHARDS < 1 )); then NUM_SHARDS=1; fi
COND_DEVICE="cuda:0"
if (( GPUS_PER_PROC >= 2 )); then
  COND_DEVICE="cuda:1"
else
  GPUS_PER_PROC=1
fi
GPU_IDS="${GPU_IDS:-$(seq -s, 0 $((NUM_SHARDS * GPUS_PER_PROC - 1)))}"

# ---------------------------------------------------------------- 子命令与输入
case "${BENCH}" in
  MMSU|MMAU) CMD="eval"; INPUT_FLAG="--items"; DEF_INPUT="${BENCH_ROOT}/${BENCH}/items.jsonl" ;;
  AudioEdit|TTS) CMD="gen"; INPUT_FLAG="--items"; DEF_INPUT="${BENCH_ROOT}/${BENCH}/items.jsonl" ;;
  *) CMD="gen"; INPUT_FLAG="--prompts"; DEF_INPUT="${BENCH_ROOT}/${BENCH}/prompts.jsonl" ;;
esac

INPUT="${INPUT:-${DEF_INPUT}}"
if [[ ! -f "${INPUT}" ]]; then
  SAMPLE="${BENCH_ROOT}/${BENCH}/prompts.sample.jsonl"
  if [[ -f "${SAMPLE}" ]]; then
    echo "[warn] 缺少 ${DEF_INPUT}，退回样本提示词（仅供冒烟，不能出指标）"
    echo "[warn] 正式评测先跑：python runs/Benchmark/${BENCH}/evaluate.py prepare ..."
    INPUT="${SAMPLE}"
  else
    echo "[fatal] 缺少输入：${DEF_INPUT}" >&2
    echo "        先跑 python runs/Benchmark/${BENCH}/evaluate.py prepare ..." >&2
    exit 2
  fi
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "============================================================"
echo " benchmark   : ${BENCH}  (${CMD})"
echo " input       : ${INPUT}"
echo " output      : ${OUT}"
echo " shards      : ${NUM_SHARDS} x ${GPUS_PER_PROC} GPU(s)   gpus=${GPU_IDS}"
echo " model       : ${MODEL_CONFIG}"
echo " steps/cfg   : ${STEPS} / ${CFG}   dtype=${DTYPE}  limit=${LIMIT}"
echo "============================================================"

# ---------------------------------------------------------------- 并行执行
pids=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  start=$((i * GPUS_PER_PROC))
  gpus=$(echo "${GPU_IDS}" | cut -d, -f $((start + 1))-$((start + GPUS_PER_PROC)))
  args=(
    "${CMD}"
    "${INPUT_FLAG}" "${INPUT}"
    --out "${OUT}"
    --shard-index "${i}" --num-shards "${NUM_SHARDS}"
    --model-config "${MODEL_CONFIG}" --ckpt "${CKPT}"
    --device cuda:0 --cond-device "${COND_DEVICE}"
    --dtype "${DTYPE}"
  )
  if [[ "${CMD}" == "gen" ]]; then
    args+=(--steps "${STEPS}" --cfg "${CFG}")
  fi
  if [[ "${LIMIT}" != "0" ]]; then
    args+=(--limit "${LIMIT}")
  fi

  echo "  shard ${i} -> GPUs ${gpus}"
  CUDA_VISIBLE_DEVICES="${gpus}" python "${SCRIPT}" "${args[@]}" "${EXTRA[@]}" &
  pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
echo "[run] shards finished (status=${status})"

# ---------------------------------------------------------------- 合并与打分
if (( status == 0 )); then
  python "${SCRIPT}" merge --out "${OUT}" || echo "[warn] merge 失败，忽略"
  case "${BENCH}" in
    AudioEdit|TTS)
      python "${SCRIPT}" score --out "${OUT}" || echo "[warn] score 失败"
      ;;
    AudioCaps|MusicCaps)
      if [[ -n "${REF:-}" ]]; then
        python "${SCRIPT}" score --out "${OUT}" --ref "${REF}" || echo "[warn] score 失败"
      else
        echo "[hint] 设置 REF=/path/to/reference_audio 可自动打分（FAD/CLAP）"
      fi
      ;;
  esac
fi

echo "[done] 结果在 ${OUT}"
exit "${status}"
