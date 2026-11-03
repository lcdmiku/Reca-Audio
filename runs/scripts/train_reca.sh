#!/usr/bin/env bash
# ReCA 后训练启动脚本（服务器）
#
#   bash runs/scripts/train_reca.sh                 # 正式：RecA，2000 步
#   SMOKE=1 bash runs/scripts/train_reca.sh         # 冒烟：2 步、2 条、5 s
#   TASK_MODE=mix LAMBDA_T2A=0.2 bash runs/scripts/train_reca.sh
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
ROOT="$(pwd)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate audio-omni || exit 1

export QWEN_OMNI_MODEL_PATH="${QWEN_OMNI_MODEL_PATH:-$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-3B/snapshots/*/ 2>/dev/null | head -1)}"
export PYTHONPATH="${ROOT}/src:${ROOT}/Audio-Omni:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

AC_ROOT="runs/Benchmark/_data/audiocaps"
MANIFEST="${MANIFEST:-${AC_ROOT}/train_manifest.jsonl}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "[prep] 构建 manifest"
  python -m reca_audio.train.make_manifest \
    --prompts "${AC_ROOT}/full2/prompts.jsonl" \
    --audio-dir "${AC_ROOT}/full2/ref" \
    --out "${MANIFEST}" || exit 1
fi

STEPS="${STEPS:-2000}"
SECONDS_PER_CLIP="${SECONDS_PER_CLIP:-11}"
BATCH="${BATCH:-1}"
ACCUM="${ACCUM:-4}"
LR="${LR:-1e-5}"
TASK_MODE="${TASK_MODE:-reca}"
LAMBDA_T2A="${LAMBDA_T2A:-0.0}"
OUT="${OUT:-runs/train/reca_v1}"
AMP_FLAG="${AMP_FLAG:---amp}"
OPTIM="${OPTIM:-adafactor}"
VAE_DEVICE="${VAE_DEVICE:-}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  STEPS=2
  SECONDS_PER_CLIP=5
  BATCH=1
  ACCUM=1
  OUT="${OUT}_smoke"
  EXTRA="--limit 2 --num-workers 0"
  echo "[smoke] 2 步 / 2 条 / 5 s"
fi

echo "[train] manifest=${MANIFEST}"
echo "[train] steps=${STEPS} mode=${TASK_MODE} lambda_t2a=${LAMBDA_T2A} out=${OUT}"

python -m reca_audio.train.train \
  --manifest "${MANIFEST}" \
  --out "${OUT}" \
  --model-config "${ROOT}/Audio-Omni/model/Audio-Omni.json" \
  --ckpt "${ROOT}/Audio-Omni/model/model.ckpt" \
  --steps "${STEPS}" \
  --seconds "${SECONDS_PER_CLIP}" \
  --batch-size "${BATCH}" \
  --grad-accum "${ACCUM}" \
  --lr "${LR}" \
  --task-mode "${TASK_MODE}" \
  --lambda-t2a "${LAMBDA_T2A}" \
  --device "${DEVICE:-cuda:0}" \
  --conditioner-device "${COND_DEVICE:-cuda:1}" \
  --optim "${OPTIM}" \
  ${VAE_DEVICE:+--vae-device "${VAE_DEVICE}"} \
  ${AMP_FLAG} ${EXTRA:-}
