#!/usr/bin/env bash
# 4 进程 × 2 卡（DiT / 理解塔分卡）训练
#
#   bash runs/scripts/train_reca_ddp.sh            # 2000 步，累积 1
#   SMOKE=1 bash runs/scripts/train_reca_ddp.sh    # 2 步
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
ROOT="$(pwd)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate audio-omni || exit 1

export QWEN_OMNI_MODEL_PATH="${QWEN_OMNI_MODEL_PATH:-$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-3B/snapshots/*/ 2>/dev/null | head -1)}"
export PYTHONPATH="${ROOT}/src:${ROOT}/Audio-Omni:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

STEPS="${STEPS:-2000}"
ACCUM="${ACCUM:-1}"
SECONDS_PER_CLIP="${SECONDS_PER_CLIP:-5}"
OUT="${OUT:-runs/train/reca_ddp}"
PROCS="${PROCS:-4}"
PORT="${PORT:-29511}"
EXTRA=""

if [[ "${SMOKE:-0}" == "1" ]]; then
  STEPS=2
  OUT="${OUT}_smoke"
  EXTRA="--limit 8 --num-workers 0 --log-every 1 --save-every 0"
  echo "[smoke] 2 步 / 4 进程"
fi

echo "[ddp] procs=${PROCS} steps=${STEPS} accum=${ACCUM} seconds=${SECONDS_PER_CLIP} out=${OUT}"

torchrun --nproc_per_node="${PROCS}" --master_port="${PORT}" \
  -m reca_audio.train.train_ddp \
  --mode live \
  --manifest "${MANIFEST:-runs/Benchmark/_data/audiocaps/train_manifest.jsonl}" \
  --out "${OUT}" \
  --steps "${STEPS}" \
  --grad-accum "${ACCUM}" \
  --seconds "${SECONDS_PER_CLIP}" \
  --gpus-per-rank 2 \
  --amp \
  --dtype "${DTYPE:-float16}" \
  --save-every 200 \
  ${EXTRA}
