#!/usr/bin/env bash
# Equal-exposure ReCA gradient A/B launcher. Formal runs require ALLOW_FORMAL=1.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
ROOT_DIR="$(pwd)"
ACTION="${1:-status}"
OUT_ROOT="${OUT_ROOT:-runs/train/reca_gradient_ab_smoke}"
SPLIT_ROOT="${SPLIT_ROOT:-runs/Benchmark/_data/audiocaps/reca_split_v1}"
MANIFEST="${MANIFEST:-${SPLIT_ROOT}/train_manifest.jsonl}"
DIAGNOSTIC_MANIFEST="${DIAGNOSTIC_MANIFEST:-${SPLIT_ROOT}/test_manifest.jsonl}"
SCHEDULE="${OUT_ROOT}/schedule.jsonl"
GPUS="${GPUS:-0,1}"
CKPT="${CKPT:-Audio-Omni/model/model.ckpt}"

validate_gpus() {
  local gpu used threshold="${GPU_MEMORY_LIMIT_MIB:-128}"
  command -v nvidia-smi >/dev/null || {
    echo "nvidia-smi is unavailable" >&2
    return 1
  }
  for gpu in ${GPUS//,/ }; do
    used="$(nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')"
    if [[ ! "${used}" =~ ^[0-9]+$ ]]; then
      echo "cannot read GPU ${gpu} memory usage: ${used}" >&2
      return 1
    fi
    if (( used > threshold )); then
      echo "GPU ${gpu} is not free: ${used} MiB used (limit ${threshold} MiB)" >&2
      return 1
    fi
  done
}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-audio-omni}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/Audio-Omni:${PYTHONPATH:-}"
export QWEN_OMNI_MODEL_PATH="${QWEN_OMNI_MODEL_PATH:-$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-3B/snapshots/*/ | head -1)}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

case "${ACTION}" in
  smoke)
    TOTAL=4
    WARMUP=1
    DIAGNOSTIC_SAMPLES=1
    DIAGNOSTIC_EXPOSURES="0,4"
    SAVE_EXPOSURES="4"
    FORMAL_FLAG=()
    ;;
  static)
    TOTAL=400
    WARMUP=20
    DIAGNOSTIC_SAMPLES="${DIAGNOSTIC_SAMPLES:-8}"
    DIAGNOSTIC_EXPOSURES="0"
    SAVE_EXPOSURES=""
    FORMAL_FLAG=()
    ;;
  run)
    if [[ "${ALLOW_FORMAL:-0}" != "1" ]]; then
      echo "refusing formal A/B without ALLOW_FORMAL=1" >&2
      exit 2
    fi
    TOTAL=400
    WARMUP=20
    DIAGNOSTIC_SAMPLES="${DIAGNOSTIC_SAMPLES:-8}"
    DIAGNOSTIC_EXPOSURES="0,100,200,400"
    SAVE_EXPOSURES="100,200,400"
    FORMAL_FLAG=(--allow-formal)
    ;;
  status)
    find "${OUT_ROOT}" -maxdepth 2 -type f \( -name 'metrics.jsonl' -o -name '*.log' \) -print 2>/dev/null || true
    exit 0
    ;;
  *)
    echo "usage: $0 {smoke|static|run|status}" >&2
    exit 2
    ;;
esac

validate_gpus

if [[ -e "${OUT_ROOT}" ]]; then
  echo "refusing existing output root: ${OUT_ROOT}" >&2
  exit 2
fi
mkdir -p "${OUT_ROOT}"

python -m reca_audio.train.run_gradient_ab \
  --arm a \
  --manifest "${MANIFEST}" \
  --schedule "${SCHEDULE}" \
  --total-exposures "${TOTAL}" \
  --warmup-exposures "${WARMUP}" \
  --prepare-only

common=(
  --manifest "${MANIFEST}"
  --diagnostic-manifest "${DIAGNOSTIC_MANIFEST}"
  --schedule "${SCHEDULE}"
  --total-exposures "${TOTAL}"
  --warmup-exposures "${WARMUP}"
  --peak-lr 2e-5
  --diagnostic-samples "${DIAGNOSTIC_SAMPLES}"
  --diagnostic-exposures "${DIAGNOSTIC_EXPOSURES}"
  --save-exposures "${SAVE_EXPOSURES}"
  --model-config "${ROOT_DIR}/Audio-Omni/model/Audio-Omni.json"
  --ckpt "${CKPT}"
  --seconds 5
  --device cuda:0
  --conditioner-device cuda:1
  --vae-device cpu
  --dtype float32
  --amp
  --omni-downsample 4
  --omni-seconds 0
  "${FORMAL_FLAG[@]}"
)

if [[ "${ACTION}" == "static" ]]; then
  CUDA_VISIBLE_DEVICES="${GPUS}" python -m reca_audio.train.run_gradient_ab \
    --arm static --out "${OUT_ROOT}/static" "${common[@]}" \
    >"${OUT_ROOT}/static.log" 2>&1
  exit 0
fi

CUDA_VISIBLE_DEVICES="${GPUS}" python -m reca_audio.train.run_gradient_ab \
  --arm a --out "${OUT_ROOT}/a" "${common[@]}" \
  >"${OUT_ROOT}/a.log" 2>&1

CUDA_VISIBLE_DEVICES="${GPUS}" python -m reca_audio.train.run_gradient_ab \
  --arm b --out "${OUT_ROOT}/b" "${common[@]}" \
  >"${OUT_ROOT}/b.log" 2>&1

echo "completed ${ACTION}: ${OUT_ROOT}"
