#!/usr/bin/env bash
# Fixed 24-clip reconstruction diagnostic for the exact-B long matrix.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
ROOT_DIR="$(pwd)"
ACTION="${1:-status}"
OUT_ROOT="${OUT_ROOT:-runs/Benchmark/output/Reconstruction_LongMatrixB_v1}"
BASE_RECON="${BASE_RECON:-runs/Benchmark/output/Reconstruction_v4_6gpu_9ce3984}"
BASE_CKPT="${BASE_CKPT:-Audio-Omni/model/model.ckpt}"
GPU_PAIR="${GPU_PAIR:-1,2}"

dit_path() {
  case "$1" in
    pure) echo "runs/train/reca_long_matrix_b_v1/pure/dit_exposure2000.pt" ;;
    t2a) echo "runs/train/reca_long_matrix_b_v1/t2a/dit_exposure2000.pt" ;;
    mix) echo "runs/train/reca_long_matrix_b_v1/mix/dit_exposure2000.pt" ;;
    *) return 2 ;;
  esac
}

if [[ "${ACTION}" == "status" ]]; then
  test -s "${OUT_ROOT}/results.json" && cat "${OUT_ROOT}/results.json" || echo "PENDING ${OUT_ROOT}"
  exit 0
fi
[[ "${ACTION}" == "run" ]] || { echo "usage: $0 {run|status}" >&2; exit 2; }
[[ ! -e "${OUT_ROOT}" ]] || { echo "refusing existing output root: ${OUT_ROOT}" >&2; exit 2; }
[[ -s "${BASE_RECON}/COMPLETE" ]] || { echo "missing prior reconstruction baseline" >&2; exit 1; }
[[ -s "${BASE_RECON}/items.json" ]] || { echo "missing reconstruction items" >&2; exit 1; }
[[ -d "${BASE_RECON}/base" ]] || { echo "missing reconstruction base audio" >&2; exit 1; }
for arm in pure t2a mix; do
  [[ -s "$(dit_path "${arm}")" ]] || { echo "missing DiT checkpoint: ${arm}" >&2; exit 1; }
done

IFS=, read -r gpu0 gpu1 <<<"${GPU_PAIR}"
for gpu in "${gpu0}" "${gpu1}"; do
  used="$(nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')"
  (( used <= ${GPU_MEMORY_LIMIT_MIB:-128} )) || { echo "GPU ${gpu} is occupied: ${used} MiB" >&2; exit 1; }
done

source "${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-audio-omni}"
export QWEN_OMNI_MODEL_PATH="${QWEN_OMNI_MODEL_PATH:-$(find "${HOME}/.cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-3B/snapshots" -mindepth 1 -maxdepth 1 -type d -print -quit)}"
export CLAP_CKPT="${CLAP_CKPT:-${HOME}/.cache/laion_clap/630k-audioset-best.pt}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/Audio-Omni:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TMP_ROOT="$(mktemp -d /tmp/reca-long-matrix-b-recon.XXXXXX)"
cleanup() {
  [[ -n "${TMP_ROOT:-}" ]] && [[ "${TMP_ROOT}" == /tmp/reca-long-matrix-b-recon.* ]] && rm -rf -- "${TMP_ROOT}"
}
trap cleanup EXIT INT TERM

mkdir -p "${OUT_ROOT}"
cp "${BASE_RECON}/items.json" "${OUT_ROOT}/items.json"
printf '%s\n' "${BASE_RECON}" >"${OUT_ROOT}/BASELINE_SOURCE"

for arm in pure t2a mix; do
  merged="${TMP_ROOT}/model_${arm}.ckpt"
  python -m reca_audio.train.merge_ckpt \
    --ckpt "${BASE_CKPT}" --dit "$(dit_path "${arm}")" --out "${merged}"
  CUDA_VISIBLE_DEVICES="${GPU_PAIR}" python runs/scripts/eval_reconstruction_matrix.py generate \
    --out "${OUT_ROOT}" --label "${arm}" --ckpt "${merged}" \
    >"${OUT_ROOT}/${arm}.log" 2>&1
  rm -f "${merged}"
done

CUDA_VISIBLE_DEVICES="${gpu0}" python runs/scripts/eval_reconstruction_matrix.py score \
  --out "${OUT_ROOT}" --base-dir "${BASE_RECON}/base" --labels pure t2a mix \
  >"${OUT_ROOT}/score.log" 2>&1
date -Iseconds >"${OUT_ROOT}/COMPLETE"
cat "${OUT_ROOT}/results.json"
