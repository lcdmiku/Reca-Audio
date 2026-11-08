#!/usr/bin/env bash
# Evaluate the exact-B long matrix with the same held-out protocol as Matrix v1.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
ROOT_DIR="$(pwd)"
ACTION="${1:-status}"
OUT_ROOT="${OUT_ROOT:-runs/Benchmark/output/LongMatrixB_eval_v1}"
BASE_EVAL="${BASE_EVAL:-runs/Benchmark/output/Matrix_v1_eval3_6gpu_9ce3984}"
BASE_CKPT="${BASE_CKPT:-Audio-Omni/model/model.ckpt}"
GPU_IDS="${GPU_IDS:-1,2,3,4,5,6}"
AC_INPUT="${AC_INPUT:-runs/Benchmark/_data/audiocaps/reca_split_v1/test_prompts.jsonl}"
AC_REF="${AC_REF:-runs/Benchmark/_data/audiocaps/reca_split_v1/test_ref}"
MC_INPUT="${MC_INPUT:-runs/Benchmark/_data/musiccaps/prompts1000.jsonl}"
MC_REF="${MC_REF:-runs/Benchmark/_data/musiccaps/ref}"

dit_path() {
  case "$1" in
    pure) echo "runs/train/reca_long_matrix_b_v1/pure/dit_exposure2000.pt" ;;
    t2a) echo "runs/train/reca_long_matrix_b_v1/t2a/dit_exposure2000.pt" ;;
    mix) echo "runs/train/reca_long_matrix_b_v1/mix/dit_exposure2000.pt" ;;
    *) return 2 ;;
  esac
}

status() {
  if [[ ! -e "${OUT_ROOT}" ]]; then
    echo "MISSING ${OUT_ROOT}"
    return 0
  fi
  [[ -s "${OUT_ROOT}/pipeline.pid" ]] && echo "pid=$(<"${OUT_ROOT}/pipeline.pid")"
  find "${OUT_ROOT}" -type f -name results.json -print -exec cat {} \;
  [[ -s "${OUT_ROOT}/pipeline.log" ]] && tail -n 10 "${OUT_ROOT}/pipeline.log"
}

validate_gpus() {
  local gpu used
  local -A seen=()
  IFS=, read -r -a gpus <<<"${GPU_IDS}"
  (( ${#gpus[@]} == 6 )) || { echo "expected six GPU IDs" >&2; return 1; }
  for gpu in "${gpus[@]}"; do
    [[ -z "${seen[${gpu}]:-}" ]] || { echo "duplicate GPU: ${gpu}" >&2; return 1; }
    seen["${gpu}"]=1
    used="$(nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')"
    (( used <= ${GPU_MEMORY_LIMIT_MIB:-128} )) || { echo "GPU ${gpu} is occupied: ${used} MiB" >&2; return 1; }
  done
}

run_case() {
  local benchmark="$1" group="$2" arm="$3" input="$4" ref="$5" checkpoint="$6"
  local out="${OUT_ROOT}/${group}/${arm}"
  [[ ! -e "${out}" ]] || { echo "refusing existing output: ${out}" >&2; return 1; }
  echo "[$(date -Iseconds)] START benchmark=${benchmark} arm=${arm}"
  INPUT="${input}" OUT="${out}" CKPT="${checkpoint}" \
    GPUS_PER_PROC=2 NUM_SHARDS=3 GPU_IDS="${GPU_IDS}" STEPS=100 CFG=6 DTYPE=float16 \
    bash runs/Benchmark/run_shards.sh --benchmark "${benchmark}"
  if [[ "${benchmark}" == "MusicCaps" ]]; then
    cat "${out}"/manifest.shard*.jsonl >"${out}/manifest.jsonl"
  fi
  CUDA_VISIBLE_DEVICES="${SCORE_GPU}" python "runs/Benchmark/${benchmark}/evaluate.py" score \
    --out "${out}" --ref "${ref}" --device cuda:0 --panns
  echo "[$(date -Iseconds)] DONE benchmark=${benchmark} arm=${arm}"
}

if [[ "${ACTION}" == "status" ]]; then
  status
  exit 0
fi
[[ "${ACTION}" == "run" ]] || { echo "usage: $0 {run|status}" >&2; exit 2; }
[[ ! -e "${OUT_ROOT}" ]] || { echo "refusing existing output root: ${OUT_ROOT}" >&2; exit 2; }
[[ -s "${BASE_EVAL}/COMPLETE" ]] || { echo "missing complete baseline evaluation" >&2; exit 1; }
for path in "${BASE_CKPT}" "${AC_INPUT}" "${MC_INPUT}"; do
  [[ -s "${path}" ]] || { echo "missing input: ${path}" >&2; exit 1; }
done
for path in "${AC_REF}" "${MC_REF}"; do
  [[ -d "${path}" ]] || { echo "missing reference directory: ${path}" >&2; exit 1; }
done
for arm in pure t2a mix; do
  [[ -s "$(dit_path "${arm}")" ]] || { echo "missing DiT checkpoint for ${arm}" >&2; exit 1; }
done
validate_gpus

source "${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-audio-omni}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_ETAG_TIMEOUT=300
export QWEN_OMNI_MODEL_PATH="${QWEN_OMNI_MODEL_PATH:-$(find "${HOME}/.cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-3B/snapshots" -mindepth 1 -maxdepth 1 -type d -print -quit)}"
export CLAP_CKPT="${CLAP_CKPT:-${HOME}/.cache/laion_clap/630k-audioset-best.pt}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/Audio-Omni:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SCORE_GPU="${GPU_IDS%%,*}"

TMP_ROOT="$(mktemp -d /tmp/reca-long-matrix-b-eval.XXXXXX)"
cleanup() {
  [[ -n "${TMP_ROOT:-}" ]] && [[ "${TMP_ROOT}" == /tmp/reca-long-matrix-b-eval.* ]] && rm -rf -- "${TMP_ROOT}"
}
trap cleanup EXIT INT TERM

mkdir -p "${OUT_ROOT}/AudioCapsHeldout" "${OUT_ROOT}/MusicCaps"
cp "${BASE_EVAL}/AudioCapsHeldout/base/results.json" "${OUT_ROOT}/AudioCapsHeldout/base_results.json"
cp "${BASE_EVAL}/MusicCaps/base/results.json" "${OUT_ROOT}/MusicCaps/base_results.json"
printf '%s\n' "${BASE_EVAL}" >"${OUT_ROOT}/BASELINE_SOURCE"

for arm in pure t2a mix; do
  merged="${TMP_ROOT}/model_${arm}.ckpt"
  python -m reca_audio.train.merge_ckpt \
    --ckpt "${BASE_CKPT}" --dit "$(dit_path "${arm}")" --out "${merged}"
  run_case AudioCaps AudioCapsHeldout "${arm}" "${AC_INPUT}" "${AC_REF}" "${merged}"
  run_case MusicCaps MusicCaps "${arm}" "${MC_INPUT}" "${MC_REF}" "${merged}"
  rm -f "${merged}"
done

date -Iseconds >"${OUT_ROOT}/COMPLETE"
echo "[pipeline] complete"
