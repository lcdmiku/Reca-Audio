#!/usr/bin/env bash
# Pure/T2A/Mix long matrix using the exact exposure-indexed arm-B optimizer regime.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
REPO_ROOT="$(pwd)"
ACTION="${1:-status}"
OUT_ROOT="${OUT_ROOT:-runs/train/reca_long_matrix_b_v1}"
SPLIT_ROOT="${SPLIT_ROOT:-runs/Benchmark/_data/audiocaps/reca_split_v1}"
MANIFEST="${MANIFEST:-${SPLIT_ROOT}/train_manifest.jsonl}"
DIAGNOSTIC_MANIFEST="${DIAGNOSTIC_MANIFEST:-${SPLIT_ROOT}/test_manifest.jsonl}"
CKPT="${CKPT:-Audio-Omni/model/model.ckpt}"
GPU_PAIRS="${GPU_PAIRS:-0,1 2,3 4,5}"
SEED="${SEED:-42}"

status() {
  local arm pid state
  for arm in pure t2a mix; do
    if [[ ! -s "${OUT_ROOT}/${arm}.pid" ]]; then
      echo "${arm} MISSING"
      continue
    fi
    pid="$(<"${OUT_ROOT}/${arm}.pid")"
    state="EXITED"
    if kill -0 "${pid}" 2>/dev/null; then
      state="RUNNING"
    fi
    echo "${arm} ${state} pid=${pid} log=${OUT_ROOT}/${arm}.log"
    [[ -s "${OUT_ROOT}/${arm}.log" ]] && tail -n 1 "${OUT_ROOT}/${arm}.log"
  done
}

validate_output() {
  if [[ -e "${OUT_ROOT}" ]]; then
    echo "refusing existing output root: ${OUT_ROOT}" >&2
    return 1
  fi
}

validate_gpus() {
  local required="$1" pair gpu used
  local -a pairs=()
  local -A seen=()
  read -r -a pairs <<<"${GPU_PAIRS}"
  if (( ${#pairs[@]} != required )); then
    echo "expected ${required} GPU pairs, got ${#pairs[@]}: ${GPU_PAIRS}" >&2
    return 1
  fi
  command -v nvidia-smi >/dev/null || {
    echo "nvidia-smi is unavailable" >&2
    return 1
  }
  for pair in "${pairs[@]}"; do
    IFS=, read -r -a pair_gpus <<<"${pair}"
    if (( ${#pair_gpus[@]} != 2 )); then
      echo "invalid GPU pair: ${pair}" >&2
      return 1
    fi
    for gpu in "${pair_gpus[@]}"; do
      if [[ -n "${seen[${gpu}]:-}" ]]; then
        echo "duplicate GPU in GPU_PAIRS: ${gpu}" >&2
        return 1
      fi
      seen["${gpu}"]=1
      used="$(nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')"
      if [[ ! "${used}" =~ ^[0-9]+$ ]] || (( used > ${GPU_MEMORY_LIMIT_MIB:-128} )); then
        echo "GPU ${gpu} is not free: ${used} MiB used" >&2
        return 1
      fi
    done
  done
}

activate_environment() {
  source "${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
  conda activate "${CONDA_ENV:-audio-omni}"
  export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/Audio-Omni:${PYTHONPATH:-}"
  export QWEN_OMNI_MODEL_PATH="${QWEN_OMNI_MODEL_PATH:-$(find "${HOME}/.cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-3B/snapshots" -mindepth 1 -maxdepth 1 -type d -print -quit)}"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
}

prepare_schedule() {
  local arm="$1" task_mode="$2"
  python -m reca_audio.train.run_gradient_ab \
    --arm b \
    --manifest "${MANIFEST}" \
    --schedule "${OUT_ROOT}/${arm}_schedule.jsonl" \
    --task-mode "${task_mode}" \
    --total-exposures "${TOTAL}" \
    --warmup-exposures "${WARMUP}" \
    --seed "${SEED}" \
    --prepare-only
}

run_arm() {
  local arm="$1" task_mode="$2" gpus="$3"
  CUDA_VISIBLE_DEVICES="${gpus}" python -m reca_audio.train.run_gradient_ab \
    --arm b \
    --manifest "${MANIFEST}" \
    --diagnostic-manifest "${DIAGNOSTIC_MANIFEST}" \
    --out "${OUT_ROOT}/${arm}" \
    --schedule "${OUT_ROOT}/${arm}_schedule.jsonl" \
    --task-mode "${task_mode}" \
    --total-exposures "${TOTAL}" \
    --warmup-exposures "${WARMUP}" \
    --peak-lr 2e-5 \
    --diagnostic-samples "${DIAGNOSTIC_SAMPLES}" \
    --diagnostic-exposures "${DIAGNOSTIC_EXPOSURES}" \
    --save-exposures "${TOTAL}" \
    --keep-final-checkpoint \
    --allow-formal \
    --model-config "${REPO_ROOT}/Audio-Omni/model/Audio-Omni.json" \
    --ckpt "${CKPT}" \
    --seconds 5 \
    --device cuda:0 \
    --conditioner-device cuda:1 \
    --vae-device cpu \
    --dtype float32 \
    --amp \
    --omni-downsample 4 \
    --omni-seconds 0 \
    --seed "${SEED}"
}

run_smoke_arm() {
  local arm="$1" task_mode="$2" gpus="$3"
  run_arm "${arm}" "${task_mode}" "${gpus}" >"${OUT_ROOT}/${arm}.log" 2>&1
}

case "${ACTION}" in
  status)
    status
    exit 0
    ;;
  smoke)
    TOTAL=4
    WARMUP=1
    DIAGNOSTIC_SAMPLES=1
    DIAGNOSTIC_EXPOSURES="0,4"
    validate_output
    GPU_PAIRS="${GPU_PAIRS:-0,1}"
    # A default six-pair layout is narrowed to one pair for sequential smoke.
    [[ "${GPU_PAIRS}" == "0,1 2,3 4,5" ]] && GPU_PAIRS="0,1"
    validate_gpus 1
    ;;
  run)
    if [[ "${ALLOW_FORMAL:-0}" != "1" ]]; then
      echo "refusing formal matrix without ALLOW_FORMAL=1" >&2
      exit 2
    fi
    TOTAL=2000
    WARMUP=100
    DIAGNOSTIC_SAMPLES="${DIAGNOSTIC_SAMPLES:-8}"
    DIAGNOSTIC_EXPOSURES="${DIAGNOSTIC_EXPOSURES:-0,1000,2000}"
    validate_output
    validate_gpus 3
    ;;
  *)
    echo "usage: $0 {smoke|run|status}" >&2
    exit 2
    ;;
esac

[[ -f "${MANIFEST}" ]] || { echo "missing manifest: ${MANIFEST}" >&2; exit 1; }
[[ -f "${DIAGNOSTIC_MANIFEST}" ]] || { echo "missing diagnostic manifest: ${DIAGNOSTIC_MANIFEST}" >&2; exit 1; }
[[ -f "${CKPT}" ]] || { echo "missing checkpoint: ${CKPT}" >&2; exit 1; }
activate_environment
mkdir -p "${OUT_ROOT}"

prepare_schedule pure pure
prepare_schedule t2a t2a
prepare_schedule mix balanced_mix

read -r -a pairs <<<"${GPU_PAIRS}"
if [[ "${ACTION}" == "smoke" ]]; then
  run_smoke_arm pure pure "${pairs[0]}"
  run_smoke_arm t2a t2a "${pairs[0]}"
  run_smoke_arm mix balanced_mix "${pairs[0]}"
  echo "completed smoke: ${OUT_ROOT}"
  exit 0
fi

declare -a pids=()
declare -a arms=(pure t2a mix)
run_arm pure pure "${pairs[0]}" >"${OUT_ROOT}/pure.log" 2>&1 &
pids+=("$!")
run_arm t2a t2a "${pairs[1]}" >"${OUT_ROOT}/t2a.log" 2>&1 &
pids+=("$!")
run_arm mix balanced_mix "${pairs[2]}" >"${OUT_ROOT}/mix.log" 2>&1 &
pids+=("$!")
for index in "${!pids[@]}"; do
  printf '%s\n' "${pids[${index}]}" >"${OUT_ROOT}/${arms[${index}]}.pid"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if (( failed )); then
  echo "one or more matrix arms failed" >&2
  exit 1
fi
echo "completed formal matrix: ${OUT_ROOT}"
