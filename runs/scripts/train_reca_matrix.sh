#!/usr/bin/env bash
# Guarded launcher for the controlled ReCA/T2A mechanism experiment.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
REPO_ROOT="$(pwd)"

ACTION="${1:-status}"
if [[ "${ACTION}" == "smoke" ]]; then
  ROOT="${ROOT:-runs/train/reca_matrix_smoke}"
else
  ROOT="${ROOT:-runs/train/reca_matrix_v1}"
fi
SPLIT_DIR="${SPLIT_DIR:-runs/Benchmark/_data/audiocaps/reca_split_v1}"
CKPT="${CKPT:-Audio-Omni/model/model.ckpt}"
STEPS="${STEPS:-2000}"
SEED="${SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_GPU_CHECK="${SKIP_GPU_CHECK:-0}"
CONDA_ENV="${CONDA_ENV:-audio-omni}"

ARMS=(pure t2a mix)

resolve_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
    return 0
  fi
  local conda_python="${HOME}/miniconda3/envs/${CONDA_ENV}/bin/python"
  if [[ -x "${conda_python}" ]]; then
    PYTHON_BIN="${conda_python}"
    return 0
  fi
  echo "python is unavailable; activate ${CONDA_ENV} or set PYTHON_BIN" >&2
  return 1
}

usage() {
  echo "usage: bash runs/scripts/train_reca_matrix.sh {smoke|launch|status}"
}

status() {
  local arm arm_dir pid state
  for arm in "${ARMS[@]}"; do
    arm_dir="${ROOT}/${arm}"
    if [[ ! -d "${arm_dir}" ]]; then
      echo "${arm} MISSING"
      continue
    fi
    if [[ ! -s "${arm_dir}/train.pid" ]]; then
      echo "${arm} UNKNOWN (no train.pid)"
      continue
    fi
    pid="$(<"${arm_dir}/train.pid")"
    state="EXITED"
    if kill -0 "${pid}" 2>/dev/null; then
      state="RUNNING"
    fi
    echo "${arm} ${state} pid=${pid} log=${arm_dir}/launcher.log"
    if [[ -s "${arm_dir}/train_log.jsonl" ]]; then
      tail -n 1 "${arm_dir}/train_log.jsonl"
    elif [[ -s "${arm_dir}/launcher.log" ]]; then
      tail -n 1 "${arm_dir}/launcher.log"
    fi
  done
}

validate_split() {
  resolve_python
  [[ -f "${SPLIT_DIR}/train_manifest.jsonl" ]] || {
    echo "missing split manifest: ${SPLIT_DIR}/train_manifest.jsonl" >&2
    return 1
  }
  [[ -f "${SPLIT_DIR}/split_report.json" ]] || {
    echo "missing split report: ${SPLIT_DIR}/split_report.json" >&2
    return 1
  }
  "${PYTHON_BIN}" - "${SPLIT_DIR}/split_report.json" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    report = json.load(handle)
intersections = report.get("intersections", {})
expected = ("train_val", "train_test", "val_test")
bad = []
for name in expected:
    counts = intersections.get(name)
    if not isinstance(counts, dict):
        bad.append(f"{name}=missing")
        continue
    for key in ("id", "sha256"):
        if counts.get(key) != 0:
            bad.append(f"{name}.{key}={counts.get(key)!r}")
if bad:
    raise SystemExit("non-disjoint split: " + ", ".join(bad))
print("split validation: disjoint by id and sha256")
PY
}

validate_outputs() {
  local arm arm_dir
  for arm in "${ARMS[@]}"; do
    arm_dir="${ROOT}/${arm}"
    if [[ -d "${arm_dir}" ]] && [[ -n "$(find "${arm_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      echo "refusing non-empty output: ${arm_dir}" >&2
      return 1
    fi
  done
}

validate_gpus() {
  [[ "${SKIP_GPU_CHECK}" == "1" ]] && return 0
  command -v nvidia-smi >/dev/null || {
    echo "nvidia-smi is unavailable" >&2
    return 1
  }
  local index used
  while IFS=, read -r index used; do
    index="${index//[[:space:]]/}"
    used="${used//[[:space:]]/}"
    if (( index <= 5 && used > 128 )); then
      echo "GPU ${index} is not free: ${used} MiB used" >&2
      return 1
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
}

print_arm() {
  local arm="$1" gpus="$2" mode="$3" lambda_reca="$4" lambda_t2a="$5"
  echo "${arm} gpus=${gpus} mode=${mode} lambda_reca=${lambda_reca} lambda_t2a=${lambda_t2a}"
}

run_arm() {
  local arm="$1" gpus="$2" mode="$3" lambda_reca="$4" lambda_t2a="$5"
  local arm_dir="${ROOT}/${arm}"
  local -a extra=()
  if [[ "${ACTION}" == "smoke" ]]; then
    extra+=(--limit 2 --num-workers 0)
  else
    extra+=(--num-workers 2)
  fi

  print_arm "${arm}" "${gpus}" "${mode}" "${lambda_reca}" "${lambda_t2a}"
  nohup env CUDA_VISIBLE_DEVICES="${gpus}" python -m reca_audio.train.train \
    --manifest "${SPLIT_DIR}/train_manifest.jsonl" \
    --out "${arm_dir}" \
    --model-config "${REPO_ROOT}/Audio-Omni/model/Audio-Omni.json" \
    --ckpt "${CKPT}" \
    --steps "${STEPS}" \
    --seconds 5 \
    --batch-size 1 \
    --grad-accum 1 \
    --lr 2e-5 \
    --weight-decay 0.01 \
    --warmup "${WARMUP}" \
    --task-mode "${mode}" \
    --lambda-reca "${lambda_reca}" \
    --lambda-t2a "${lambda_t2a}" \
    --cfg-dropout 0 \
    --device cuda:0 \
    --conditioner-device cuda:1 \
    --dtype float32 \
    --amp \
    --optim adafactor \
    --vae-device cpu \
    --omni-downsample 4 \
    --omni-seconds 0 \
    --save-every "${SAVE_EVERY}" \
    --log-every "${LOG_EVERY}" \
    --seed "${SEED}" \
    "${extra[@]}" \
    >"${arm_dir}/launcher.log" 2>&1 &
  echo "$!" >"${arm_dir}/train.pid"
  echo "${arm} started pid=$(<"${arm_dir}/train.pid")"
}

launch() {
  validate_split
  [[ -f "${CKPT}" ]] || {
    echo "missing checkpoint: ${CKPT}" >&2
    return 1
  }
  validate_outputs
  validate_gpus

  print_arm pure 0,1 reca 1 0
  print_arm t2a 2,3 t2a 0 1
  print_arm mix 4,5 mix 1 1
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "dry-run: validation passed; no directories or processes created"
    return 0
  fi

  source "${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
  conda activate "${CONDA_ENV}"
  export QWEN_OMNI_MODEL_PATH="${QWEN_OMNI_MODEL_PATH:-$(find "${HOME}/.cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-3B/snapshots" -mindepth 1 -maxdepth 1 -type d -print -quit)}"
  [[ -n "${QWEN_OMNI_MODEL_PATH}" ]] && [[ -d "${QWEN_OMNI_MODEL_PATH}" ]] || {
    echo "Qwen snapshot is unavailable; set QWEN_OMNI_MODEL_PATH" >&2
    return 1
  }
  export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/Audio-Omni:${PYTHONPATH:-}"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

  mkdir -p "${ROOT}/pure" "${ROOT}/t2a" "${ROOT}/mix"
  run_arm pure 0,1 reca 1 0
  run_arm t2a 2,3 t2a 0 1
  run_arm mix 4,5 mix 1 1
}

case "${ACTION}" in
  status)
    status
    ;;
  smoke)
    STEPS="${STEPS:-2}"
    [[ "${STEPS}" == "2000" ]] && STEPS=2
    WARMUP="${WARMUP:-1}"
    SAVE_EVERY="${SAVE_EVERY:-2}"
    LOG_EVERY="${LOG_EVERY:-1}"
    launch
    ;;
  launch)
    WARMUP="${WARMUP:-100}"
    SAVE_EVERY="${SAVE_EVERY:-500}"
    LOG_EVERY="${LOG_EVERY:-10}"
    launch
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
