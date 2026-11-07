#!/usr/bin/env bash
# Guarded Base/A/B T2A diagnostic for the 400-exposure gradient experiment.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
REPO_ROOT="$(pwd)"

ACTION="${1:-status}"
SPLIT_ROOT="${SPLIT_ROOT:-runs/Benchmark/_data/audiocaps/reca_split_v1}"
BASE_CKPT="${BASE_CKPT:-Audio-Omni/model/model.ckpt}"
A_DIT="${A_DIT:-runs/train/reca_gradient_ab_v1/a/dit_exposure0400.pt}"
B_DIT="${B_DIT:-runs/train/reca_gradient_ab_v1/b/dit_exposure0400.pt}"
OUT_ROOT="${OUT_ROOT:-runs/Benchmark/output/GradientAB_T2A_89}"
EXPECTED_COUNT="${EXPECTED_COUNT:-89}"
GPU_IDS="${GPU_IDS:-1,2,3,4,5,6}"
NUM_SHARDS="${NUM_SHARDS:-3}"
GPUS_PER_PROC="${GPUS_PER_PROC:-2}"
SCORE_GPU="${SCORE_GPU:-6}"
STEPS="${STEPS:-100}"
CFG="${CFG:-6.0}"
AUDIO_SECONDS="${AUDIO_SECONDS:-10}"
SEED="${SEED:-1234}"
DTYPE="${DTYPE:-float16}"
GPU_MEMORY_LIMIT_MIB="${GPU_MEMORY_LIMIT_MIB:-128}"
SKIP_GPU_CHECK="${SKIP_GPU_CHECK:-0}"
CONDA_ENV="${CONDA_ENV:-audio-omni}"
TMP_ROOT=""

usage() {
  echo "usage: bash runs/scripts/eval_gradient_ab_t2a.sh {dry-run|run|status}"
}

cleanup() {
  local exit_status=$?
  trap - EXIT
  if [[ -n "${TMP_ROOT}" ]] && [[ "${TMP_ROOT}" == /tmp/reca-gradient-ab-eval.* ]] && [[ -d "${TMP_ROOT}" ]]; then
    rm -rf -- "${TMP_ROOT}"
    echo "[cleanup] removed ${TMP_ROOT}"
  fi
  exit "${exit_status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

activate_environment() {
  # shellcheck disable=SC1090
  source "${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
  conda activate "${CONDA_ENV}"
  PYTHON_BIN="${PYTHON_BIN:-python}"
  export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/Audio-Omni:${PYTHONPATH:-}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export QWEN_OMNI_MODEL_PATH="${QWEN_OMNI_MODEL_PATH:-$(find "${HOME}/.cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-3B/snapshots" -mindepth 1 -maxdepth 1 -type d -print -quit)}"
  [[ -n "${QWEN_OMNI_MODEL_PATH}" ]] && [[ -d "${QWEN_OMNI_MODEL_PATH}" ]] || {
    echo "Qwen snapshot is unavailable; set QWEN_OMNI_MODEL_PATH" >&2
    return 1
  }
}

validate_inputs() {
  local path
  for path in \
    "${SPLIT_ROOT}/test_manifest.jsonl" \
    "${SPLIT_ROOT}/split_report.json" \
    "${BASE_CKPT}" "${A_DIT}" "${B_DIT}" \
    "Audio-Omni/model/Audio-Omni.json" \
    "runs/Benchmark/AudioCaps/build_unique_sha_subset.py" \
    "runs/Benchmark/AudioCaps/evaluate.py" \
    "runs/Benchmark/run_shards.sh"; do
    [[ -f "${path}" ]] || {
      echo "missing required file: ${path}" >&2
      return 1
    }
  done
  [[ "${EXPECTED_COUNT}" == "89" ]] || {
    echo "formal diagnostic requires EXPECTED_COUNT=89, got ${EXPECTED_COUNT}" >&2
    return 1
  }
  [[ "${NUM_SHARDS}" == "3" ]] && [[ "${GPUS_PER_PROC}" == "2" ]] || {
    echo "formal diagnostic requires 3 shards x 2 GPUs" >&2
    return 1
  }
}

validate_output_target() {
  if [[ -e "${OUT_ROOT}" ]]; then
    echo "refusing existing output root: ${OUT_ROOT}" >&2
    return 1
  fi
}

validate_gpus() {
  [[ "${SKIP_GPU_CHECK}" == "1" ]] && return 0
  command -v nvidia-smi >/dev/null || {
    echo "nvidia-smi is unavailable" >&2
    return 1
  }
  local -a gpu_array
  local gpu used
  IFS=',' read -r -a gpu_array <<<"${GPU_IDS}"
  [[ "${#gpu_array[@]}" == "6" ]] || {
    echo "expected six GPU IDs, got ${GPU_IDS}" >&2
    return 1
  }
  [[ "$(printf '%s\n' "${gpu_array[@]}" | sort -u | wc -l)" == "6" ]] || {
    echo "GPU IDs must be unique: ${GPU_IDS}" >&2
    return 1
  }
  for gpu in "${gpu_array[@]}"; do
    used="$(nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')"
    if (( used > GPU_MEMORY_LIMIT_MIB )); then
      echo "GPU ${gpu} is not free: ${used} MiB used" >&2
      return 1
    fi
  done
}

build_subset() {
  local root="$1"
  "${PYTHON_BIN}" runs/Benchmark/AudioCaps/build_unique_sha_subset.py \
    --manifest "${SPLIT_ROOT}/test_manifest.jsonl" \
    --split-report "${SPLIT_ROOT}/split_report.json" \
    --prompts-out "${root}/prompts.jsonl" \
    --refs-out "${root}/ref" \
    --metadata-out "${root}/selection.json" \
    --expected-count "${EXPECTED_COUNT}"
}

write_status() {
  local state="$1" model="${2:-}"
  "${PYTHON_BIN}" - "${OUT_ROOT}/run_status.json" "${state}" "${model}" <<'PY'
import datetime
import json
import os
import sys

path, state, model = sys.argv[1:]
payload = {
    "state": state,
    "model": model or None,
    "pid": os.getpid(),
    "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
PY
}

write_contract() {
  "${PYTHON_BIN}" - "${OUT_ROOT}/run_contract.json" "${GPU_IDS}" "${BASE_CKPT}" "${A_DIT}" "${B_DIT}" <<PY
import datetime
import json
import socket
import sys

path, gpu_ids, base, arm_a, arm_b = sys.argv[1:]
payload = {
    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "host": socket.gethostname(),
    "model_order": ["base", "a", "b"],
    "checkpoints": {"base": base, "a": arm_a, "b": arm_b},
    "training_schedule": {"micro_batches": 400, "reca": 211, "t2a": 189,
                          "a_optimizer_updates": 400, "b_optimizer_updates": 100},
    "subset": {"expected_unique_sha256": ${EXPECTED_COUNT}},
    "generation": {"seed": ${SEED}, "steps": ${STEPS}, "cfg": ${CFG},
                   "seconds": ${AUDIO_SECONDS}, "dtype": "${DTYPE}"},
    "parallelism": {"gpu_ids": gpu_ids, "num_shards": ${NUM_SHARDS},
                    "gpus_per_process": ${GPUS_PER_PROC}, "score_gpu": "${SCORE_GPU}"},
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
PY
}

verify_generation() {
  local model_out="$1"
  "${PYTHON_BIN}" - "${model_out}/manifest.jsonl" "${EXPECTED_COUNT}" <<'PY'
import json
import pathlib
import sys

manifest = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])
rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
if len(rows) != expected:
    raise SystemExit(f"expected {expected} manifest rows, found {len(rows)}")
ids = [str(row.get("id")) for row in rows]
if len(set(ids)) != expected:
    raise SystemExit("generated manifest contains duplicate IDs")
bad = [row for row in rows if row.get("error") or not row.get("out") or not pathlib.Path(row["out"]).is_file()]
if bad:
    raise SystemExit(f"incomplete generation rows: {len(bad)}")
audio = list((manifest.parent / "audio").glob("*.wav"))
if len(audio) != expected:
    raise SystemExit(f"expected {expected} wav files, found {len(audio)}")
print(f"[verify] complete generation rows={len(rows)} wav={len(audio)}")
PY
}

run_model() {
  local label="$1" checkpoint="$2"
  local model_out="${OUT_ROOT}/${label}"
  validate_gpus
  write_status generating "${label}"
  echo "[$(date -Iseconds)] START model=${label} ckpt=${checkpoint}"
  INPUT="${OUT_ROOT}/subset/prompts.jsonl" \
  OUT="${model_out}" CKPT="${checkpoint}" \
  GPU_IDS="${GPU_IDS}" NUM_SHARDS="${NUM_SHARDS}" GPUS_PER_PROC="${GPUS_PER_PROC}" \
  STEPS="${STEPS}" CFG="${CFG}" DTYPE="${DTYPE}" REF="" \
    bash runs/Benchmark/run_shards.sh --benchmark AudioCaps \
      --seconds "${AUDIO_SECONDS}" --seed "${SEED}"
  verify_generation "${model_out}"
  write_status scoring "${label}"
  CUDA_VISIBLE_DEVICES="${SCORE_GPU}" "${PYTHON_BIN}" \
    runs/Benchmark/AudioCaps/evaluate.py score \
    --out "${model_out}" --ref "${OUT_ROOT}/subset/ref" --device cuda:0 --panns
  [[ -s "${model_out}/results.json" ]] || {
    echo "missing score output for ${label}" >&2
    return 1
  }
  echo "[$(date -Iseconds)] DONE model=${label}"
}

show_status() {
  echo "output=${OUT_ROOT}"
  if [[ -s "${OUT_ROOT}/run_status.json" ]]; then
    cat "${OUT_ROOT}/run_status.json"
  else
    echo "state=MISSING"
  fi
  local label
  for label in base a b; do
    if [[ -s "${OUT_ROOT}/${label}/results.json" ]]; then
      echo "${label} SCORED ${OUT_ROOT}/${label}/results.json"
    elif [[ -s "${OUT_ROOT}/${label}/manifest.jsonl" ]]; then
      echo "${label} GENERATED"
    else
      echo "${label} PENDING"
    fi
  done
}

dry_run() {
  activate_environment
  validate_inputs
  validate_output_target
  validate_gpus
  TMP_ROOT="$(mktemp -d /tmp/reca-gradient-ab-eval.XXXXXX)"
  build_subset "${TMP_ROOT}/subset"
  "${PYTHON_BIN}" - "${TMP_ROOT}/subset/selection.json" <<'PY'
import json
import sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
print(f"[dry-run] subset_count={data['count']} prompts_sha256={data['prompts_sha256']}")
PY
  echo "[dry-run] order=base,a,b gpus=${GPU_IDS} shards=${NUM_SHARDS}x${GPUS_PER_PROC}"
  echo "[dry-run] seed=${SEED} steps=${STEPS} cfg=${CFG} seconds=${AUDIO_SECONDS} dtype=${DTYPE}"
  echo "[dry-run] strict merge: ${A_DIT} -> /tmp/.../model_a.ckpt"
  echo "[dry-run] strict merge: ${B_DIT} -> /tmp/.../model_b.ckpt"
  echo "[dry-run] output=${OUT_ROOT}"
}

run_formal() {
  activate_environment
  validate_inputs
  validate_output_target
  validate_gpus
  TMP_ROOT="$(mktemp -d /tmp/reca-gradient-ab-eval.XXXXXX)"
  mkdir -p "${OUT_ROOT}"
  exec > >(tee -a "${OUT_ROOT}/pipeline.log") 2>&1
  echo "[$(date -Iseconds)] formal evaluation start pid=$$"
  df -h /tmp "${HOME}"
  du -sh "${HOME}"
  build_subset "${OUT_ROOT}/subset"
  write_contract
  write_status running base

  run_model base "${BASE_CKPT}"

  local merged="${TMP_ROOT}/model_a.ckpt"
  write_status merging a
  "${PYTHON_BIN}" -m reca_audio.train.merge_ckpt \
    --ckpt "${BASE_CKPT}" --dit "${A_DIT}" --out "${merged}"
  run_model a "${merged}"
  rm -f -- "${merged}"

  merged="${TMP_ROOT}/model_b.ckpt"
  write_status merging b
  "${PYTHON_BIN}" -m reca_audio.train.merge_ckpt \
    --ckpt "${BASE_CKPT}" --dit "${B_DIT}" --out "${merged}"
  run_model b "${merged}"
  rm -f -- "${merged}"

  write_status complete
  df -h /tmp "${HOME}"
  du -sh "${HOME}"
  echo "[$(date -Iseconds)] formal evaluation complete"
}

case "${ACTION}" in
  dry-run) dry_run ;;
  run) run_formal ;;
  status) show_status ;;
  *) usage >&2; exit 2 ;;
esac
