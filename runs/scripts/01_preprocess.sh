#!/usr/bin/env bash
# 把 raw 音频统一成 24 kHz 单声道 PCM16（不切片，训练时再裁）
#
# 用法：
#   export DATA_ROOT=/path/to/data/reca-audio
#   bash scripts/01_preprocess.sh
#
# 冒烟（只处理 200 个）：
#   LIMIT=200 bash scripts/01_preprocess.sh
set -euo pipefail

: "${DATA_ROOT:?请先 export DATA_ROOT=/path/to/data/reca-audio}"
WORKERS="${WORKERS:-16}"
LIMIT="${LIMIT:-0}"
PY="${PY:-python}"

extra=()
if [[ "${LIMIT}" != "0" ]]; then
  extra=(--limit "${LIMIT}")
fi

run() {
  local name="$1" input="$2" out="$3"
  if [[ ! -d "${input}" ]]; then
    echo "[skip] ${name}: 目录不存在 ${input}"
    return 0
  fi
  echo "[pre ] ${name}"
  "${PY}" -m reca_audio.data.preprocess \
    --input "${input}" --out "${out}" --workers "${WORKERS}" "${extra[@]}"
}

run fsd50k "${DATA_ROOT}/raw/fsd50k/FSD50K.dev_audio" "${DATA_ROOT}/processed/fsd50k"
run fma    "${DATA_ROOT}/raw/fma/fma_small"            "${DATA_ROOT}/processed/fma"

echo "[done] 输出在 ${DATA_ROOT}/processed/"
