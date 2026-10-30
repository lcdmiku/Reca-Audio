#!/usr/bin/env bash
# 下载训练数据：FSD50K（音效，~80h）+ FMA-small（音乐，~66h）
#
# 用法：
#   export DATA_ROOT=/path/to/data/reca-audio
#   bash scripts/00_download_data.sh all
#   bash scripts/00_download_data.sh fsd50k   # 只下音效
#   bash scripts/00_download_data.sh fma      # 只下音乐
#
# 依赖：wget zip unzip（Ubuntu: sudo apt install -y wget zip unzip）
set -euo pipefail

: "${DATA_ROOT:?请先 export DATA_ROOT=/path/to/data/reca-audio}"
RAW="${DATA_ROOT}/raw"
mkdir -p "${RAW}"

need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[fatal] 缺少命令：$1（Ubuntu: sudo apt install -y wget zip unzip）" >&2
    exit 1
  fi
}

fetch() {  # url dest
  local url="$1" dest="$2"
  if [[ -s "$dest" ]]; then
    echo "[skip] $(basename "$dest")"
    return 0
  fi
  echo "[get ] $url"
  # 用 .part 中转，保证断点续传后文件名正确
  wget -c --tries=20 --timeout=60 --waitretry=5 -O "${dest}.part" "$url"
  mv "${dest}.part" "$dest"
}

fsd50k() {
  local d="${RAW}/fsd50k"
  mkdir -p "$d"
  local base="https://zenodo.org/api/records/4060432/files"

  # 元数据与小文件
  for f in FSD50K.doc.zip FSD50K.ground_truth.zip FSD50K.metadata.zip; do
    fetch "${base}/${f}/content" "${d}/${f}"
  done

  # dev 音频：5 个分卷 + 1 个 zip（顺序不能变）
  for i in 01 02 03 04 05; do
    fetch "${base}/FSD50K.dev_audio.z${i}/content" "${d}/FSD50K.dev_audio.z${i}"
  done
  fetch "${base}/FSD50K.dev_audio.zip/content" "${d}/FSD50K.dev_audio.zip"

  # eval 音频：1 个分卷 + 1 个 zip
  fetch "${base}/FSD50K.eval_audio.z01/content" "${d}/FSD50K.eval_audio.z01"
  fetch "${base}/FSD50K.eval_audio.zip/content" "${d}/FSD50K.eval_audio.zip"

  (
    cd "$d"
    if [[ ! -d FSD50K.dev_audio ]]; then
      [[ -f unsplit.dev.zip ]] || zip -s 0 FSD50K.dev_audio.zip --out unsplit.dev.zip
      unzip -q unsplit.dev.zip
    fi
    if [[ ! -d FSD50K.eval_audio ]]; then
      [[ -f unsplit.eval.zip ]] || zip -s 0 FSD50K.eval_audio.zip --out unsplit.eval.zip
      unzip -q unsplit.eval.zip
    fi
    unzip -qo FSD50K.ground_truth.zip
    unzip -qo FSD50K.metadata.zip
  )
  echo "[ok  ] FSD50K -> ${d}"
  du -sh "${d}" 2>/dev/null || true
}

fma() {
  local d="${RAW}/fma"
  mkdir -p "$d"
  fetch "https://os.unil.cloud.switch.ch/fma/fma_small.zip" "${d}/fma_small.zip"
  fetch "https://github.com/mdeff/fma/releases/download/v1.0/fma_metadata.zip" "${d}/fma_metadata.zip"
  (
    cd "$d"
    [[ -d fma_small ]] || unzip -q fma_small.zip
    [[ -d fma_metadata ]] || unzip -q fma_metadata.zip
  )
  echo "[ok  ] FMA-small -> ${d}"
  du -sh "${d}" 2>/dev/null || true
}

main() {
  need wget
  need unzip
  echo "[info] DATA_ROOT=${DATA_ROOT}"
  df -h "${DATA_ROOT}" || true
  case "${1:-all}" in
    fsd50k) need zip; fsd50k ;;
    fma)    fma ;;
    all)    need zip; fsd50k; fma ;;
    *) echo "用法: $0 [all|fsd50k|fma]" >&2; exit 2 ;;
  esac
  echo "[done]"
}

main "$@"
