#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."

root=runs/Benchmark/output/AudioEdit_joint_qwen_v1
pair="${GPU_PAIR:-1,2}"
python="${PYTHON_BIN:-$HOME/miniconda3/envs/audio-omni/bin/python}"
export PYTHONPATH="$PWD/Audio-Omni:$PWD/src:$PWD/runs/Benchmark${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QWEN_OMNI_MODEL_PATH="${QWEN_OMNI_MODEL_PATH:-$(find "$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-3B/snapshots" -mindepth 1 -maxdepth 1 -type d -print -quit)}"
export CLAP_CKPT="${CLAP_CKPT:-$HOME/.cache/laion_clap/630k-audioset-best.pt}"
test -d "$QWEN_OMNI_MODEL_PATH"
test -s "$CLAP_CKPT"
[[ "$pair" =~ ^[0-7],[0-7]$ && ${pair%,*} != "${pair#*,}" ]]
export CUDA_VISIBLE_DEVICES="$pair"
for gpu in "${pair%,*}" "${pair#*,}"; do
    used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)"
    if (( used > 128 )); then
        echo "GPU $gpu occupied: $used MiB" >&2
        exit 1
    fi
done

reserve=$((2 * 1024 * 1024))
kib="$(du -sk "$HOME" | cut -f1)"
free="$(df -Pk runs/Benchmark/output | awk 'NR==2 {print $4}')"
test "$((kib + reserve))" -lt "$((400 * 1024 * 1024))"
test "$free" -gt "$reserve"
test ! -e "$root/COMPLETE"
mkdir -p "$root"
if [[ ! -s "$root/GENERATION_COMPLETE" ]]; then
    "$python" -u runs/Benchmark/AudioEdit/evaluate_joint_edit.py generate --out "$root" >>"$root/generation.log" 2>&1
fi
test -s "$root/GENERATION_COMPLETE"
"$python" -u runs/Benchmark/AudioEdit/evaluate_joint_edit.py score --out "$root" >>"$root/score.log" 2>&1
test -s "$root/results.json"
date -Is >"$root/COMPLETE"
