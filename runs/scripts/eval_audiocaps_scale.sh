#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
eval_args=()
default_out=runs/audits/audiocaps_scale_eval_v2
entry=runs/scripts/eval_audiocaps_scale.py
reserve_kib=$((4 * 1024 * 1024))
if [[ "${1:-}" == --low-lr ]]; then
    shift
    eval_args+=(--low-lr)
    default_out=runs/audits/audiocaps_low_lr_eval_v1
elif [[ "${1:-}" == --reca-then-t2a ]]; then
    shift
    entry=runs/scripts/eval_reca_then_t2a.py
    default_out=runs/audits/audiocaps_reca_then_t2a_eval_v1
    reserve_kib=$((2 * 1024 * 1024))
fi
out="${1:-$default_out}"
python="${PYTHON_BIN:-$HOME/miniconda3/envs/audio-omni/bin/python}"
export PYTHONPATH="$PWD/src:$PWD/Audio-Omni:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CLAP_CKPT="${CLAP_CKPT:-$HOME/.cache/laion_clap/630k-audioset-best.pt}"
export QWEN_OMNI_MODEL_PATH="${QWEN_OMNI_MODEL_PATH:-$(find "$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-3B/snapshots" -mindepth 1 -maxdepth 1 -type d -print -quit)}"
test -s "$CLAP_CKPT"
test -d "$QWEN_OMNI_MODEL_PATH"
test ! -e "$out"
test "$(df -Pk "$HOME" | awk 'NR==2 {print $4}')" -gt "$reserve_kib"
test "$(du -sk "$HOME" | cut -f1)" -lt $((400 * 1024 * 1024 - reserve_kib))

pairs=(1,2 3,4 5,6)
for gpu in 1 2 3 4 5 6; do
    used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)"
    if (( used > 128 )); then
        echo "GPU $gpu occupied: $used MiB" >&2
        exit 1
    fi
done
run() {
    local pair="$1"
    shift
    CUDA_VISIBLE_DEVICES="$pair" "$python" -u "$entry" "$@" --out "$out" "${eval_args[@]}"
}
run 1,2 prepare
pids=()
for shard in "${!pairs[@]}"; do
    run "${pairs[$shard]}" generate --shard "$shard" --shards "${#pairs[@]}" >"$out/shard$shard.log" 2>&1 &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
test "$failed" -eq 0
run 1,2 score >"$out/score.log" 2>&1
test -s "$out/results.json"
date -Is >"$out/COMPLETE"
