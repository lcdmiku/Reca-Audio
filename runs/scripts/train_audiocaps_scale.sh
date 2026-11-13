#!/usr/bin/env bash
# Data-pool-size comparison using the existing arm-B training entry point.
set -euo pipefail
cd "$(dirname "$0")/../.."
data="${DATA_ROOT:-runs/Benchmark/_data/audiocaps/official_train_scale_v1}"
out="${OUT_ROOT:-runs/train/audiocaps_scale_t2a_v1}"
small_pair="${SMALL_GPU_PAIR:-3,4}"
large_pair="${LARGE_GPU_PAIR:-5,6}"
python="${PYTHON_BIN:-$HOME/miniconda3/envs/audio-omni/bin/python}"
export PYTHONPATH="$PWD/src:$PWD/Audio-Omni:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export QWEN_OMNI_MODEL_PATH="${QWEN_OMNI_MODEL_PATH:-$(find "$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-3B/snapshots" -mindepth 1 -maxdepth 1 -type d -print -quit)}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

test ! -e "$out"
test -d "$QWEN_OMNI_MODEL_PATH"
# Reserve room for two ~12 GB final checkpoints and up to ~7 GB of audio.
test "$(du -sk "$HOME" | cut -f1)" -lt $((365 * 1024 * 1024))
test "$(df -Pk "$HOME" | awk 'NR==2 {print $4}')" -gt $((35 * 1024 * 1024))
if [[ ! -s "$data/audit.json" ]]; then
    "$python" -u runs/scripts/prepare_audiocaps_scale.py --out "$data"
fi
[[ "$small_pair" =~ ^[0-9]+,[0-9]+$ && "$large_pair" =~ ^[0-9]+,[0-9]+$ ]]
test "$(tr ',' '\n' <<<"$small_pair,$large_pair" | sort -u | wc -l)" -eq 4
IFS=, read -r -a gpus <<<"$small_pair,$large_pair"
for gpu in "${gpus[@]}"; do
    used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)"
    if (( used > 128 )); then
        echo "GPU $gpu occupied: $used MiB" >&2
        exit 1
    fi
done

mkdir -p "$out"
cp "$data/audit.json" "$out/dataset_audit.json"
cp DEPLOYED_COMMIT "$out/DEPLOYED_COMMIT"
run_arm() {
    local count="$1" pair="$2"
    CUDA_VISIBLE_DEVICES="$pair" "$python" -u -m reca_audio.train.run_gradient_ab \
        --arm b --task-mode t2a --manifest "$data/train_${count}.jsonl" \
        --schedule "$data/schedule_${count}.jsonl" --out "$out/t2a_${count}" \
        --diagnostic-manifest runs/Benchmark/_data/audiocaps/reca_split_v1/test_manifest.jsonl \
        --total-exposures 2000 --warmup-exposures 100 --peak-lr 2e-5 \
        --diagnostic-samples 8 --diagnostic-exposures 0,1000,2000 --save-exposures 2000 \
        --keep-final-checkpoint --allow-formal --seconds 10 --omni-downsample 4 --omni-seconds 0 \
        --device cuda:0 --conditioner-device cuda:1 --vae-device cpu --dtype float32 --amp --seed 42
}

run_arm 707 "$small_pair" >"$out/t2a_707.log" 2>&1 &
small_pid=$!
run_arm 7070 "$large_pair" >"$out/t2a_7070.log" 2>&1 &
large_pid=$!
printf '%s\n' "$small_pid" >"$out/t2a_707.pid"
printf '%s\n' "$large_pid" >"$out/t2a_7070.pid"
failed=0
wait "$small_pid" || failed=1
wait "$large_pid" || failed=1
test "$failed" -eq 0
test -s "$out/t2a_707/dit_exposure2000.pt"
test -s "$out/t2a_7070/dit_exposure2000.pt"
date -Is >"$out/COMPLETE"
