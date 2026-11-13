#!/usr/bin/env bash
# Data-pool-size comparison using the existing arm-B training entry point.
set -euo pipefail
cd "$(dirname "$0")/../.."
data="${DATA_ROOT:-runs/Benchmark/_data/audiocaps/official_train_scale_v1}"
out="${OUT_ROOT:-runs/train/audiocaps_scale_matrix_v2}"
pure_pair="${PURE_GPU_PAIR:-1,2}"
t2a_pair="${T2A_GPU_PAIR:-3,4}"
mix_pair="${MIX_GPU_PAIR:-5,6}"
python="${PYTHON_BIN:-$HOME/miniconda3/envs/audio-omni/bin/python}"
export PYTHONPATH="$PWD/src:$PWD/Audio-Omni:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export QWEN_OMNI_MODEL_PATH="${QWEN_OMNI_MODEL_PATH:-$(find "$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-3B/snapshots" -mindepth 1 -maxdepth 1 -type d -print -quit)}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

test ! -e "$out"
test -d "$QWEN_OMNI_MODEL_PATH"
if [[ ! -s "$data/audit.json" ]]; then
    if [[ -n "${PREP_PID:-}" ]]; then
        while [[ ! -s "$data/audit.json" ]]; do
            state="$(ps -o stat= -p "$PREP_PID")"
            [[ -n "$state" && "$state" != Z* ]] || exit 1
            sleep 20
        done
    else
        "$python" -u runs/scripts/prepare_audiocaps_scale.py --out "$data"
    fi
fi
# Bound projected home use using the observed size of the same checkpoint format.
reference=runs/train/reca_long_matrix_b_10s_v1/t2a/dit_exposure2000.pt
reserve_kib=$((4 * ($(stat -c %s "$reference") / 1024 + 1) + 2 * 1024 * 1024))
home_kib="$(du -sk "$HOME" | cut -f1)"
test "$((home_kib + reserve_kib))" -lt $((400 * 1024 * 1024))
test "$(df -Pk "$HOME" | awk 'NR==2 {print $4}')" -gt "$reserve_kib"
for pair in "$pure_pair" "$t2a_pair" "$mix_pair"; do
    [[ "$pair" =~ ^[0-9]+,[0-9]+$ ]]
done
test "$(tr ',' '\n' <<<"$pure_pair,$t2a_pair,$mix_pair" | sort -u | wc -l)" -eq 6
IFS=, read -r -a gpus <<<"$pure_pair,$t2a_pair,$mix_pair"
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
"$python" -u runs/scripts/prepare_audiocaps_scale.py --coverage-data "$data" --out "$out"
run_arm() {
    local arm="$1" count="$2" pair="$3" task="$4"
    CUDA_VISIBLE_DEVICES="$pair" "$python" -u -m reca_audio.train.run_gradient_ab \
        --arm b --task-mode "$task" --manifest "$data/train_${count}.jsonl" \
        --schedule "$out/${arm}_schedule.jsonl" --out "$out/$arm" \
        --diagnostic-manifest runs/Benchmark/_data/audiocaps/reca_split_v1/test_manifest.jsonl \
        --total-exposures 7072 --warmup-exposures 352 --peak-lr 2e-5 \
        --diagnostic-samples 8 --diagnostic-exposures 0,2000,4000,7072 --save-exposures 7072 \
        --keep-final-checkpoint --allow-formal --seconds 10 --omni-downsample 4 --omni-seconds 0 \
        --device cuda:0 --conditioner-device cuda:1 --vae-device cpu --dtype float32 --amp --seed 42 \
        >"$out/$arm.log" 2>&1
}

run_arm pure_7070 7070 "$pure_pair" pure &
pure_pid=$!
(run_arm t2a_7070 7070 "$t2a_pair" t2a && run_arm t2a_707 707 "$t2a_pair" t2a) &
t2a_pid=$!
run_arm mix_7070 7070 "$mix_pair" balanced_mix &
mix_pid=$!
failed=0
for pid in "$pure_pid" "$t2a_pid" "$mix_pid"; do
    wait "$pid" || failed=1
done
test "$failed" -eq 0
for arm in pure_7070 t2a_7070 mix_7070 t2a_707; do
    test -s "$out/$arm/dit_exposure7072.pt"
done
date -Is >"$out/COMPLETE"
