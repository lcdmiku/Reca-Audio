#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

mode="${1:?use smoke or formal}"
out="${2:?provide a new output directory}"
pair_a="${GPU_PAIR_A:-3,4}"
pair_b="${GPU_PAIR_B:-5,6}"
python_bin="${PYTHON_BIN:-$HOME/miniconda3/envs/audio-omni/bin/python}"
export PYTHONPATH="$PWD/src:$PWD/Audio-Omni:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CLAP_CKPT="${CLAP_CKPT:-$HOME/.cache/laion_clap/630k-audioset-best.pt}"
export QWEN_OMNI_MODEL_PATH="${QWEN_OMNI_MODEL_PATH:-$(find "$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-3B/snapshots" -mindepth 1 -maxdepth 1 -type d -print -quit)}"
test -s "$CLAP_CKPT"
test -d "$QWEN_OMNI_MODEL_PATH"
test ! -e "$out"
test "$(df -Pk "$HOME" | awk 'NR==2 {print $4}')" -gt 10485760
test "$(du -sk "$HOME" | cut -f1)" -lt 406847488

check_pair() {
    [[ "$1" =~ ^[0-9]+,[0-9]+$ ]] || return 1
    local first second gpu used
    IFS=, read -r first second <<<"$1"
    [[ "$first" != "$second" ]] || return 1
    for gpu in "$first" "$second"; do
        used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)"
        if (( used > 128 )); then
            echo "GPU $gpu occupied ($used MiB)" >&2
            return 1
        fi
    done
}

run() {
    local pair="$1" phase="$2"
    shift 2
    CUDA_VISIBLE_DEVICES="$pair" "$python_bin" -u runs/scripts/diagnose_posttrain.py "$phase" --out "$out" "$@"
}

check_pair "$pair_a"
if [[ "$mode" == smoke ]]; then
    run "$pair_a" prepare --smoke
    run "$pair_a" generate >"$out/generate.log" 2>&1
    run "$pair_a" geometry >"$out/geometry.log" 2>&1
    run "$pair_a" score >"$out/score.log" 2>&1
elif [[ "$mode" == formal ]]; then
    test -s "${3:?provide completed smoke directory}/COMPLETE"
    test -s "$3/results.json"
    check_pair "$pair_b"
    test "$(tr ',' '\n' <<<"$pair_a,$pair_b" | sort -u | wc -l)" -eq 4
    run "$pair_a" prepare
    run "$pair_a" geometry >"$out/geometry.log" 2>&1
    check_pair "$pair_b"
    run "$pair_a" generate --shard 0 --shards 2 >"$out/shard0.log" 2>&1 &
    first_pid=$!
    run "$pair_b" generate --shard 1 --shards 2 >"$out/shard1.log" 2>&1 &
    second_pid=$!
    failed=0
    wait "$first_pid" || failed=1
    wait "$second_pid" || failed=1
    test "$failed" -eq 0
    run "$pair_a" score >"$out/score.log" 2>&1
else
    echo "expected smoke or formal" >&2
    exit 2
fi
test -s "$out/results.json"
date -Is >"$out/COMPLETE"
