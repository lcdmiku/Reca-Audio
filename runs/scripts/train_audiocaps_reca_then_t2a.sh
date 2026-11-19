#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
ref=runs/train/audiocaps_low_lr_v1
out=runs/train/audiocaps_reca_then_t2a_v1
manifest=runs/Benchmark/_data/audiocaps/official_train_scale_v1/train_7070.jsonl
diagnostic=runs/Benchmark/_data/audiocaps/reca_split_v1/test_manifest.jsonl
pair="${GPU_PAIR:-1,2}"
python="${PYTHON_BIN:-$HOME/miniconda3/envs/audio-omni/bin/python}"
export PYTHONPATH="$PWD/src:$PWD/Audio-Omni:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QWEN_OMNI_MODEL_PATH="${QWEN_OMNI_MODEL_PATH:-$(find "$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-3B/snapshots" -mindepth 1 -maxdepth 1 -type d -print -quit)}"
test -s "$ref/COMPLETE" && test -s "$diagnostic" && test -d "$QWEN_OMNI_MODEL_PATH" || exit 1
[[ "$pair" =~ ^[0-9]+,[0-9]+$ && ${pair%,*} != ${pair#*,} ]]
check_gpus() {
    for gpu in "${pair%,*}" "${pair#*,}"; do
        used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)"
        if (( used > 128 )); then echo "GPU $gpu occupied: $used MiB" >&2; return 1; fi
    done
}
check_gpus
reserve=$(( $(stat -c %s "$ref/mix_7070/dit_exposure7072.pt") / 1024 + 1 + 2*1024*1024 ))
home_kib="$(du -sk "$HOME" | cut -f1)"
if (( home_kib + reserve >= 400 * 1024 * 1024 )) || (( $(df -Pk "$HOME" | awk 'NR==2 {print $4}') <= reserve )); then
    echo "Insufficient storage budget for one endpoint plus reserve" >&2
    exit 1
fi
mkdir "$out"
trap 'date -Is > "$out/FAILED"' ERR
cp DEPLOYED_COMMIT "$out/DEPLOYED_COMMIT"
cp "$ref/dataset_audit.json" "$out/dataset_audit.json"
"$python" - "$manifest" "$out/dataset_audit.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
manifest, audit_path = map(Path, sys.argv[1:])
audit = json.loads(audit_path.read_text())
if hashlib.sha256(manifest.read_bytes()).hexdigest() != audit["pools"]["7070"]["manifest_sha256"] or any(audit["heldout_intersection"].values()):
    raise ValueError("historical training manifest mismatch or held-out intersection")
PY
"$python" runs/scripts/prepare_reca_then_t2a.py --source "$ref/mix_7070_schedule.jsonl" \
    --coverage "$ref/source_coverage.json" --out "$out/schedule.jsonl"
head -n 2 "$ref/inputs.sha256" > "$out/base_inputs.sha256"
sha256sum --check "$out/base_inputs.sha256"
check_gpus
CUDA_VISIBLE_DEVICES="$pair" "$python" -u -m reca_audio.train.run_gradient_ab \
    --arm b --task-mode reca_then_t2a --manifest "$manifest" --schedule "$out/schedule.jsonl" \
    --out "$out/sequential_7070" --diagnostic-manifest "$diagnostic" \
    --total-exposures 7072 --warmup-exposures 352 --peak-lr 5e-6 \
    --diagnostic-samples 8 --diagnostic-exposures 0,2000,3536,4000,7072 --save-exposures 7072 \
    --keep-final-checkpoint --allow-formal --seconds 10 --omni-downsample 4 --omni-seconds 0 \
    --device cuda:0 --conditioner-device cuda:1 --vae-device cpu --dtype float32 --amp --seed 42 \
    > "$out/sequential_7070.log" 2>&1
test -s "$out/sequential_7070/dit_exposure7072.pt"
date -Is > "$out/COMPLETE"
