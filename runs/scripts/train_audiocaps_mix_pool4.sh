#!/usr/bin/env bash
# Reuse every low-LR Mix slot and change only the ReCA projected audio tokens.
set -euo pipefail
cd "$(dirname "$0")/../.."
ref=runs/train/audiocaps_low_lr_v1
out=runs/train/audiocaps_mix_pool4_v1
arm=mix_pool4_7070
manifest=runs/Benchmark/_data/audiocaps/official_train_scale_v1/train_7070.jsonl
diagnostic=runs/Benchmark/_data/audiocaps/reca_split_v1/test_manifest.jsonl
source_schedule="$ref/mix_7070_schedule.jsonl"
schedule_sha=bd5ca9ffa653c06af7d0c1eefb1e36e2fbf6ed09f8bf1b496afb9da1107126a0
pair="${GPU_PAIR:-1,2}"
python="${PYTHON_BIN:-$HOME/miniconda3/envs/audio-omni/bin/python}"
export PYTHONPATH="$PWD/src:$PWD/Audio-Omni:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export QWEN_OMNI_MODEL_PATH="${QWEN_OMNI_MODEL_PATH:-$(find "$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-3B/snapshots" -mindepth 1 -maxdepth 1 -type d -print -quit)}"
test -s "$ref/COMPLETE"
test ! -e "$out"
test -d "$QWEN_OMNI_MODEL_PATH"
test -s "$diagnostic"
[[ "$pair" =~ ^[0-9]+,[0-9]+$ && ${pair%,*} != ${pair#*,} ]]
check_gpus() {
    local gpu used
    for gpu in "${pair%,*}" "${pair#*,}"; do
        used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)"
        if (( used > 128 )); then echo "GPU $gpu occupied: $used MiB" >&2; return 1; fi
    done
}
check_gpus
reserve=$(( $(stat -c %s "$ref/mix_7070/dit_exposure7072.pt") / 1024 + 1 + 2*1024*1024 ))
home_kib="$(du -sk "$HOME" | cut -f1)"
if (( home_kib + reserve >= 400 * 1024 * 1024 || $(df -Pk . | awk 'NR==2 {print $4}') <= reserve )); then
    echo "Insufficient storage: home=$home_kib KiB, checkpoint plus reserve=$reserve KiB" >&2
    exit 1
fi
"$python" - "$ref" "$manifest" "$source_schedule" "$schedule_sha" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

ref, manifest, schedule = map(Path, sys.argv[1:4])
audit = json.loads((ref / "dataset_audit.json").read_text())
coverage = json.loads((ref / "source_coverage.json").read_text())["mix_7070"]
contract = json.loads((ref / "mix_7070/contract.json").read_text())
expected = dict(arm="b", task_mode="balanced_mix", accumulation=4, schedule=str(schedule),
                optimizer_updates=1768, total_exposures=7072, peak_lr=5e-6, warmup_exposures=352)
if any(contract.get(k) != v for k, v in expected.items()):
    raise ValueError("Unexpected reference Mix training contract")
if (coverage["manifest"] != str(manifest) or coverage["exposures"] != {"reca": 3536, "t2a": 3536}
        or coverage["schedule_sha256"] != sys.argv[4] or hashlib.sha256(schedule.read_bytes()).hexdigest() != sys.argv[4]
        or hashlib.sha256(manifest.read_bytes()).hexdigest() != audit["pools"]["7070"]["manifest_sha256"]
        or any(audit["heldout_intersection"].values())
        or (ref / "inputs.sha256").read_text().split()[0] != "8ddfdd6c965b7b1fc5c5af01e306cdfd78e44f56fd584789b8987233438e65b3"):
    raise ValueError("Reference Mix data, Base or schedule mismatch")
PY
mkdir "$out"
trap 'date -Is > "$out/FAILED"' ERR
cp DEPLOYED_COMMIT "$ref/dataset_audit.json" "$ref/source_coverage.json" "$out/"
cp "$source_schedule" "$out/schedule.jsonl"
printf '%s\n' "$schedule_sha" > "$out/schedule.jsonl.sha256"
head -n 2 "$ref/inputs.sha256" > "$out/base_inputs.sha256"
sha256sum --check "$out/base_inputs.sha256"
check_gpus
CUDA_VISIBLE_DEVICES="$pair" "$python" -u -m reca_audio.train.run_gradient_ab \
    --arm b --task-mode balanced_mix --reca-audio-pool-width 4 --manifest "$manifest" --schedule "$out/schedule.jsonl" \
    --schedule-digest "$schedule_sha" --out "$out/$arm" --diagnostic-manifest "$diagnostic" \
    --total-exposures 7072 --warmup-exposures 352 --peak-lr 5e-6 \
    --grad-clip 1.0 --weight-decay 0.01 --diagnostic-samples 8 --diagnostic-exposures 0,2000,4000,7072 \
    --save-exposures 7072 --keep-final-checkpoint --allow-formal --seconds 10 --omni-downsample 4 --omni-seconds 0 \
    --device cuda:0 --conditioner-device cuda:1 --vae-device cpu --dtype float32 --amp --seed 42 \
    > "$out/$arm.log" 2>&1
test -s "$out/$arm/dit_exposure7072.pt"
date -Is > "$out/COMPLETE"
