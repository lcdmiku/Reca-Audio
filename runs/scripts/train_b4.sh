#!/usr/bin/env bash
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate audio-omni
cd ~/Reca-Audio
export QWEN_OMNI_MODEL_PATH=$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-3B/snapshots/*/ | head -1)
export PYTHONPATH=$PWD/src:$PWD/Audio-Omni:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m reca_audio.train.train \
  --manifest runs/Benchmark/_data/audiocaps/train_manifest.jsonl \
  --out "${OUT:-runs/train/reca_b4}" \
  --steps "${STEPS:-2000}" --seconds 5 --batch-size 1 --grad-accum 1 \
  --lr "${LR:-2e-5}" --warmup 100 --optim adafactor --vae-device cpu --amp \
  --cfg-dropout 0 \
  --omni-downsample "${DS:-4}" --omni-seconds "${OMNI_S:-1.5}" \
  --save-every 500 --log-every 10 --num-workers 2
