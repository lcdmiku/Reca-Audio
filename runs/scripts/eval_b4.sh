#!/usr/bin/env bash
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate audio-omni
cd ~/Reca-Audio
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DOWNLOAD_TIMEOUT=300 HF_HUB_ETAG_TIMEOUT=300
export QWEN_OMNI_MODEL_PATH=$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-3B/snapshots/*/ | head -1)
export CLAP_CKPT=/home/linchudeng/.cache/laion_clap/630k-audioset-best.pt
export PYTHONPATH=$PWD/src:$PWD/Audio-Omni:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date +%H:%M:%S)] merge b4"
python -m reca_audio.train.merge_ckpt --ckpt Audio-Omni/model/model.ckpt \
  --dit runs/train/reca_b4/latest.pt --out Audio-Omni/model/model_reca_b4.ckpt

echo "[$(date +%H:%M:%S)] 重建自检"
python - <<'PY'
import numpy as np
from audio_omni.api import AudioOmni
m = AudioOmni("Audio-Omni/model/Audio-Omni.json", "Audio-Omni/model/model_reca_b4.ckpt",
              device="cuda:0", conditioner_device="cuda:1", dtype="float16")
src = "runs/Benchmark/_data/audiocaps/full2/ref/103549.wav"
for cfg in (1.0, 7.0):
    vals = []
    for seed in (0, 1, 2):
        out = m.reconstruct(src, steps=50, seconds_total=10, seed=seed, cfg_scale=cfg).float().cpu().numpy()
        if out.ndim > 1: out = out.mean(0)
        spec = np.abs(np.fft.rfft(out*np.hanning(len(out))))
        vals.append(round(float(np.exp(np.mean(np.log(spec+1e-12)))/(np.mean(spec)+1e-12)), 3))
    print(f"[recon cfg={cfg}] flatness={vals}", flush=True)
PY

echo "[$(date +%H:%M:%S)] AudioCaps 生成 + 打分"
INPUT=$PWD/runs/Benchmark/_data/audiocaps/full/prompts1000.jsonl REF=$PWD/runs/Benchmark/_data/audiocaps/full2/ref \
OUT=$PWD/runs/Benchmark/output/AudioCaps_b4 CKPT=$PWD/Audio-Omni/model/model_reca_b4.ckpt GPUS_PER_PROC=2 \
  bash runs/Benchmark/run_shards.sh --benchmark AudioCaps
python runs/Benchmark/AudioCaps/evaluate.py score --out runs/Benchmark/output/AudioCaps_b4 \
  --ref runs/Benchmark/_data/audiocaps/full2/ref --device cuda:0 --panns
echo "[$(date +%H:%M:%S)] done"
