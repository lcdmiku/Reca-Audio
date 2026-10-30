#!/usr/bin/env bash
# 给 Audio-Omni 建独立环境（V100 / sm70）。
#
# 为什么不用 dl 环境：Audio-Omni 的 requirements 钉的是 torch 2.6.0+cu124 与
# transformers 5.3.0，跟 dl 里的 torch 2.14 差太多，混用会互相踩。
#
# 用法（仓库根目录）：
#   bash runs/server/setup_audio_omni.sh
set -euo pipefail

ENV_NAME="${ENV_NAME:-audio-omni}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
echo "[info] HF_ENDPOINT=${HF_ENDPOINT}"

if [[ ! -d Audio-Omni ]]; then
  echo "[fatal] 请在仓库根目录运行（找不到 Audio-Omni/）" >&2
  exit 1
fi

if conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[skip] 环境已存在：${ENV_NAME}"
else
  conda create -y -n "${ENV_NAME}" python=3.11
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "[info] python: $(python -V 2>&1)"

# torch 2.6.0 的 PyPI 轮子默认就是 cu124，V100(sm70) 有内核
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

pip install "transformers==5.3.0" accelerate qwen-omni-utils \
  einops einops-exts einx omegaconf timm safetensors huggingface_hub \
  k-diffusion torchdiffeq torchsde v-diffusion-pytorch \
  descript-audio-codec descript-audiotools alias-free-torch auraloss torch-stoi julius \
  resampy librosa soundfile torchlibrosa pyloudnorm pedalboard pypinyin jieba PyWavelets matplotlib \
  decord av

# 不要在 V100 上装 flash-attn / natten
pip install -e Audio-Omni --no-deps

python - <<'PY'
import torch
print("torch       ", torch.__version__, "| cuda", torch.version.cuda, "| cap", torch.cuda.get_device_capability(0))
import transformers
print("transformers", transformers.__version__)
from audio_omni.api import AudioOmni  # noqa: F401
print("[ok] audio_omni 可导入")
PY

echo
echo "[next] 下载权重（HuggingFace 走镜像）："
echo "  huggingface-cli download HKUSTAudio/Audio-Omni --local-dir Audio-Omni/model"
echo "  huggingface-cli download Qwen/Qwen2.5-Omni-3B --local-dir \$HF_HOME/qwen2.5-omni-3b"
echo "  export QWEN_OMNI_MODEL_PATH=\$HF_HOME/qwen2.5-omni-3b   # 可选，避免再次解析模型名"
echo
echo "[note] V100 上 Qwen 会退回 fp32（约 12 GB 显存）；可用 AUDIO_OMNI_QWEN_DTYPE=float16 试更省显存"
