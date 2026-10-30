#!/usr/bin/env bash
# 在服务器上准备 conda 环境。
#
# 为什么是「克隆」而不是 `conda env create`：
#   服务器上 dl 环境里已经有一套能在 V100(sm70) 上跑的 torch（2.14.1+cu126），
#   重建环境会从 pytorch 频道重装，又慢又可能装到不带 sm70 内核的版本。
#   克隆走硬链接，几分钟完成，且不动原环境。
#
# 用法（仓库根目录）：
#   bash server/setup_env.sh
#   # 或指定别的底座：
#   BASE_ENV=dl ENV_NAME=reca bash server/setup_env.sh
set -euo pipefail

ENV_NAME="${ENV_NAME:-reca-audio}"
BASE_ENV="${BASE_ENV:-dl}"

if [[ ! -f pyproject.toml ]]; then
  echo "[fatal] 请在仓库根目录运行（找不到 pyproject.toml）" >&2
  exit 1
fi

echo "[info] 目标环境=${ENV_NAME}  克隆自=${BASE_ENV}"

if conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[skip] 环境已存在：${ENV_NAME}"
else
  conda create -y -n "${ENV_NAME}" --clone "${BASE_ENV}"
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "[info] python: $(python -V 2>&1)"
pip install -U librosa soundfile "transformers>=4.44,<5" datasets accelerate timm pyyaml tqdm tensorboard
pip install -e .

python - <<'PY'
import torch, librosa, transformers, soundfile
print("torch       ", torch.__version__, "| cuda", torch.version.cuda)
print("capability  ", torch.cuda.get_device_capability(0), "| n_gpu", torch.cuda.device_count())
print("librosa     ", librosa.__version__)
print("transformers", transformers.__version__)
print("soundfile   ", soundfile.__version__)
PY

echo "[ok] 环境就绪：conda activate ${ENV_NAME}"
