#!/usr/bin/env bash
# D0 服务器体检：把输出整段贴回来即可。
# 用法：bash server/00_check_env.sh
set -uo pipefail

echo "===== 1. 主机 ====="
uname -a || true
cat /etc/os-release 2>/dev/null | head -3 || true
echo

echo "===== 2. GPU ====="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true
nvidia-smi -L || true
echo

echo "===== 3. 资源 ====="
echo "-- CPU --"; nproc || true
echo "-- MEM --"; free -g || true
echo "-- DISK --"; df -h . /tmp 2>/dev/null || true
echo

echo "===== 4. conda ====="
which conda || echo "conda NOT found"
conda --version 2>/dev/null || true
conda env list 2>/dev/null || true
echo

echo "===== 5. python / torch ====="
which python || true
python -V 2>/dev/null || true
python - <<'PY' 2>/dev/null || echo "python 侧检查失败（可能未装 torch）"
import sys
print("python", sys.version.split()[0])
try:
    import torch
    print("torch", torch.__version__, "| cuda", torch.version.cuda)
    print("cuda_available", torch.cuda.is_available(), "| n_gpu", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("capability", torch.cuda.get_device_capability(0))
        x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
        print("fp16_matmul_ok", float((x @ x).abs().mean()))
        import torch.nn.functional as F
        q = torch.randn(1, 8, 256, 64, device="cuda", dtype=torch.float16)
        print("sdpa_ok", tuple(F.scaled_dot_product_attention(q, q, q).shape))
except Exception as e:
    print("torch_check_error:", repr(e))
PY
echo

echo "===== 6. 网络 ====="
for u in https://huggingface.co https://hf-mirror.com https://zenodo.org https://pypi.org; do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 8 "$u" 2>/dev/null || echo "ERR")
  echo "$u -> $code"
done
echo

echo "===== 7. ffmpeg ====="
ffmpeg -version 2>/dev/null | head -1 || echo "ffmpeg NOT found"
echo

echo "===== DONE ====="
