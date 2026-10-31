#!/usr/bin/env bash
# 预取打分模型（对应 Reca Benchmark 里 evaluation/download_models.sh 的角色）。
#
# 用 HuggingFace 镜像，避免打分阶段现下几 GB 权重时超时。
#
#   bash runs/Benchmark/download_models.sh            # CLAP + Whisper-small
#   ASR_MODEL=openai/whisper-large-v3 bash runs/Benchmark/download_models.sh
#   WITH_PANNS=1 bash runs/Benchmark/download_models.sh
#   WITH_FADTK=1 bash runs/Benchmark/download_models.sh
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
ASR_MODEL="${ASR_MODEL:-openai/whisper-small}"

echo "[download] HF_ENDPOINT=${HF_ENDPOINT}"
echo "[download] ASR_MODEL=${ASR_MODEL}"

python - <<PY
import os, sys

ok = True

# 1) CLAP：FAD / CLAP score 用
try:
    import laion_clap
    m = laion_clap.CLAP_Module(enable_fusion=False, device="cpu")
    m.load_ckpt()
    print("[ok] laion-clap checkpoint")
except Exception as e:
    ok = False
    print(f"[warn] CLAP 预取失败：{type(e).__name__}: {e}")
    print("       若卡在 huggingface.co，可手动下载 lukewys/laion_clap 的 ckpt 后放到 ~/.cache/laion_clap/")

# 2) Whisper：WER 用
try:
    from transformers import pipeline
    pipeline("automatic-speech-recognition", model=os.environ.get("ASR_MODEL", "openai/whisper-small"), device=-1)
    print("[ok] ASR model")
except Exception as e:
    ok = False
    print(f"[warn] Whisper 预取失败：{type(e).__name__}: {e}")

# 3) PANNs：KL / IS / FD（可选）
if os.environ.get("WITH_PANNS") == "1":
    try:
        from panns_inference import AudioTagging
        AudioTagging(checkpoint_path=None, device="cpu")
        print("[ok] PANNs CNN14")
    except Exception as e:
        print(f"[warn] PANNs 预取失败：{type(e).__name__}: {e}")

sys.exit(0 if ok else 1)
PY

if [[ "${WITH_FADTK:-0}" == "1" ]]; then
  echo "[download] fadtk 需要单独安装：pip install fadtk，首次使用会自行下载 VGGish"
fi

echo "[done] 打分模型就绪（缺的会在运行时跳过并提示）"
