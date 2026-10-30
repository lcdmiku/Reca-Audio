# 服务器侧说明（RecA-Audio）

## 工作流（三段式）

```
本地 Windows 工作区（我写代码/配置/文档）
        │  git / rsync / 共享盘
        ▼
服务器（你执行脚本，把 stdout/stderr 贴回来）
        │
        ▼
结果（日志、指标、音频样例）→ 回到工作区 → 我分析、改代码
```

- 我**不直接操作服务器**；所有命令都写成可直接复制的形式。
- 所有路径走配置/环境变量，不写死；代码用 `pathlib`，兼容 Linux。

## 目录约定（服务器上）

```
$DATA_ROOT/
├── raw/            # 原始音频（下载来的，不动）
├── processed/      # 重采样 24 kHz、单声道、10 s 切片
├── features/       # 各编码器预计算特征（.npy + index.jsonl）
└── eval/           # 固定评测集（100–200 条 + 提示词）

$RUN_ROOT/
├── ckpt/           # 训练 checkpoint
├── logs/           # tensorboard + stdout
└── samples/        # 生成样例 wav
```

建议在 `~/.bashrc` 里设两个变量，脚本只读它们：

```bash
export DATA_ROOT=/path/to/data/reca-audio
export RUN_ROOT=/path/to/runs/reca-audio
```

## D0：体检

```bash
bash server/00_check_env.sh
```

把整段输出贴回来。重点看四项：GPU 是否 8 张、`capability` 是否为 `(7, 0)`、盘位是否 ≥ 300 GB、`huggingface.co` / `hf-mirror.com` 是否可达。

## D1：环境

```bash
conda env create -f server/environment.yml
conda activate reca-audio
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability(0))"
```

期望：`2.5.1 12.1 (7, 0)`。

若 conda 解依赖太慢或失败，用 pip 兜底：

```bash
conda create -n reca-audio python=3.10 -y
conda activate reca-audio
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.26.4 librosa==0.10.2 soundfile transformers datasets accelerate timm pyyaml tqdm tensorboard
```

## V100 的三条铁律

1. **永不 bf16**：`torch.autocast("cuda", dtype=torch.float16)` + `GradScaler`。
2. **不装 flash-attn / xformers**：没有 sm70 内核，装了也会在运行时崩。
3. **不装 cu128/cu130 轮子**：官方已不带 sm70 内核，装了 `cuda.is_available()` 也可能为 False。

## 同步方式

推荐 git（私有仓库或服务器上的裸仓库）：

```bash
# 本地
git add -A && git commit -m "..." && git push
# 服务器
git pull
```

没有 git 就用 rsync（在本地 PowerShell 里执行）：

```powershell
rsync -avz --exclude .venv --exclude data ./ user@server:/path/to/reca-audio/
```
