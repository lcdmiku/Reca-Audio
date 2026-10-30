# RecA-Audio

把 *Reconstruction Alignment*（[arXiv:2509.07295](https://arxiv.org/abs/2509.07295)）迁移到**音频生成后训练**的课程项目。

核心问题不是"能不能涨点"，而是：**音频理解编码器里还剩多少可重建的声学信息，这个信息量如何决定迁移收益**——预期是一条倒 U 曲线。

## 文档

| 文件 | 内容 |
|---|---|
| [RecA-Audio-实验思路.md](RecA-Audio-实验思路.md) | 完整思路、假设、机制分析 |
| [RecA-Audio-执行方案-课程报告版.md](RecA-Audio-执行方案-课程报告版.md) | 课程报告版：实验清单与报告结构 |
| [RecA-Audio-两周冲刺方案.md](RecA-Audio-两周冲刺方案.md) | 两周冲刺：V100 约束与日计划 |
| [执行清单-D0到D14.md](执行清单-D0到D14.md) | 逐项待办 |
| [server/README.md](server/README.md) | 服务器侧环境与约定 |

## 快速开始（服务器）

```bash
# 0) 环境（克隆已有的 dl 环境，复用其中能在 V100 上跑的 torch）
bash server/setup_env.sh
conda activate reca-audio

# 0.1) HuggingFace 走镜像（服务器上 huggingface.co 不通）
export HF_ENDPOINT=https://hf-mirror.com

# 1) 路径
export DATA_ROOT=/path/to/data/reca-audio
export RUN_ROOT=/path/to/runs/reca-audio

# 2) 体检（第一次必跑，把输出贴回来）
bash server/00_check_env.sh

# 3) 下载数据（FSD50K ~80h 音效 + FMA-small ~66h 音乐，约 26 GB）
bash scripts/00_download_data.sh all

# 4) 预处理（先冒烟 200 个）
LIMIT=200 bash scripts/01_preprocess.sh
bash scripts/01_preprocess.sh
```

预处理后应得到：

```
$DATA_ROOT/processed/fsd50k/index.jsonl
$DATA_ROOT/processed/fsd50k/manifest.json
$DATA_ROOT/processed/fma/index.jsonl
$DATA_ROOT/processed/fma/manifest.json
```

## 仓库结构

```
├── src/reca_audio/
│   └── data/preprocess.py      # 格式归一 + 响度归一 + 去重
├── scripts/
│   ├── 00_download_data.sh     # FSD50K / FMA-small
│   └── 01_preprocess.sh
├── server/
│   ├── environment.yml         # V100 专用（cu121, python 3.10）
│   ├── 00_check_env.sh
│   └── README.md
└── docs 见上表
```

## V100 三条铁律

1. 永不 `bfloat16`，用 `float16` + `GradScaler`
2. 不装 `flash-attn` / `xformers`
3. 不装 cu128+ 的 torch（不带 sm70 内核）

## 数据许可

- FSD50K：CC-BY（部分 clip 为 CC-BY-NC / Sampling+，见 `dev_clips_info_FSD50K.json`）
- FMA-small：CC-BY / 部分 CC-BY-NC-SA，见官方 metadata

仅用于课程研究，不再分发原始音频。
