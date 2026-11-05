# RecA-Audio

## Controlled ReCA/T2A experiment matrix

The guarded matrix launcher uses one SHA-grouped, deterministic 80/10/10
internal split. Create it once on the server; the command refuses to overwrite
an existing split directory:

```bash
python -m reca_audio.train.split_manifest \
  --manifest runs/Benchmark/_data/audiocaps/train_manifest.jsonl \
  --out runs/Benchmark/_data/audiocaps/reca_split_v1 \
  --seed 42
```

Then smoke-test, launch, and inspect the three trainable arms:

```bash
ROOT=runs/train/reca_matrix_smoke bash runs/scripts/train_reca_matrix.sh smoke
ROOT=runs/train/reca_matrix_v1 bash runs/scripts/train_reca_matrix.sh launch
ROOT=runs/train/reca_matrix_v1 bash runs/scripts/train_reca_matrix.sh status
```

The launcher assigns Pure ReCA to GPUs 0-1, T2A-only to GPUs 2-3, and the
1:1 stochastic ReCA/T2A mix to GPUs 4-5. It validates the split report,
checkpoint, GPU availability, and empty output directories before creating
anything. The unchanged Audio-Omni checkpoint is the Base arm and is evaluated
separately; it is not another training process.

把 **Reconstruction Alignment（ReCA, [arXiv:2509.07295](https://arxiv.org/abs/2509.07295)）** 迁移到
**Audio-Omni**（SIGGRAPH 2026：冻结 Qwen2.5-Omni-3B 理解塔 + 可训练 DiT）的音频生成后训练，
并搭一套可复现的音频理解 / 生成评测。

> 课程项目。核心问题：**音频理解编码器里还剩多少可重建的声学信息，这个信息量如何决定迁移收益。**

---

## 当前状态（2026-11-05）

| 模块 | 状态 | 位置 |
|---|---|---|
| Audio-Omni 推理（T2A / T2M / 编辑 / 理解） | ✅ 双卡组件并行，V100 可跑 | `Audio-Omni/` |
| 理解评测 MMSU / MMAU | ✅ 全量跑完：**59.70 % / 57.70 %** | `runs/Benchmark/{MMSU,MMAU}` |
| 生成评测 AudioCaps / MusicCaps | ✅ 基线：CLAP-FAD **0.2230 / 0.3140** | `runs/Benchmark/{AudioCaps,MusicCaps}` |
| ReCA 后训练 | ⚠️ b6 已完成：CLAP-FAD **0.2533** / CLIP **0.4166**（基线 0.2230 / 0.4575）——纯 ReCA 弱于基线，下一步混合训练 | `src/reca_audio/train/` |
| VGGish-FAD | ⛔ 缺权重（只有 GitHub 渠道） | — |

全部数字与协议说明见 **[docs/result.md](docs/result.md)**。

---

## 快速开始（服务器）

```bash
# 环境
conda activate audio-omni            # python 3.11 / torch 2.6.0+cu124 / transformers 4.57.3
cd ~/Reca-Audio

# 必需的环境变量
export QWEN_OMNI_MODEL_PATH=$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-3B/snapshots/*/ | head -1)
export HF_ENDPOINT=https://hf-mirror.com          # 服务器直连 huggingface.co 不通
export HF_HUB_DOWNLOAD_TIMEOUT=300 HF_HUB_ETAG_TIMEOUT=300
export CLAP_CKPT=~/.cache/laion_clap/630k-audioset-best.pt

# 一次生成类评测（4 分片 × 2 卡）
INPUT=$PWD/runs/Benchmark/_data/audiocaps/full/prompts1000.jsonl \
REF=$PWD/runs/Benchmark/_data/audiocaps/full2/ref \
OUT=$PWD/runs/Benchmark/output/AudioCaps_try \
  bash runs/Benchmark/run_shards.sh --benchmark AudioCaps

# 理解类评测
bash runs/Benchmark/run_shards.sh --benchmark MMSU     # 需先 prepare 出 items.jsonl
```

ReCA 训练：

```bash
SMOKE=1 bash runs/scripts/train_reca.sh        # 2 步冒烟
STEPS=2000 bash runs/scripts/train_reca.sh     # 单进程 fp32 主权重
bash runs/scripts/train_reca_ddp.sh            # 4 进程 × 2 卡（DDP，供参考）
```

---

## 仓库结构

```
RecA-Audio/
├── Audio-Omni/                    # 上游模型代码（含我们为 V100/RecA 做的补丁）
│   ├── audio_omni/                #   loader / conditioners / dit / api
│   └── model/                     #   Audio-Omni.json + model.ckpt(21G) + model_reca_b6.ckpt(33G)
├── src/reca_audio/
│   ├── data/preprocess.py         # 早期数据预处理（未用于当前流程）
│   └── train/                     # ★ ReCA 训练：数据、条件、flow loss、训练循环、合并
├── runs/
│   ├── Benchmark/                 # ★ 评测：MMSU / MMAU / AudioCaps / MusicCaps + common/
│   │   ├── _data/                 #   数据（gitignored）
│   │   └── output/                #   结果（gitignored）
│   ├── scripts/                   # 启停脚本：train_* / pipeline_* / deploy.* / const.py
│   └── train/                     # 训练日志与检查点（gitignored）
├── docs/                          # 见下方文档索引
└── pyproject.toml
```

大文件（`runs/Benchmark/_data`、`output`、`runs/train`、`*.ckpt`）都不入库。

---

## 文档索引

| 文档 | 用途 |
|---|---|
| [docs/result.md](docs/result.md) | **所有评测数字**、指标方向约定、可比性说明 |
| [docs/reca_training.md](docs/reca_training.md) | ReCA 训练设计、显存策略、踩过的坑 |
| [docs/deploy.md](docs/deploy.md) | 本地→服务器的部署流程（三个保险） |
| [docs/Audio-Omni-RecA-CHANGES.md](docs/Audio-Omni-RecA-CHANGES.md) | 对上游 Audio-Omni 的每一处改动 |
| [docs/Audio-Omni.txt](docs/Audio-Omni.txt) | Audio-Omni 论文纯文本 |
| `docs/RecA-Audio-*.md`、`docs/执行清单-D0到D14.md` | 早期选题与冲刺方案（历史材料） |

---

## 数据来源

| 数据 | 来源 | 本地位置 |
|---|---|---|
| AudioCaps（4411 条音频 + caption） | Gitee LFS `hf-datasets/OpenSound-AudioCaps` | `runs/Benchmark/_data/audiocaps/full2/` |
| MusicCaps（1004 条） | HF `mahendra0203/musiccaps_processed` | `runs/Benchmark/_data/musiccaps/` |
| MMSU / MMAU | HF parquet | `runs/Benchmark/_data/hf/` |

AudioCaps 的原始 41 GB 分片为省磁盘已删除，需要重建子集时按 `docs/result.md` 里的命令重新拉取。

---

## 关键坑（都已踩过，务必先读）

1. **`<|AUDIO|>` 占位符**：Reconstruct 的 prompt 必须含该 token，否则 processor 只产出 1 个文本 token，
   音频被静默丢弃（不同音频的条件特征完全相同）。`docs/reca_training.md` 有完整证据链。
2. **重建不能开 CFG**：`cfg_scale>1` 会把 cross-attn 条件整体置零，实测重建质量从平坦度 0.03–0.09
   退化到 0.13–0.26。T2A 才用 CFG。
3. **fp16 主权重会吃掉更新**：每步 |Δw| 约 1e-7，被 fp16 舍入吃掉；必须 fp32 主权重 + autocast。
4. **Reconstruct 必须用系统模板**：`build_prompt("Reconstruct", …)`，裸文本会让理解塔输出 OOD 特征。
5. **V100（sm70）**：只能 fp16，无 bf16 / FlashAttention / TF32。
6. **HF 镜像会超时**：默认 10 s 读超时，需 `HF_HUB_DOWNLOAD_TIMEOUT=300`。

---

## 给 agent / 新同学的入口

1. 先读 [docs/result.md](docs/result.md)（结论与数字）和 [docs/reca_training.md](docs/reca_training.md)（坑）。
2. **服务器代码只能通过部署流程同步**，规则见 [docs/deploy.md](docs/deploy.md)：
   本地是唯一真源，`runs/scripts/deploy.ps1` + `deploy.sh` 会校验 hash 并写 `DEPLOYED_COMMIT`。
   不要在服务器上直接改代码（曾因此把本地文件编码写坏）。
3. 约束：**不用 sudo**、只在 `/home/linchudeng/` 下操作、不要删除他人文件。
4. 评测协议各不相同，**不要跨协议比较数字**（MMSU/MMAU 用官方脚本，AudioCaps/MusicCaps 用自建 harness）。

## 许可

课程研究用途。Audio-Omni 模型与代码遵循上游许可；AudioCaps / MusicCaps / MMSU / MMAU
遵循各自数据集许可，不分发原始音频。
