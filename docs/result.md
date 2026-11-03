# 评测结果

> 全部数字由 `runs/Benchmark/` 下的脚本在服务器 `Reca` 上实测产生。
> 原始产物在服务器 `runs/Benchmark/output/<benchmark>/`（数据与权重不入库）。
> **不同协议/提示词的分数不可并排比较**，每条结果都标注了可比性。

## 0. 环境

| 项 | 值 |
|---|---|
| 主机 | `210.28.133.13:22869`（Ubuntu 22.04，40 核 / 472 GB） |
| GPU | 8 × Tesla V100-SXM2-32GB（sm70） |
| 环境 | conda `audio-omni`：torch 2.6.0+cu124 / transformers 4.57.3 / numpy 1.23.5 |
| 模型 | `Audio-Omni/model/model.ckpt`（21 GB），DiT fp16 |
| 并行 | 组件并行（DiT+VAE 在 cuda:0，Qwen 理解塔在 cuda:1）+ 进程分片 |

实测显存：

```
[loader] cuda:0 allocated=6.70GB    # DiT fp16 + VAE
[loader] cuda:1 allocated=18.94GB   # Qwen2.5-Omni-3B（fp32）
```

## 1. MMSU（理解，全量 5000 条）

- **数据**：HF `ddwang2000/MMSU` 全部 3 个 parquet 分片（1667+1667+1666 = 5000）。
- **判分**：复刻官方 `mmsu_evaluation.py`（首字母 / 倒数第二字符；无法解析不计入分母）。
- **提示词**：四选一、要求只回字母（官方 inference 模板未公开）。

| 维度 | 准确率 | n |
|---|---|---|
| **Overall** | **59.70 %** | 5000 |
| Perception | 43.06 % | 2580 |
| — Linguistics | 46.18 % | — |
| — Paralinguistics | 38.22 % | — |
| Reasoning | 77.44 % | 2420 |
| — Linguistics | 83.07 % | — |
| — Paralinguistics | 42.39 % | — |

未解析回复：0 条。耗时：约 13 分钟（4 分片 × 2 卡）。

### 与论文对比

| | 论文（Audio-Omni） | 本次 |
|---|---|---|
| MMSU | 56.83 | **59.70** |
| Perception / Reasoning | 42.37 / 72.76（Qwen2.5-Omni-3B 行） | 43.06 / 77.44 |

差 +2.9 个点，主要来自提示词差异（我们要求“只回字母”，判分更容易命中）与解码差异；
**数据是全集，这一点是可比的**。

## 2. MMAU（理解，test-mini 1000 条）

- **数据**：HF `gamma-lab-umd/MMAU-test-mini`（`test_mini.parquet`，1000 条带答案）。
- **判分**：复刻官方 `evaluation.py` 的 token 集合匹配（答案 token 全覆盖 + 不含错误选项独有 token）。
- **提示词**：要求“只输出选项标签+原文，不要解释”（必须，否则长解释会带进错误选项 token）。

| 维度 | 准确率 | n |
|---|---|---|
| **Overall** | **57.70 %** | 1000 |
| sound | 66.97 % | 333 |
| speech | 55.56 % | 333 |
| music | 50.60 % | 334 |
| easy / medium / hard | 57.59 / 61.67 / 48.73 | 224 / 540 / 236 |

无预测：0 条。耗时：约 6 分钟（4 分片 × 2 卡）。

### 与论文对比

| | 论文（Audio-Omni） | 本次 |
|---|---|---|
| MMAU | 63.30 | **57.70** |

**不可直接比**，两点差异：

1. 我们只能跑公开答案的 **test-mini（1000 条）**，论文的 63.30 大概率是完整 test（9000 条，答案不公开）；
2. 提示词不同。

值得记录的中间现象：同一批 100 条，**宽松提示词（允许解释）只有 24%，改成“只输出标签+原文”后升到 72%**——
说明这套 token 匹配判分对输出格式极其敏感，任何对比都必须注明提示词。

## 3. 生成类：AudioCaps / MusicCaps（T2A / T2M 全量）

### 3.1 设置

- 采样：**100 步 ODE、CFG=6.0**（论文设置），DiT fp16，理解塔在 cuda:1；
- 并行：4 分片 × 2 卡（`GPUS_PER_PROC=2`），单条 ~21 s；
- 指标后端：**CLAP**（`laion_clap` 630k-audioset）+ **PANNs**（CNN14，KL/IS/FD）；
- 数据：
  - AudioCaps：Gitee `hf-datasets/OpenSound-AudioCaps`（LFS，41 个分片 41 GB）；
  - MusicCaps：HF `mahendra0203/musiccaps_processed`（audio + caption）。

### 3.2 结果

| 数据集 | n（生成/参考） | CLAP-FAD ↓ | KL ↓ | IS ↑ | FD ↓ | CLAP score ↑ |
|---|---|---|---|---|---|---|
| **AudioCaps** | 1000 / 4411 | **0.2230** | 0.001054 | 1.0010 | 43.83 | 0.4575 |
| **MusicCaps** | 1004 / 1004 | **0.3140** | 0.000805 | 1.0008 | 18.55 | 0.3909 |

生成耗时：AudioCaps 1000 条约 95 分钟；MusicCaps 1004 条约 100 分钟（4 分片并行）。

### 3.3 VGGish-FAD 状态：未跑通

`fadtk` 的 VGGish 通过 `torch.hub.load('harritaylor/torchvggish', 'vggish')` 取权重，
服务器无法访问 GitHub（`RemoteDisconnected`），因此 **VGGish-FAD 缺失**。

补齐只需两步（拿到文件后即可启用 `--vggish`）：

1. `vggish-10086976.pth`、`vggish_pca_params-970ea276.pth`
   放到 `~/.cache/torch/hub/checkpoints/`；
2. `harritaylor/torchvggish` 源码放到 `~/.cache/torch/hub/harritaylor_torchvggish_main/`。

### 3.4 可比性（重要）

| 项 | 论文 | 本次 |
|---|---|---|
| AudioCaps 数据 | 官方 test 975 条 | **1000 条**（镜像 4411 条里取前 1000） |
| MusicCaps 数据 | 官方 test | 1004 条（镜像全量） |
| FAD 嵌入 | VGGish（T2A_FAD 1.86） | CLAP + PANNs，**数值不可比** |

结论：这两行数字是**同一协议下的基线**，用于后续 RecA 后训练模型的前后对比；
与论文 1.86/1.94 之间隔着数据集、嵌入、参考集三重差异，不能直接比大小。

### 3.5 复现

```bash
cd ~/Reca-Audio
source ~/miniconda3/etc/profile.d/conda.sh && conda activate audio-omni
export QWEN_OMNI_MODEL_PATH=$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-3B/snapshots/*/ | head -1)
export CLAP_CKPT=~/.cache/laion_clap/630k-audioset-best.pt

# AudioCaps
python runs/Benchmark/AudioCaps/build_subset.py \
  --shards-dir runs/Benchmark/_data/OpenSound-AudioCaps/data --pattern "test-*.parquet" \
  --ids-csv runs/Benchmark/_data/audiocaps/test.csv \
  --ref-dir runs/Benchmark/_data/audiocaps/full2/ref \
  --out-jsonl runs/Benchmark/_data/audiocaps/full2/prompts.jsonl
INPUT=.../prompts1000.jsonl REF=.../full2/ref OUT=.../AudioCaps_full \
  GPUS_PER_PROC=2 bash runs/Benchmark/run_shards.sh --benchmark AudioCaps
python runs/Benchmark/AudioCaps/evaluate.py score --out .../AudioCaps_full --ref .../full2/ref --panns

# MusicCaps
python runs/Benchmark/AudioCaps/build_subset.py \
  --shards-dir runs/Benchmark/_data/musiccaps --pattern "train-*.parquet" \
  --id-cols youtube_id,start_time --max-items 1000 \
  --ref-dir runs/Benchmark/_data/musiccaps/ref \
  --out-jsonl runs/Benchmark/_data/musiccaps/prompts1000.jsonl
INPUT=.../prompts1000.jsonl REF=.../musiccaps/ref OUT=.../MusicCaps_full \
  GPUS_PER_PROC=2 bash runs/Benchmark/run_shards.sh --benchmark MusicCaps
python runs/Benchmark/MusicCaps/evaluate.py score --out .../MusicCaps_full --ref .../musiccaps/ref --panns
```

## 4. 编辑类（AudioEdit）与 TTS（Seed-TTS en）

- 代码就绪，**缺数据**；TTS 的 WER 还需 Whisper（当前默认 small，论文用 large 级）。
- 论文参考：AE_FAD 3.27 / AE_LSD 2.27 / CLAP 0.32；TTS_WER 1.77。

## 5. 状态总览

| Benchmark | 状态 | 本次结果 | 论文 | 可比性 |
|---|---|---|---|---|
| MMSU | ✅ 全量 5000 | **59.70 %** | 56.83 | 数据可比，提示词不同 |
| MMAU | ✅ test-mini 1000 | **57.70 %** | 63.30 | 子集 + 提示词不同 |
| AudioCaps | ✅ 1000 条（T2A） | **CLAP-FAD 0.2230 / KL 0.00105 / IS 1.0010 / FD 43.83 / CLAP 0.4575** | FAD 1.86 | 数据子集 + 嵌入不同，不可比 |
| MusicCaps | ✅ 1004 条（T2M） | **CLAP-FAD 0.3140 / KL 0.000805 / IS 1.0008 / FD 18.55 / CLAP 0.3909** | FAD 1.94 | 同上 |
| AudioEdit | ⛔ 缺数据 | — | AE_FAD 3.27 | — |
| TTS | ⛔ 缺数据 | — | WER 1.77 | — |

## 6. 复现命令

```bash
cd ~/Reca-Audio
source ~/miniconda3/etc/profile.d/conda.sh && conda activate audio-omni
export QWEN_OMNI_MODEL_PATH=$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-3B/snapshots/*/ | head -1)

# MMSU
python runs/Benchmark/MMSU/evaluate.py prepare --parquet "runs/Benchmark/_data/hf/mmsu_train*.parquet"
bash runs/Benchmark/run_shards.sh --benchmark MMSU

# MMAU
python runs/Benchmark/MMAU/evaluate.py prepare --parquet runs/Benchmark/_data/hf/mmau_test_mini.parquet
bash runs/Benchmark/run_shards.sh --benchmark MMAU
```

## 7. UltraEval-Audio（统一评测框架，已接入）

上游：https://github.com/OpenBMB/UltraEval-Audio（ACL 2026 demo），按官方
`docs/how eval your model.md` 接入。

### 接入方式

| 文件 | 内容 |
|---|---|
| `audio_evals/models/audio_omni.py` | 适配器：prompt 里的 audio → `AudioOmni.understand()` |
| `registry/model/audio_omni.yaml` | 注册 `--model audio-omni`（模型路径 + 双卡） |
| `INTEGRATION.md` | 接入说明与补丁记录 |

仓库用**稀疏检出**（完整 checkout 在 Windows 上因 `lib/Kimi-Audio/...` 路径过深失败），
只保留框架代码与 registry，未包含各模型私有实现。

### 验证

```bash
cd runs/Benchmark/UltraEval-Audio
python audio_evals/main.py --dataset sample --model audio-omni --limit 2
```

结果：`{'wer(%)': 82.76, 'fail_rate(%)': 0.0}`

- 第 0 条 CER 0%（输出与参考一致）；
- 第 1 条 CER 184%——模型在答案后追加对话文字，属于提示词遵循问题；
- 结果文件：`res/audio-omni/sample/<时间>.jsonl`。

### 环境补丁（重要）

| 问题 | 处理 |
|---|---|
| 环境是 `datasets==2.14.7`，框架要求 3.6.0 | `huggingface.py` 加 `trust_remote_code` 回退 |
| `jiwer` 4.x 删了 `compute_measures` | 降到 `jiwer==2.6.0` |
| `sacrebleu` 2.x 无 `TOKENIZERS` | 降到 `sacrebleu==1.5.1`（`--no-deps`） |
| `pycocotools` 需编译 | `coco.py` 惰性导入，仅 caption 指标不可用 |

### 已知限制

- 真实数据集需要 `HF_HUB_DOWNLOAD_TIMEOUT` / `HF_HUB_ETAG_TIMEOUT` 调到 300，否则 10 秒超时会误报下载失败；
- UltraEval 的评分口径与我们 `runs/Benchmark/MMSU` 的官方脚本**不同，不可混用**。

### 基线结果（2026-11-02）

| 数据集 | 指标 | 值 | n | 说明 |
|---|---|---|---|---|
| LibriSpeech test-clean | WER | **6.37 %** | 2619 | corpus-level，框架汇总 |
| LibriSpeech test-clean | 逐条 WER 均值 / 中位数 | 11.44 % / 0.00 | 2619 | 长尾拉高均值 |
| `sample`（冒烟） | CER/WER | 82.76 % | 2 | 样本太少，仅证明链路 |

细节：

- 27 条 WER > 100，主要是模型在转写后追加“如果还有类似音频…”这类对话尾巴；
- 102 条输出含对话尾巴，中位数为 0 说明大多数转写完全正确；
- 无空预测，`fail_rate = 0`。

**参考**：Qwen2.5-Omni 在同类报告里 test-clean 约 2.4%。这里 6.37% 的差距主要来自
输出格式（模型爱加对话尾巴）而不是识别能力——若把尾部裁掉，逐条均值会显著下降。
后续做 RecA 对比时，**必须保持同一提示词与同一套后处理**，否则数字不可比。

运行命令：

```bash
cd runs/Benchmark/UltraEval-Audio
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DOWNLOAD_TIMEOUT=300 HF_HUB_ETAG_TIMEOUT=300
export QWEN_OMNI_MODEL_PATH=$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-3B/snapshots/*/ | head -1)
python audio_evals/main.py --dataset librispeech-test-clean --model audio-omni
```

耗时约 70 分钟（2619 条，单进程，双卡组件并行）。

## 8. 下一步

1. 补齐 **VGGish-FAD**（需要 torch.hub 的权重与源码，见 3.3；服务器访问不到 GitHub）；
2. 给 AudioEdit / TTS 准备 items.jsonl；
3. 用官方 inference 提示词重跑 MMSU/MMAU（若能拿到），消除提示词差异；
4. 在 UltraEval 上补 test-other / dev-clean，形成完整 ASR 基线；
5. 修 RecA 实验：在同一批 T2A/T2M 数据上对比 base / RecA 后训练。
