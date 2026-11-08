# 评测结果

> 全部数字由 `runs/Benchmark/` 下的脚本在服务器 `Reca` 上实测产生。
> 原始产物在服务器 `runs/Benchmark/output/<benchmark>/`（数据与权重不入库）。
> **不同协议/提示词的分数不可并排比较**，每条结果都标注了可比性。

## 指标方向约定

表头箭头表示**该指标的“变好方向”**，不是数值增减：

- **↓ = 越小越好**
- **↑ = 越大越好**

| 指标 | 含义 | 方向 |
|---|---|---|
| CLAP-FAD | 生成分布与真实分布在 CLAP 嵌入下的 Fréchet 距离 | **↓ 越小越好** |
| KL | PANNs 后验分布与参考分布的 KL 散度 | **↓ 越小越好** |
| IS | Inception Score（多样性 + 可判别性） | **↑ 越大越好** |
| FD | PANNs 嵌入下的 Fréchet 距离 | **↓ 越小越好** |
| CLIP / CLAP score | 文本-音频对齐相似度 | **↑ 越大越好** |
| MMSU / MMAU 准确率 | 选择题命中率 | **↑ 越大越好** |
| WER / CER | 词/字错误率 | **↓ 越小越好** |

表中**加粗 = 该列最佳值**（仅在多行对比表里使用）。

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
| Overall | 59.70 % | 5000 |
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
| Overall | 57.70 % | 1000 |
| sound | 66.97 % | 333 |
| speech | 55.56 % | 333 |
| music | 50.60 % | 334 |
| easy / medium / hard | 57.59 / 61.67 / 48.73 | 224 / 540 / 236 |

无预测：0 条。耗时：约 6 分钟（4 分片 × 2 卡）。

### 与论文对比

| | 论文（Audio-Omni） | 本次 |
|---|---|---|
| MMAU | **63.30** | 57.70 |

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
| **AudioCaps** | 1000 / 4411 | 0.2230 | 2.196 | 7.706 | 43.83 | 0.4575 |
| **MusicCaps** | 1004 / 1004 | 0.3140 | 1.325 | 3.386 | 18.55 | 0.3909 |

参照值（同一 CLAP 后端）：**真实 AudioCaps 音频 vs 其 caption = 0.5306**（300 条）。
生成 0.4575 比真实数据低 0.073（相对 ~14%），属于“可用但未饱和”，不是塌缩。

> 早期版本报过 KL≈0.001、IS≈1.00 —— 那是**指标实现 bug**：PANNs 输出是 527 类多标签
> sigmoid，代码错误地做了 softmax，把尖峰后验压平。改成 L1 归一化后即为上表数值。

### 3.3 指标实现说明（KL / IS / FD）

- PANNs：`panns_inference` 的 CNN14，clipwise 后验 + 2048 维 embedding；
- KL / IS：对后验做 **L1 归一化**（不是 softmax），KL 用 `p_gen` 与参考集均值分布；
- FD：PANNs embedding 上的 Fréchet 距离；
- CLAP-FAD：`laion_clap` 630k-audioset 的 512 维音频嵌入；
- 所有嵌入均**分批**计算（一次 32 条），否则 1000+ 条会 OOM。

生成耗时：AudioCaps 1000 条约 95 分钟；MusicCaps 1004 条约 100 分钟（4 分片并行）。

### 3.4 VGGish-FAD 状态：未跑通

`fadtk` 的 VGGish 通过 `torch.hub.load('harritaylor/torchvggish', 'vggish')` 取权重，
服务器无法访问 GitHub（`RemoteDisconnected`），因此 **VGGish-FAD 缺失**。

补齐只需两步（拿到文件后即可启用 `--vggish`）：

1. `vggish-10086976.pth`、`vggish_pca_params-970ea276.pth`
   放到 `~/.cache/torch/hub/checkpoints/`；
2. `harritaylor/torchvggish` 源码放到 `~/.cache/torch/hub/harritaylor_torchvggish_main/`。

### 3.5 可比性（重要）

| 项 | 论文 | 本次 |
|---|---|---|
| AudioCaps 数据 | 官方 test 975 条 | 1000 条（镜像 4411 条里取前 1000） |
| MusicCaps 数据 | 官方 test | 1004 条（镜像全量） |
| FAD 嵌入 | VGGish（T2A_FAD 1.86） | CLAP + PANNs，数值不可比 |

结论：这两行数字是**同一协议下的基线**，用于后续 RecA 后训练模型的前后对比；
与论文 1.86/1.94 之间隔着数据集、嵌入、参考集三重差异，不能直接比大小。

### 3.6 复现

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

## 4. 编辑类与 TTS（已移除）

评测代码与占位脚本已删除：缺 items.jsonl / 参考音频，无法产出指标。
论文参考值：AE_FAD 3.27 / AE_LSD 2.27 / CLAP 0.32；TTS_WER 1.77。

## 5. ReCA 后训练实验

> ⚠️ **2026-11-05 更新：本节 5.2–5.5 记录的 v3 / b2 / b3 / b4 全部作废。**
> 复查发现 Reconstruct prompt 缺少 `<|AUDIO|>` 占位符，processor 只产出 1 个文本 token，
> 理解塔**从未看到音频**——不同音频得到的条件特征余弦相似度恒为 1.0000。
> 这些轮次实际训练的是“无音频条件”的模型，loss 平台与 T2A 退化都由该 bug 解释。
> 修复（`<|AUDIO|>` + 去掉 `p.sum()>0` 过滤）后按 5.6 重训 b6。
> 以下数据保留仅作过程记录。

### 5.1 方法

冻结 VAE 与理解塔，只训练 **DiT（3037.3 M 参数）**：

- 条件：同一段音频经 Qwen2.5-Omni thinker，`build_prompt("Reconstruct", 随机模板)`；
  模板来自 `runs/scripts/const.py` 的 357 条，保证多样性；
- 防泄漏：`audio_input_prompt`（mel 条件）固定传零，避免模型直接从 mel 抄声学细节；
- 目标：rectified flow，`x_t=(1-t)·x1 + t·ε`、`v=ε−x1`，与 `sample_discrete_euler` 一致；
- **信息瓶颈**（模仿原文“输入降分辨率、输出不变”）：只对理解塔输入做低通与随机截断，
  重建目标始终是完整 44.1 kHz、未压缩的原始音频。

### 5.2 训练轮次

| 代号 | 步数 | lr | 精度 | 瓶颈 | 优化器 | 末段 loss | 状态 |
|---|---|---|---|---|---|---|---|
| v3 | 600 | 1e-5 | fp16 | 无 | Adafactor | 1.066 | ❌ 作废 |
| b2 | 300 | 2e-5 | fp32 | 低通 2× + 3 s | Adafactor | ~1.00 | ❌ 作废 |
| b3 | 3000 | 2e-5 | fp32 | 低通 2× + 3 s | Adafactor | 1.061 | ❌ 作废 |
| b4 | 2000 | 2e-5 | fp32 | 低通 4× + 1.5 s | Adafactor | 1.059 | ❌ 作废 |
| **b6** | 2000 | 2e-5 | fp32 | 低通 4×、整段 5 s | Adafactor | 见 5.6 | ✅ 有效（修复后） |

参考：**预训练模型在同一瓶颈配置下 loss = 1.07 ± 0.04**（8 个种子）。
所有轮次的 loss 都在 1.0–1.07 附近徘徊，没有实质下降。

详细分段：

| 轮次 | 0–500 | 500–1000 | 1000–1500 | 1500–2000 | 2000–2500 | 2500–3000 |
|---|---|---|---|---|---|---|
| b3 | 1.067 | 1.072 | 1.072 | 1.069 | 1.060 | 1.061 |
| b4 | 1.067 | 1.067 | 1.059 | 1.059 | — | — |

### 5.3 评测结果

**AudioCaps（1000 条，T2A）**

| 模型 | CLAP-FAD ↓ | KL ↓ | IS ↑ | FD ↓ | CLIP ↑ |
|---|---|---|---|---|---|
| 基线（原始 ckpt） | **0.2230** | 2.196 | **7.706** | **43.83** | **0.4575** |
| **b6（2000 步，修复后）** | 0.2533 | **2.002** | 6.215 | 44.51 | 0.4166 |
| v3（600 步）❌ 作废 | 0.2228 | 2.182 | 7.626 | 43.01 | 0.4613 |
| b3（3000 步，瓶颈 2×/3 s）❌ 作废 | 0.2794 | — | — | — | 0.3606 |
| b4（2000 步，瓶颈 4×/1.5 s）❌ 作废 | 评估中止 | — | — | — | — |

**b6 vs 基线**：CLAP-FAD 0.2230 → **0.2533（变差 0.030）**，CLIP 0.4575 → **0.4166（变差 0.041）**，
IS 7.71 → 6.21（变差），KL 2.196 → **2.002（变好）**，FD 43.83 → 44.51（基本持平）。

**MusicCaps（1004 条，T2M）**

| 模型 | CLAP-FAD ↓ | KL ↓ | IS ↑ | FD ↓ | CLIP ↑ |
|---|---|---|---|---|---|
| 基线 | 0.3140 | **1.325** | 3.386 | **18.55** | 0.3909 |
| v3（600 步）❌ 作废 | 0.3018 | 1.337 | 3.409 | 18.64 | 0.3923 |

### 5.4 关键发现

0. **决定性 bug：音频条件从未生效（2026-11-05 发现）**
   - `build_prompt("Reconstruct", …)` 的文本里没有 `<|AUDIO|>` 占位符，Qwen processor 只产出
     1 个 token（带占位符是 251 个，其中 250 个音频 token）；
   - 结果：不同音频的条件特征余弦相似度 = **1.0000**，修复后 = **0.4196**；
   - 叠加 `conditioners.py` 用 `p.sum() > 0` 过滤音频（真实波形零均值，实测 **77%** 的 AudioCaps
     片段和值为负会被丢弃），音频条件彻底失效。
1. **两个已修的 bug（本轮之前）**
   - 提示词少了 `build_prompt("Reconstruct", …)` 包装 → 理解塔输出 OOD 特征 → 生成白噪声
     （频谱平坦度 0.600 vs 正常 0.032，CLIP −0.048）；
   - flow 方向写反（用了 `x1−ε` 而不是 `ε−x1`）→ loss 在降、采样全是噪声。
     证据：原始模型在两种目标下 loss 分别为 **3.04 / 0.98**。
2. **fp16 主权重吃掉了权重更新**：每步 |Δw| 仅 1.7e-7，而 fp32 为 6.1e-6（36×）。
3. **lr 1e-4 过大**：loss 从 1.10 涨到 1.31（高于预训练 1.07），已中止。
4. 作废轮次里的 T2A 退化（FAD 0.223→0.279、CLIP 0.458→0.361）与 loss 平台，
   主因都是第 0 条；修复后需重测才能判断纯 ReCA 是否真的损害文本生成。

### 5.5 结论（修复后重估）

- 旧的“纯 ReCA 损害 T2A / loss 平台”结论**不成立**：那几轮的条件根本没有音频信息；
- 修复后 b6 的重建自检：**cfg=1 时平坦度 0.037–0.094**（原始模型 0.032），
  而 **cfg=7 时退化到 0.125–0.262** —— 重建时开 CFG 会把音频条件一起 null 掉，
  与 ReCA 的设定冲突，实测印证；
- b6（修复后 2000 步）的 T2A 结果：**CLAP-FAD 0.2230 → 0.2533、CLIP 0.4575 → 0.4166、IS 7.71 → 6.21**
  都变差，只有 **KL 2.196 → 2.002** 变好。说明纯重建对齐（λ_T2A=0）确实把生成头往重建方向拉，
  牺牲了文本-音频对齐与多样性——与 ReCA 原文强调的混合训练（λ_RecA / λ_T2A 权衡）一致。

### 5.6 修复后重训（b6）与产物

配置：2000 步、lr 2e-5、fp32 主权重 + AMP、Adafactor、低通 4× 且**不裁剪**（整段 5 s）。

| 检查项 | 结果 |
|---|---|
| 重建平坦度 cfg=1（seed 0/1/2） | 0.094 / 0.037 / 0.089 |
| 重建平坦度 cfg=7（对照） | 0.262 / 0.125 / 0.236 |
| AudioCaps 1000 条（T2A） | CLAP-FAD 0.2533 / KL 2.002 / IS 6.215 / FD 44.51 / CLIP 0.4166 |

| 类型 | 路径 | 备注 |
|---|---|---|
| 权重 | `Audio-Omni/model/model_reca_b6.ckpt` | 33 GB |
| 训练日志 | `runs/train/reca_b6/train_log.jsonl` | 保留 |
| 评测结果 | `runs/Benchmark/output/AudioCaps_b6/` | 已完成 |
| 训练代码 | `src/reca_audio/train/` | 说明见 [reca_training.md](reca_training.md) |
| 旧轮次产物 | 已删除 | 为省磁盘删除；日志仍在 `runs/train/reca_*/train_log.jsonl` |

### 5.7 Matrix v1：Pure ReCA / T2A-only / Mix 1:1（2026-11-07）

本轮从同一个原始 Audio-Omni checkpoint 出发，三组都训练 2000 step。训练集是按音频
SHA256 分组后的 3530 行（707 个唯一音频）；内部 test 是 445 行（89 个唯一音频），与
train/val 的 sample ID 和音频 SHA256 交集均为 0。`task_mode=mix` 每个 batch 只采样一个
任务，最终实际计数为 ReCA 1004 / T2A 996，不是每个 batch 同时计算两项 loss。

生成评测统一使用 100-step ODE、CFG=6、fp16、seed=1234，以及 3 个双卡 shard（GPU
0--5）。所有模型使用相同 prompts、分片和 seed 规则；8 个 case 均无生成失败。

**AudioCaps held-out（445 条，越低：FAD/KL/FD；越高：IS/CLAP）**

| 模型 | CLAP-FAD ↓ | KL ↓ | IS ↑ | FD ↓ | CLAP ↑ |
|---|---:|---:|---:|---:|---:|
| Base | **0.3514** | 2.1666 | **7.0272** | 64.0793 | **0.4610** |
| Pure ReCA | 0.3839 | 2.0406 | 6.1474 | **61.2357** | 0.4310 |
| T2A-only | 0.4287 | 2.0330 | 5.8390 | 65.1342 | 0.3945 |
| Mix 1:1 | 0.4277 | **2.0213** | 5.9148 | 61.8488 | 0.3894 |

**MusicCaps（1004 条）**

| 模型 | CLAP-FAD ↓ | KL ↓ | IS ↑ | FD ↓ | CLAP ↑ |
|---|---:|---:|---:|---:|---:|
| Base | 0.3238 | 1.3412 | 3.4425 | 18.5154 | 0.3882 |
| Pure ReCA | 0.2723 | 1.3633 | 3.5994 | 17.5523 | 0.4117 |
| T2A-only | 0.3208 | **1.2922** | 3.2923 | 18.7068 | 0.4135 |
| Mix 1:1 | **0.2611** | 1.3843 | **3.6931** | **16.2328** | **0.4147** |

**Held-out reconstruction 诊断（24 条）**

该诊断严格匹配训练数据通路：目标为原始 5 秒音频；Qwen 输入先从 16 kHz 降到 4 kHz
再升回 16 kHz；`audio_input_prompt` 仍是模型固定要求的 10 秒静音；采样 100 step、
CFG=1。24 条是 held-out test 中确定性排序后的前 24 条，因此只用于模型间配对诊断，
不作为总体 benchmark 置信区间。

| 模型 | 配对 CLAP cosine ↑ | CLAP-FAD ↓ | LSD ↓ |
|---|---:|---:|---:|
| Base | 0.1483 ± 0.1477 | 1.3833 | 63.0893 |
| Pure ReCA | **0.2160 ± 0.1205** | **1.3465** | 61.7886 |
| T2A-only | 0.1585 ± 0.1274 | 1.4005 | **60.9439** |
| Mix 1:1 | 0.1950 ± 0.1333 | 1.3675 | 61.5467 |

**源码事实与可支持的结论：**

- Pure 和 Mix 的配对 CLAP reconstruction 都高于 Base，说明 reconstruction alignment
  确实发生；Mix 的 alignment 弱于 Pure，符合它只有约一半 ReCA 更新。
- 在 AudioCaps held-out 上，Pure 的 CLAP-FAD 更高、IS/CLAP 更低；Mix 不但没有保护
  文本条件能力，CLAP 还比 Pure 更低。因此“1:1 T2A replay 足以阻止破坏”的假设被否定。
- T2A-only 同样在 AudioCaps held-out 上明显退化，说明问题不能只归因于 audio hidden
  states 的 acoustic shortcut；全量 DiT 持续训练造成的参数漂移、训练分布重复和目标形式
  也可能是主要因素。
- MusicCaps 上 Pure/Mix 出现相反的正迁移，Mix 在除 KL 外的四项指标最佳。因此本轮不是
  “生成能力整体下降”，而是明显的域依赖 trade-off；不能用单一 AudioCaps 或训练 loss
  概括整体能力。
- 下一组最小实验应优先约束相对 Base 的参数漂移（更低 LR/更短训练、L2-SP/teacher
  distillation，或仅训练 cross-attention/projection），而不是只继续提高 T2A batch 比例。

产物：

- 生成结果：`runs/Benchmark/output/Matrix_v1_eval3_6gpu_9ce3984/`
- 重建诊断：`runs/Benchmark/output/Reconstruction_v4_6gpu_9ce3984/`
- 训练日志：`runs/train/reca_matrix_v1/{pure,t2a,mix}/train_log.jsonl`

### 5.8 Gradient accumulation 受控实验：单测、smoke 与静态诊断（2026-11-07）

本阶段只验证诊断和等曝光 A/B 基础设施，**尚未启动 400 micro-batch A/B**。统一 LR
曲线按 micro-batch 曝光量索引：总曝光 400、warmup 20、peak LR `2e-5`；B 每次 update
使用四个曝光位置 LR 的平均值。Adafactor 的 `relative_step=False`、
`scale_parameter=False`，内部没有 relative-step 调度。

本地 23 项聚焦测试通过。服务器 smoke 使用同一份 4-record schedule（SHA256
`67f8c82f...eedc6`），A 完成 4 次 update，B 完成 1 次 update：

| Arm | 曝光 | updates | 实际 LR | pre-clip grad norm | 全局 drift（曝光 4） |
|---|---:|---:|---|---|---:|
| A / accum 1 | 4 | 4 | 2e-5 / 1.5e-5 / 5e-6 / 0 | 2.455 / 1.960 / 2.214 / 2.832 | 0.002328 |
| B / accum 4 | 4 | 1 | **1.0e-5**（四位置均值） | 1.222 | 0.000942 |

所有 smoke update 的 pre-clip norm 都大于阈值 1，因此都触发 gradient clipping。两臂
均完成 exposure 0/4 固定诊断；checkpoint 保存、reload 校验成功并立即删除，没有保留
smoke 权重。smoke 只用 1 个诊断样本，variance 必然为 0，不能用于判断 accumulation
是否降低方差。

原始 checkpoint 的正式静态诊断使用 held-out test 前 8 个样本，同一音频分别计算
ReCA/T2A 梯度，不创建 optimizer、没有 `optimizer.step()`：

| 指标 | ReCA | T2A |
|---|---:|---:|
| mean loss | 1.01485 | 1.00109 |
| mean sample gradient norm | 3.52132 | 3.66839 |
| mean gradient norm | 1.43977 | 1.64179 |
| gradient variance trace | 13.48325 | 17.49572 |

同一样本 ReCA/T2A 梯度 cosine 的均值为 **0.74650**，8 个样本上整体为正，因此在原始
checkpoint 处没有观察到强烈的全局反向冲突；但这不能推出训练过程中始终无冲突。
逐 block cosine 呈明显深度结构：早中层最低的是 layer 5（0.527）、layer 3（0.556）、
layer 6（0.563），后层 layer 30--35 大约为 0.91--0.94。ReCA/T2A 的归一化方差量级
接近（全局约 6.5 倍 mean-gradient squared norm），说明小 batch 的样本梯度噪声很大，
但仅凭静态结果还不能证明 accumulation=4 会改善最终 T2A。

产物：

- smoke：`runs/train/reca_gradient_ab_smoke_02cf985/`
- 静态诊断：`runs/train/reca_gradient_static_02cf985/static/metrics.jsonl`
- 两处均无模型权重；完成后 `/home/linchudeng` 仍约 350G。

### 5.9 Equal-exposure A/B：accumulation 1 vs 4（2026-11-07）

正式小规模实验消费同一份 400-record schedule（SHA256
`daf5abc2...119298`），实际任务数为 ReCA 211 / T2A 189。A 使用 accumulation 1、
执行 400 次 optimizer update；B 使用 accumulation 4、执行 100 次 update。因此这是
**相同数据预算下不同优化制度**的比较，不是相同 update 数的比较。

合同和产物检查均通过：固定诊断位于 exposure 0/100/200/400；checkpoint 对应 A 的
step 100/200/400 和 B 的 step 25/50/100，均成功 reload。中间 checkpoint 已删除，只保留
exposure 400 最终权重。

**训练更新稳定性**

| Arm | update 数 | pre-clip norm 均值 | norm variance | norm CV | clipping rate | 平均 clip scale |
|---|---:|---:|---:|---:|---:|---:|
| A / accum 1 | 400 | 2.1728 | 0.7526 | 0.3993 | 99.75% | 0.5145 |
| B / accum 4 | 100 | **1.1859** | **0.1122** | **0.2825** | **66.0%** | **0.8596** |

B 的 update-norm variance 比 A 低约 85%，CV 低约 29%，并把裁剪率从 99.75% 降到
66%。因此 accumulation 4 明显减少了实际 optimizer update 的幅度波动和 clipping
饱和；这支持增大 effective batch 用于后续训练稳定性，但其中也包含“四个梯度先平均”
这一机制本身，不能解释为数据分布噪声消失。

**固定 held-out 诊断（8 条）**

| Exposure | Arm | ReCA loss | T2A loss | ReCA variance | T2A variance | R/T cosine | global drift |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | A/B | 1.01485 | 1.00109 | 13.483 | 17.496 | 0.7465 | 0 |
| 100 | A | 1.00364 | 0.99949 | 8.139 | 7.743 | 0.8348 | 0.013512 |
| 100 | B | **0.98170** | **0.98215** | 9.972 | 9.138 | 0.8218 | **0.007478** |
| 200 | A | 1.00729 | 1.01051 | 5.752 | 8.781 | 0.8883 | 0.017012 |
| 200 | B | **0.98775** | **0.98648** | **4.674** | **5.237** | 0.8704 | **0.009453** |
| 400 | A | 0.98321 | 0.98402 | 4.407 | 4.555 | **0.8963** | 0.017635 |
| 400 | B | **0.97995** | **0.97956** | **4.297** | **4.471** | 0.8897 | **0.009804** |

两臂的绝对固定批次 variance 都随梯度整体变小而下降；但以 mean-gradient norm²
归一化后，最终 ReCA/T2A variance 分别为 A 的 6.64/6.89、B 的 7.00/6.92，B 并没有
降低相对样本梯度方差。因此需要区分：**B 明显稳定了四样本平均后的 update，但没有证据
表明单样本梯度分布本身变得更一致。**

ReCA/T2A cosine 全程为正，且 A/B 都从 0.7465 上升到约 0.89，不支持“两个任务发生
全局反向梯度冲突”。冲突/差异主要集中在早中层；最终最低仍是 layer 5（A 0.661、
B 0.610）以及 layer 2--6。参数漂移最大则集中在 layer 30--35，其次是 layer 0--2。

B 最终 global drift 为 0.009804，比 A 的 0.017635 低约 44%。同时 B 的固定 T2A flow
loss 不劣于 A，但该 loss 不是 T2A 生成指标；在对两个最终 checkpoint 运行相同 T2A
生成 benchmark 前，不能推出 B 保留了更多文本生成能力，也不能建立 drift 与 AudioCaps
指标下降之间的相关性。

产物：

- `runs/train/reca_gradient_ab_v1/a/dit_exposure0400.pt`（约 12.16 GB）
- `runs/train/reca_gradient_ab_v1/b/dit_exposure0400.pt`（约 12.16 GB）
- `runs/train/reca_gradient_ab_v1/{a,b}/metrics.jsonl`
- 完成后 `/home/linchudeng` 约 373G，GPU 与实验进程均已释放。

### 5.10 Equal-exposure A/B：89-clip held-out T2A（2026-11-08）

为回答“更稳定、漂移更小的 accumulation 4 是否保留更多文本生成能力”，从 SHA 隔离的
AudioCaps test 分区按 manifest 首次出现顺序，每个唯一音频 SHA256 只保留一条 caption，
得到 89 条确定性诊断集。prompt manifest SHA256 为
`006c0fbb...1a2401e`；Base/A/B 共用 prompt 顺序、分片和 seed。生成统一使用 100-step
ODE、CFG=6、10 秒、fp16、seed=1234，以及 GPU 1--6 上的三个双卡 shard。

A/B 都是同一条 400-micro-batch Mix 序列（ReCA 211 / T2A 189）训练出的 checkpoint，
不是 Pure ReCA。A 执行 400 次 accumulation-1 update；B 执行 100 次 accumulation-4
update。两者数据曝光量相同，但 optimizer update 数不同。

| 模型 | CLAP-FAD ↓ | KL ↓ | IS ↑ | FD ↓ | CLAP ↑ |
|---|---:|---:|---:|---:|---:|
| Base | **0.42515** | 2.19518 | **6.84548** | **70.65282** | **0.46226** |
| A / accum 1 | 0.52285 | 2.27633 | 6.47920 | 80.23866 | 0.39179 |
| B / accum 4 | 0.46687 | **2.13032** | 6.37711 | 77.04059 | 0.44044 |

相对 Base，A 的 CLAP 下降 0.07047、CLAP-FAD 增加 0.09770；B 的 CLAP 只下降
0.02181、CLAP-FAD 增加 0.04172。B 相对 A 的 CLAP 提高 0.04865，并将相对 Base 的
CLAP 损失缩小约 69%；CLAP-FAD、KL 和 FD 也都优于 A，但 IS 比 A 低 0.10209。因此
accumulation 4 并非所有指标一致占优，且 89 条上的分布指标方差较大。

**可支持的结论：**

- 在相同 400 次数据曝光下，B 的 update 更稳定、global drift 更低，同时 T2A CLAP
  明显更接近 Base；方向上支持“限制参数漂移有助于保留文本条件能力”。
- B 仍未恢复 Base：CLAP 低 0.02181，CLAP-FAD 和 FD 仍更差。因此增大 effective
  batch 只能缓解、不能消除短程 Mix 后训练的 catastrophic interference。
- B 的 KL 优于 Base，但 IS/FD/CLAP 没有同步改善；不能把单个分布指标解释为 T2A
  整体变好。
- 这里只有 A/B 两个训练制度，且 update 数也不同，不能建立 drift 与指标的统计相关性，
  也不能把差异唯一归因于“梯度方差降低”。
- 结果支持下一步在 accumulation 4 下单独测试更低 peak LR，而不是立刻启动长训练。

运行过程中先由 Linux dry-run 发现 Bash 特殊变量 `SECONDS` 会使输出时长漂移，正式运行
前改为 `AUDIO_SECONDS=10`。Base 首次评分又因未导出 `CLAP_CKPT` 而尝试联网下载并失败；
89 条 Base 音频完整保留，随后使用本地 1.8 GB CLAP 与 313 MB PANNs 缓存、offline 模式
断点续跑，没有重复生成 Base。最终三组均为 89/89，无残留 `/tmp` 合并 checkpoint。

产物：

- 评测：`runs/Benchmark/output/GradientAB_T2A_89_fc1a3eb/`（约 495 MB）
- Base/A/B 指标：各自子目录的 `results.json`
- 子集 provenance：`subset/selection.json` 与 `run_contract.json`
- 实际断点续跑版本：`182d9a2`；最终 hardened 启动器部署版本：`d55227a`
- A/B DiT checkpoint 各约 12 GB，继续保留；完成后用户目录仍约 373G。

## 6. 状态总览

| Benchmark | 状态 | 本次结果 | 论文 | 可比性 |
|---|---|---|---|---|
| MMSU | ✅ 全量 5000 | 59.70 % | 56.83 | 数据可比，提示词不同 |
| MMAU | ✅ test-mini 1000 | 57.70 % | 63.30 | 子集 + 提示词不同 |
| AudioCaps | ✅ 1000 条（T2A） | CLAP-FAD 0.2230 / KL 2.196 / IS 7.706 / FD 43.83 / CLAP 0.4575 | FAD 1.86 | 数据子集 + 嵌入不同，不可比 |
| MusicCaps | ✅ 1004 条（T2M） | CLAP-FAD 0.3140 / KL 1.325 / IS 3.386 / FD 18.55 / CLAP 0.3909 | FAD 1.94 | 同上 |
| ReCA 后训练 | ⚠️ 修复 bug 后重训 b6 | b6：CLAP-FAD 0.2533 / KL 2.002 / IS 6.215 / FD 44.51 / CLIP 0.4166（基线 0.2230 / 2.196 / 7.706 / 43.83 / 0.4575，除 KL 外均变差） | — | 自建协议，见第 5 节 |

## 7. 复现命令

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

## 8. 已移除的评测组件

为精简仓库结构，以下组件已删除（代码、数据与占位文档一并移除）：

| 组件 | 移除原因 |
|---|---|
| `runs/Benchmark/UltraEval-Audio` | 只覆盖 ASR/TTS，没有 T2A(FAD) 评测；接入后仅跑通 `sample` 冒烟与 LibriSpeech 基线，维护成本高于收益 |
| `runs/Benchmark/AudioEdit` | 缺 items.jsonl 与参考音频，无法产出指标 |
| `runs/Benchmark/TTS` | 缺 items.jsonl，且 WER 需要 Whisper large |

历史记录：UltraEval 上曾得到 **LibriSpeech test-clean WER 6.37 %（2619 条）**，
该结果随框架一并作废、不可复现，仅作过程记录保留。

## 9. 下一步

1. 在相同 400-micro-batch schedule 上增加一个 **accumulation 4、peak LR `1e-5`** 的
   小规模 C 组，其他因素保持与 B 相同；重点验证进一步降低 drift 能否缩小剩余的
   Base--B CLAP gap，仍不启动完整训练；
2. 若低 LR 仍不能保留 Base，再改为**只训部分参数**（如 cross-attn/projection）或加入
   L2-SP / Base teacher 约束，而不是继续单独提高 T2A batch 比例；
3. 在小规模配置通过后，再用 445 条 held-out AudioCaps 和 MusicCaps 做正式域内/域外验证；
4. 补齐 **VGGish-FAD**（需要 torch.hub 的权重与源码，见 3.4；服务器访问不到 GitHub）；
5. 用官方 inference 提示词重跑 MMSU/MMAU（若能拿到），消除提示词差异。
