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

1. **ReCA 混合训练**：b6 已验证纯重建（λ_T2A=0）会让 T2A 变差（FAD +0.030、CLIP −0.041），
   下一步按 `TASK_MODE=mix LAMBDA_T2A=1` 重跑，验证“重建 + 文本各半”能否兼顾（代码已支持，未实跑）；
2. 若混合仍无收益，改为**只训部分参数**（如 cross-attn 投影层）做轻量对齐；
3. 补齐 **VGGish-FAD**（需要 torch.hub 的权重与源码，见 3.4；服务器访问不到 GitHub）；
4. 用官方 inference 提示词重跑 MMSU/MMAU（若能拿到），消除提示词差异；
5. 用 b6（修复 `<|AUDIO|>` 后重训）的结果替换第 5 节中作废的旧轮次。
