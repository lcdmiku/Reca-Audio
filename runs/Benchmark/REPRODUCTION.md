# 复现状态与路线

> 结论：**现在还不能复现论文数字。** 代码骨架齐了，但数据、指标后端、协议对齐三块没补齐，
> 而且整条链路**一次都没在真机上跑通过**。

## 一、逐项状态

| Benchmark | 代码 | 数据 | 指标后端 | 跑通过 | 能复现论文？ |
|---|---|---|---|---|---|
| AudioCaps (T2A) | ✅ | ❌ 只有 10 条样本 | ⚠️ 默认 CLAP-FAD，非论文的 VGGish/PANNs | ❌ | 否 |
| MusicCaps (T2M) | ✅ | ❌ | ⚠️ 同上 | ❌ | 否 |
| MMSU | ✅ | ❌ 无 items.jsonl | ✅ 确定性判分 | ❌ | 否 |
| MMAU | ✅ | ❌ | ✅ | ❌ | 否 |
| AudioEdit | ✅ | ❌ | ⚠️ CLAP-FAD | ❌ | 否 |
| TTS (Seed-TTS en) | ✅ | ❌ | ⚠️ 默认 whisper-small，论文用 large 级 | ❌ | 否 |
| V2A / V2M | ❌ 未实现 | ❌ | ❌ | ❌ | 否 |

## 二、三个硬缺口

### 1. 数据（最大的一块）

| 需要的东西 | 现状 | 怎么补 |
|---|---|---|
| AudioCaps `test.csv`（975 条 caption） | 没有 | 官方发布物，`prepare --csv` |
| AudioCaps 测试音频（975 条 10s） | 没有 | YouTube/AudioSet 源，需自行获取 |
| MusicCaps `musiccaps-public.csv` + 音频 | 没有 | 同上 |
| MMSU / MMAU / AudioEdit 的 `items.jsonl` | 没有 | HF dataset，`prepare --dataset` |
| Seed-TTS en 测试集 | 没有 | 官方仓库 |

**没有参考音频就算不了 FAD**（FAD 是生成分布 vs 真实分布的 Fréchet 距离），只能算 CLAP score。

### 2. 指标后端

| 论文口径 | 我们默认 | 补齐方式 |
|---|---|---|
| KL / IS / FAD / FD on PANNs | CLAP 嵌入上的 Fréchet | `pip install panns-inference` + `--panns` |
| FAD (VGGish) | 无 | `pip install fadtk` + `--vggish` |
| TTS WER（Whisper-large 级） | whisper-small | `--asr-model openai/whisper-large-v3` |

**CLAP-FAD ≠ VGGish-FAD**，两者数值不可比。这是当前最容易误读的一点。

### 3. 协议对齐

| 项 | 论文 | 我们 |
|---|---|---|
| 采样步数 | 100 | 100（已对齐） |
| CFG | 6.0 | 6.0（已对齐） |
| 生成长度 | 10 s 评测 | 模型 `sample_size=485100` ≈ 11 s，未裁剪 |
| 精度 | 未说明（大概率 bf16/fp32） | 默认 fp16 DiT（V100 上没有 bf16） |
| 评测集划分 | 各自官方 test split | 取决于你 `prepare` 的文件 |

fp16 与长度差异会带来小幅偏移，不足以解释数量级差异，但**报告里必须写清**。

## 三、从零到“能出数字”的阶梯

### L0 链路自检（约 1 小时）

```bash
bash -n runs/Benchmark/run_shards.sh
LIMIT=2 STEPS=30 GPUS_PER_PROC=1 bash runs/Benchmark/run_shards.sh --benchmark AudioCaps
```
**验收**：`runs/Benchmark/output/AudioCaps/audio/` 出现 2 个 wav，`manifest.jsonl` 有 2 行。

### L1 指标自检（约 1 小时）

```bash
bash runs/Benchmark/download_models.sh
python -c "
from runs.Benchmark.common.metrics import clap_text_audio_score
" 2>/dev/null || echo "用脚本内的 score 子命令更稳"
```
**验收**：`score` 能跑出 `fad_clap` 与 `clip` 两个数（数值本身无意义，10 条样本）。

### L2 单个 benchmark 全量（约 1 天）

```bash
python runs/Benchmark/AudioCaps/evaluate.py prepare --csv /data/audiocaps/test.csv --id-col audiocap_id
REF=/data/audiocaps_test_audio GPUS_PER_PROC=2 \
  bash runs/Benchmark/run_shards.sh --benchmark AudioCaps
```
**验收**：975 条生成完毕，`results.json` 有 `fad_clap` / `clip`。
预计生成时间：4 分片 × 约 40 s/条 × 244 条/片 ≈ **2.5–3 小时**（V100 fp16）。

### L3 对齐论文口径（约 0.5 天）

```bash
pip install panns-inference fadtk jiwer
REF=... bash runs/Benchmark/run_shards.sh --benchmark AudioCaps
# 或手动：
python runs/Benchmark/AudioCaps/evaluate.py score --out runs/Benchmark/output/AudioCaps \
  --ref /data/audiocaps_test_audio --panns --vggish
```
**验收**：`results.json` 同时有 `fad_vggish`、`kl`、`is`、`fd`，才可与论文 Table 3 并排。

### L4 其余 benchmark（每个 0.5–1 天）

MMSU/MMAU → AudioEdit → TTS。V2A/V2M 需要先补代码和视频依赖。

## 四、不要做的事

- **不要**用 10 条样本的任何数字对外说“复现”；
- **不要**把 `fad_clap` 和论文的 `T2A_FAD=1.86` 并排比较；
- **不要**在没有官方 test split 的情况下跟论文比——分布不同，FAD 无意义。

## 五、最短可用路径（如果只有两天）

1. 第一天上午：L0 + L1，确认链路与显存；
2. 第一天下午：拿到 AudioCaps `test.csv` + 音频，L2 跑起来；
3. 第二天：L3 装 PANNs/VGGish 重打分，写清协议差异。

MMSU 可以并行做（不需要参考音频，判分确定性），是**最快能拿到一个真实数字**的分支。
