# 评测结果

> 所有结果均由 `runs/Benchmark/` 下的脚本在服务器 `Reca` 上实测产生，
> 原始文件在 `runs/Benchmark/output/<benchmark>/`（服务器侧，不入库）。
> 与论文数字的可比性单独标注，**不要把不同协议的分数并排比较**。

## 0. 环境

| 项 | 值 |
|---|---|
| 主机 | `210.28.133.13:22869`（Ubuntu 22.04，40 核 / 472 GB） |
| GPU | 8 × Tesla V100-SXM2-32GB（sm70） |
| 环境 | conda `audio-omni`：torch 2.6.0+cu124 / transformers 4.57.3 / numpy 1.23.5 |
| 模型 | `Audio-Omni/model/model.ckpt`（21 GB），DiT fp16，理解塔 cuda:1 |
| 推理 | 理解类贪心解码；生成类 100 步 ODE、CFG=6.0（论文设置） |

显存实测（双卡组件并行）：

```
[loader] cuda:0 allocated=6.70GB    # DiT fp16 + VAE
[loader] cuda:1 allocated=18.94GB   # Qwen2.5-Omni-3B 理解塔（fp32）
```

---

## 1. MMSU（理解）

- **协议**：官方 `mmsu_evaluation.py` 判分规则（首字母 / 倒数第二字符），
  无法解析的回复不计入分母。
- **数据**：HF `ddwang2000/MMSU` 的 shard 0（1667 条），**200 条分层抽样**（stride=5）。
- **提示词**：四选一、要求只回字母（官方 inference 模板未公开，详见下方差异说明）。

| 维度 | 准确率 | n |
|---|---|---|
| **Overall** | **68.00 %** | 200 |
| Perception | 48.28 % | 87 |
| — Linguistics | 40.91 % | — |
| — Paralinguistics | 71.43 % | — |
| Reasoning | 83.19 % | 113 |
| — Linguistics | 83.19 % | — |

未解析回复：0 条。

### 与论文的差异（重要）

| 项 | 论文 | 本次 |
|---|---|---|
| Audio-Omni MMSU | **56.83** | 68.00 |
| 数据范围 | 全部 5000 条 | shard 0 前 1667 条中抽 200 |
| 提示词 | 未公开（各家自接模型） | 自定四选一模板 |

**结论：这个 68.00 不能当作复现论文的 56.83。** 差异来源按可能性排序：

1. **子集偏差**：只覆盖 shard 0，任务分布与全集不同；
2. **提示词不同**：我们明确要求"只回字母"，更容易被判分规则解析，也可能提升答题率；
3. 论文的数字来自他们自己的 inference 脚本。

要得到可比数字，必须跑满 5000 条并使用同一提示词——目前只下载了 1/3 的分片。

### 复现命令

```bash
cd ~/Reca-Audio
PY=~/miniconda3/envs/audio-omni/bin/python
export HF_ENDPOINT=https://hf-mirror.com

$PY runs/Benchmark/MMSU/evaluate.py prepare \
  --parquet runs/Benchmark/_data/hf/mmsu_train0.parquet --limit 200 --stride 5
$PY runs/Benchmark/MMSU/evaluate.py eval \
  --items runs/Benchmark/MMSU/items.jsonl --out runs/Benchmark/output/MMSU \
  --device cuda:0 --cond-device cuda:1 --dtype float16
$PY runs/Benchmark/MMSU/evaluate.py merge --out runs/Benchmark/output/MMSU
$PY runs/Benchmark/MMSU/evaluate.py score --out runs/Benchmark/output/MMSU
```

耗时：模型加载 ≈ 1.5 min，推理 200 条 ≈ 1.5 min。

---

## 2. MMAU（理解）

- 数据已下载（`runs/Benchmark/_data/hf/mmau_test_mini.parquet`，1.2 GB / 1000 条），
  schema 已确认：`context{bytes,path} / instruction / choices / answer / other_attributes`。
- **尚未运行**：`prepare` 目前只走 HF 流式加载，尚未加本地 parquet 分支（MMSU 已加）。
- 参考：论文 Audio-Omni MMAU = **63.30**。

---

## 3. 生成类（AudioCaps / MusicCaps）

- 代码就绪，**缺数据**：需要官方 `test.csv` + 真实测试音频（FAD 的参考分布）。
- 指标后端未装：论文用 PANNs（KL/IS/FAD/FD），当前默认 CLAP-FAD，**不可比**。
- 参考：论文 T2A_FAD 1.86、T2M_FAD 1.94。

## 4. 编辑类（AudioEdit）

- 代码就绪，**缺数据**（HF `HKUSTAudio/AudioEdit`）。
- 参考：论文 AE_FAD 3.27 / AE_LSD 2.27 / CLAP 0.32。

## 5. TTS（Seed-TTS en）

- 代码就绪，**缺数据**；WER 需要 Whisper（当前默认 small，论文用 large 级）。
- 参考：论文 TTS_WER 1.77。

---

## 6. 状态总览

| Benchmark | 状态 | 结果 | 可复现论文？ |
|---|---|---|---|
| MMSU | ✅ 已跑（200 条子集） | 68.00 % | 否（子集 + 提示词不同） |
| MMAU | ⏳ 数据已下，待接入 | — | — |
| AudioCaps | ⛔ 缺数据/指标后端 | — | 否 |
| MusicCaps | ⛔ 缺数据/指标后端 | — | 否 |
| AudioEdit | ⛔ 缺数据 | — | 否 |
| TTS | ⛔ 缺数据 | — | 否 |

## 7. 下一步

1. 下载 MMSU shard 1–2，跑满 5000 条，替换本文档中的子集数字；
2. 给 MMAU `prepare` 加本地 parquet 分支，跑 test-mini 1000 条；
3. 装 `panns-inference` / `fadtk`，把生成类指标对齐论文口径；
4. 拿到 AudioCaps/MusicCaps 音频后再开生成类评测。
