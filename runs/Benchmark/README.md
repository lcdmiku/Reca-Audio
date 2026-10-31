# Benchmark 评测

对应论文（*Audio-Omni*, SIGGRAPH 2026）第 5.2 节的评测协议。**源码里原本没有任何评测脚本**，
这里补齐；生成与打分分离，换指标不用重跑采样。

## 论文协议

| 任务 | 数据集 | 指标 | 推理设置 |
|---|---|---|---|
| 理解 | **MMSU**（47 项口语任务） | 准确率 | 贪心解码 |
| 理解 | **MMAU**（27 项推理技能） | 准确率 | 贪心解码 |
| 生成 | **AudioCaps**（T2A） | KL↓ / IS↑ / FAD↓ / FD↓ | 100 步 ODE，CFG=6.0 |
| 生成 | **MusicCaps**（T2M） | 同上 | 同上 |
| TTS | Seed-TTS en | WER↓ | 同上 |
| 编辑 | **AudioEdit**（add/remove/extract/style） | AE_FAD↓ / AE_LSD↓ / CLAP↑ | 同上 |

> 论文的 KL/IS/FAD/FD 都建立在 **PANNs** 嵌入上；FAD 另可用 VGGish 口径。
> 本目录默认用 **CLAP 嵌入**算 Fréchet 距离（`fad_clap`），因为 CLAP 是模型自带的依赖，
> 不需要额外下载权重。**它与论文的 VGGish-FAD 不可直接比较**；要对齐论文数值请装 `fadtk`
> 并加 `--vggish`，装 `panns-inference` 并加 `--panns`。

## 目录结构

```
runs/Benchmark/
├── common/                 # 共享：生成、切分、指标、数据导出
│   ├── audio_io.py
│   ├── generation.py
│   ├── metrics.py
│   ├── datasets.py
│   └── understanding.py
├── AudioCaps/evaluate.py   # T2A
├── MusicCaps/evaluate.py   # T2M
├── MMSU/evaluate.py        # 理解
├── MMAU/evaluate.py        # 理解
├── AudioEdit/evaluate.py   # 编辑
├── TTS/evaluate.py         # Seed-TTS en，WER
├── run_shards.sh           # 多卡并行启动器
└── README.md
```

> **未覆盖**：V2A（VGGSound）与 V2M（V2M-bench）。它们需要视频解码（`decord`/`av`/`mmaudio`），
> 且要额外加载 Synchformer，属于另一套依赖；需要时再按同样结构加 `runs/Benchmark/V2A/`。

## 统一数据格式

所有脚本只吃 jsonl，一行一条：

```jsonc
// 生成
{"id": "00001", "prompt": "A clock ticking."}
// 理解
{"id": "00001", "question": "...", "answer": "...", "choices": ["A","B"], "audio": "/path/a.wav"}
// 编辑
{"id": "00001", "task": "Add", "source_audio": "/path/src.wav", "desc": "skateboarding", "target_audio": "/path/tgt.wav"}
```

HF 上的数据集可以用各自的 `prepare` 子命令导出；数据集 id 不确定时用 `--dataset` 显式指定。

## 三步流程（以 AudioCaps 为例）

> 仓库**不自带** benchmark 提示词。先用 `prompts.sample.jsonl`（10 条）打通链路，
> 再 `prepare` 导出完整测试集。`score` 需要真实参考音频，没有数据集时只能跑 `gen`。

```bash
# 0) 冒烟：2 条样本，确认能出声
python runs/Benchmark/AudioCaps/evaluate.py gen \
  --prompts runs/Benchmark/AudioCaps/prompts.sample.jsonl \
  --out runs/Benchmark/_runs/audiocaps_smoke --limit 2 \
  --model-config Audio-Omni/model/Audio-Omni.json --ckpt Audio-Omni/model/model.ckpt \
  --device cuda:0 --cond-device cuda:1 --dtype float16 --steps 30
```

```bash
# 1) 生成（8 卡 → 4 个分片，每片 2 卡：主干 cuda:0、理解塔 cuda:1）
GPUS_PER_PROC=2 bash runs/Benchmark/run_shards.sh runs/Benchmark/AudioCaps/evaluate.py 4 \
  --prompts runs/Benchmark/AudioCaps/prompts.jsonl \
  --out runs/Benchmark/_runs/audiocaps \
  --model-config Audio-Omni/model/Audio-Omni.json --ckpt Audio-Omni/model/model.ckpt \
  --dtype float16 --steps 100 --cfg 6.0

# 2) 合并分片清单
python runs/Benchmark/AudioCaps/evaluate.py merge --out runs/Benchmark/_runs/audiocaps

# 3) 打分（需要 AudioCaps test 的真实音频目录）
python runs/Benchmark/AudioCaps/evaluate.py score \
  --out runs/Benchmark/_runs/audiocaps --ref /data/audiocaps_test_audio
```

### 显存与分片怎么选

| 模式 | 每进程占用 | 吞吐 | 什么时候用 |
|---|---|---|---|
| `GPUS_PER_PROC=1`（8 片 × 1 卡） | fp16 ≈ 22–25 GB，fp32 ≈ 30 GB | 8× | 单卡放得下（32 GB 卡 + fp16 勉强） |
| `GPUS_PER_PROC=2`（4 片 × 2 卡） | 主干 8 GB + 理解塔 12 GB | 4× | **推荐起点**，稳，不会 OOM |

`run_shards.sh` 会强制设置 `--device cuda:0` 和 `--cond-device cuda:0/1`，所以调用方不用再传这两个参数；
传了也会被脚本覆盖。

理解类（MMSU/MMAU）把 `gen/score` 换成 `eval/merge`，编辑类把 `merge` 后的 `score` 直接打在
`manifest.jsonl` 上（`target_audio` 就在 items 里）。

## 依赖

| 用途 | 包 | 是否必需 |
|---|---|---|
| CLAP-FAD / CLAP score | `laion-clap`（已在 Audio-Omni 依赖里） | 必需 |
| KL / IS / FD(PANNs) | `pip install panns-inference` | 可选 |
| VGGish-FAD | `pip install fadtk` | 可选 |
| WER | `transformers` 的 Whisper pipeline + `pip install jiwer` | TTS 时必需 |

## 已有基线的参考

论文 Table 3/4（Audio-Omni 自身）：

| 指标 | 值 |
|---|---|
| T2A_FAD | 1.86 |
| T2M_FAD | 1.94 |
| TTS_WER | 1.77 |
| AE_FAD | 3.27 |
| AE_LSD | 2.27 |
| CLAP（编辑） | 0.32 |

自己复现时注意：**FAD 的参考集不同、嵌入模型不同，数值不能跨协议比**。报告里务必写清
「CLAP-FAD」还是「VGGish-FAD」。

## 注意

- 生成很慢（100 步 ≈ 数十秒/条），先用 `--limit 10` 打通，再全量；
- 每卡都要放得下完整模型；单卡放不下就用 `--cond-device` 把理解塔挪到第二张卡；
- 打分阶段只读 wav，可以反复跑、换 `--device` 不用重新生成。
