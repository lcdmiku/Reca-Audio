# ReCA 训练数据完整审计（2026-11-09）

## 1. 审计范围与口径

本报告审计当前 `reca_long_matrix_b_v1` 实际使用的数据路径，不依据 README 推测：

- 源 manifest：`runs/Benchmark/_data/audiocaps/train_manifest.jsonl`
- 源 manifest SHA256：`b2f346d1ff01ebdb1d18550aed2fe23177b37f77e7d3ebe1f404b3fa73955cdc`
- 正式 split：`runs/Benchmark/_data/audiocaps/reca_split_v1/{train,val,test}_manifest.jsonl`
- 实际 schedule：`runs/train/reca_long_matrix_b_v1/{pure,t2a,mix}_schedule.jsonl`
- 在线训练 crop：44.1 kHz、5 秒；按每个 exposure 的 `data_seed` 复现
- caption–crop 筛查：LAION-CLAP，同一 caption 分别对完整音频和实际 5 秒 crop 打分

CLAP 只用于问题筛查，不等价于人工标注真值。低分项需要试听后才能判定为错标。

## 2. 数据总体统计

| 项目 | Source | Train | Val | Test |
|---|---:|---:|---:|---:|
| manifest 行数 | 4411 | 3530 | 436 | 445 |
| 不同文件路径 | 4411 | 3530 | 436 | 445 |
| 独立音频 SHA256 | **884** | **707** | **88** | **89** |
| 空 caption | 0 | 0 | 0 | 0 |
| 缺失/不可读文件 | 0 | 0 | 0 | 0 |

4411 行不是 4411 段独立音频。SHA 组大小为：881 组各 5 个路径、1 组 4 个路径、2 组
各 1 个路径。也就是说，数据主体是同一音频内容的约 5 条 AudioCaps caption/文件副本。
split 按 SHA 分组后再划分，因此 train/val/test 的 ID 和 SHA 交集均为 0。

所有 4411 个文件均为 24 kHz；声道分布为单声道 421、双声道 3980、六声道 10（后者
对应 2 个独立 SHA）。独立 SHA 的时长中位数为 10.0 秒，均值 9.862 秒，最短 1.742 秒；
RMS 中位数为 -22.20 dBFS。

## 3. 训练 schedule 覆盖

Pure/T2A/Mix 使用相同的 2000 个 sample index、data seed、flow seed 和 prompt seed；仅 task
标签不同。每组覆盖：

- 1524 / 3530 个 caption 行（43.2%）；
- 671 / 707 个 train SHA（94.9%）；
- 每个被覆盖 SHA 的曝光次数：最少 1、中位数 3、P95 6、最多 9；
- Mix 精确为 1000 ReCA + 1000 T2A。

因此三组对照的数据和 crop 完全对齐。需要注意，采样单位是 caption 行，不是独立 SHA；
同一 SHA 的约五个副本使音频被重复采样。因为绝大多数 SHA 恰有五个副本，整体权重近似
均匀，但不是严格的“每个独立音频等概率”。Pure ReCA 根本不使用 caption，却仍承担这层
重复行采样。

## 4. Caption–crop 对齐

对 T2A schedule 的 2000 个真实 exposure 逐一复现 5 秒随机 crop：

| CLAP | 均值 | P05 | 中位数 | P95 | 最小 | 最大 |
|---|---:|---:|---:|---:|---:|---:|
| 完整音频–caption | **0.5254** | 0.3248 | 0.5451 | 0.6634 | -0.0008 | 0.7347 |
| 实际 crop–caption | **0.4087** | 0.1547 | 0.4288 | 0.6015 | -0.1592 | 0.7102 |
| crop − full | **-0.1166** | -0.3410 | -0.0930 | 0.0160 | -0.7415 | 0.2704 |

关键结果：

- 564 / 2000（28.2%）exposure 的 crop 相对完整音频下降超过 0.15；涉及 476 个 caption
  行、311 个独立 SHA。
- bottom 5%（crop CLAP < 0.15474）有 100 次 exposure，涉及 88 行、72 个独立 SHA。
- 1086 / 2000（54.3%）caption 含 `followed by / then / before / after / while / as` 等时序词。
  这些 caption 的 crop 均值为 0.3992，低于非时序 caption 的 0.4201；bottom-5% 比例也
  更高（5.6% vs 4.3%）。
- Mix 的 1000 个 T2A exposure 中，286 个（28.6%）下降超过 0.15，53 个落入全体
  bottom-5%。所以 Mix 的文本监督确实含有较强的随机 crop 标签噪声。

这不是 ReCA 条件与 target 的错位：在 ReCA exposure 中，Qwen 条件和 VAE target 都来自
同一个 5 秒 crop。错位发生在 T2A/Mix 的文本分支，因为 caption 描述原始约 10 秒完整事件，
而 target 是随机 5 秒子段。

代表性高风险样本：

| ID | crop 起点(s) | crop CLAP | full CLAP | 差值 | caption |
|---|---:|---:|---:|---:|---|
| 107118 | 4.661 | -0.126 | 0.616 | **-0.742** | A door pounding and rattling |
| 106360 | 1.816 | 0.005 | 0.642 | **-0.637** | Brief clanking followed by a woman speaking and sewing machine working |
| 107383 | 0.082 | 0.051 | 0.656 | **-0.604** | A man speaks followed by a toilet flush |
| 106058 | 4.704 | 0.014 | 0.610 | **-0.596** | A male voice talking and the bleat of a goat |
| 105401 | 0.986 | -0.013 | 0.574 | **-0.587** | A man speaks, metal clinks, then a drill operates |
| 103571 | 0.229 | 0.072 | 0.650 | **-0.578** | Multi-stage train/bell/mower/steam/crowd caption |
| 104951 | 1.499 | 0.039 | 0.593 | **-0.554** | A man talking followed by plastic crinkling and crumpling |

## 5. 音频质量问题

以下阈值是自动筛查条件，不等于应直接删除：

| 问题 | 路径/行数 | 独立 SHA | 口径 |
|---|---:|---:|---|
| 少于 5 秒 | 25 | **5** | 训练会在尾部补零 |
| 近静音 | 145 | **29** | 全段 RMS < -45 dBFS |
| 超过一半静音 | 285 | **57** | 100 ms frame 中 >50% 低于 -50 dBFS |
| 明显削波 | 120 | **24** | `abs(sample)>=0.999` 超过 0.1% |
| 六声道输入 | 10 | **2** | 训练直接对所有声道求均值 |

五个短音频独立 ID 为 `102815`（4.899s）、`102923`（3.274s）、`103062`（3.518s）、
`103222`（4.272s）、`103435`（1.742s）。所有路径均可读，没有 NaN/缺文件；近静音和长静音
可能是合法的稀疏事件录音，应结合 caption 与试听决定是否过滤。

问题清单共有 1034 条记录、372 个独立 SHA。其中 audio-level 455 条路径记录对应 91 个
独立 SHA；crop-level 579 次 exposure 对应 321 个独立 SHA。完整清单见配套 CSV。

## 6. Benchmark 重叠

| Benchmark | 与 Train 的 ID/文件名重叠 | 与 Train 的 SHA 重叠 | 判断 |
|---|---:|---:|---|
| 旧 `AudioCaps full/prompts1000` | 780 | **507** | 严重污染，不可作为当前训练的 held-out 结果 |
| MusicCaps 1004 | 0 | **0** | 未发现 ID/文件名/SHA 重叠 |
| `reca_split_v1/test` | 0 | **0** | 可信内部 held-out（89 独立 SHA、445 captions） |

因此，早期在 `full/prompts1000` 上得到的训练后指标不能用于证明泛化。最近的
`LongMatrixB_eval_v1` 使用 split test 445 条，按 SHA 与 train 隔离；MusicCaps 也未发现内容重叠。

## 7. 当前“低通 4×”的真实处理

结论：**ReCA 音频条件当前确实执行 4× 低通，但它不是把最终张量保持在 4 kHz，也不是
对 VAE target 做降采样。**实际路径是：

```text
磁盘 WAV（本数据全部 24 kHz）
  -> 求声道均值
  -> 重采样到 44.1 kHz
  -> 在线随机裁/补到 5 秒
  ├─> 复制为双声道 -> VAE target（44.1 kHz；不做 4× 低通）
  └─> 求均值 -> 44.1k -> 16k
       -> 16k -> 4k -> 16k（torchaudio sinc resample）
       -> Qwen processor / thinker
       -> hidden_states[-2] -> RMSNorm -> Linear(2048, 768)
       -> DiT dense condition
```

具体含义：

- `omni_downsample=4` 由正式 launcher 显式传入；`omni_seconds=0`，所以没有额外的
  Qwen-only 1 秒截断。
- 16k→4k 的 anti-alias resample 去掉约 2 kHz 以上频率，再 4k→16k 只恢复采样点数，
  不会恢复高频。因此是“保持 5 秒长度和 16k 接口形状的低通瓶颈”。
- Qwen processor 看到的仍是 16 kHz 长度数组，不知道中间经过 4 kHz，因此不会发生
  sampling-rate shape 报错；token 数也不会因 4× 降采样而直接缩短。
- 数据原始带宽最多约 12 kHz（24 kHz Nyquist）；VAE 虽以 44.1 kHz 接收 target，不能凭
  上采样创造 12 kHz 以上信息。Qwen 条件则进一步被限制到约 2 kHz。
- `audio_input_prompt` 仍是固定 10 秒、44.1 kHz 全零波形，只提供 mel conditioner 所需
  形状，不泄漏目标声学细节。
- 当前取的是 Qwen 倒数第二层、完整有效序列（文本 + audio token 等），不是只选 audio
  token；长度右补到 2048，超长时线性插值到 2048。

## 8. 结论与建议

1. 数据文件完整、split 的 SHA 隔离正确，最近 held-out 与 MusicCaps 结果可继续使用。
2. 最大的数据风险不是缺文件，而是 **10 秒 caption + 随机 5 秒 target**。它污染 T2A-only
   和 Mix 监督，但不破坏 ReCA 的 audio-condition/target 同段对齐。
3. 后续最小单变量实验应优先把 T2A target 改为完整 caption 对应的固定/完整窗口，或使用
   带时间边界的 caption；不要同时改变低通、LR 和模型结构。
4. Pure ReCA 可按独立 SHA 采样，避免五 caption 副本造成无意义重复；T2A 仍可在选中 SHA
   后随机选一条 caption。
5. 先人工试听 bottom-5%、短音频、近静音和削波的独立 SHA，再决定过滤规则；不能根据
   CLAP 或响度阈值直接删除合法稀疏声音。

远程完整产物：`runs/audits/reca_training_data_20261109_v2/`。

