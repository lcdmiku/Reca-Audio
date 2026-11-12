# 四组 post-training 诊断

## 已核验的历史证据

本轮使用 `Audio-Omni/model/model.ckpt` 与 `runs/train/reca_long_matrix_b_10s_v1/{pure,t2a,mix}/dit_exposure2000.pt`。服务器同时存在 `reca_long_matrix_b_v1`，对应 5 秒训练。10 秒训练启动器的仓库版本可追溯到 `704847f4f707d2404d94897e99b6610c757f7a8e`，但 checkpoint 的训练合同没有写入 Git commit，故历史启动时的准确部署 SHA 尚无直接记录。运行报告将单独记录本轮部署版本和 checkpoint 文件 SHA256。

| 模型 | 用户提供的 AudioCaps 历史值 | 10 秒训练实际 AudioCaps | 用户提供的 MusicCaps 历史值 | 10 秒训练实际 MusicCaps |
|---|---:|---:|---:|---:|
| Base | 0.4610 | 0.46096 | 0.3882 | 0.38817 |
| Pure ReCA | 0.4458 | 0.43858 | 0.4086 | 0.39908 |
| T2A-only | 0.4310 | 0.43096 | 0.3998 | 0.39981 |
| Mix | 0.4286 | 0.42864 | 0.4057 | 0.40567 |

用户提供的 Pure 数值与 `docs/result.md` 的 5 秒结果相符。10 秒数值来自 `LongMatrixB_10s_eval_v1` 原始评分，Base 来自 `Matrix_v1_eval3_6gpu_9ce3984`。每组每域抽查首条历史 WAV，均为 44.1 kHz、483328 帧（约 10.960 秒）。这与配置中的 485100 经 VAE 比率 2048 向下取整一致。历史 manifest 的 `seconds=10` 记录的是条件长度，生成窗口取自配置。新实验保持条件长度并显式比较 441000 和 485100 两个生成窗口。

| 模型 | ReCA 曝光 | T2A 曝光 | 计划更新 |
|---|---:|---:|---:|
| Pure | 2000 | 0 | 500 |
| T2A-only | 0 | 2000 | 500 |
| Mix | 1000 | 1000 | 500 |

三组的训练合同均为 batch 1、累积 4、100 曝光 warmup、峰值学习率 2e-5。入口 `run_gradient_ab.py` 使用 Adafactor，关闭 relative step、parameter scaling 和内部 warmup。训练主权重为 fp32，AMP fp16，生成诊断为 fp16。三组序列各覆盖 1524 个 manifest row index，行数包含同一音频的不同 caption，不能将其视作独立音频数。

## 训练轨迹

固定诊断使用相同数据和随机数。下表是诊断样本上的单步 Flow MSE，不是独立生成结果。

| 模型 | ReCA 0 → 2000 | T2A 0 → 2000 | 两任务梯度 cosine 0 → 2000 |
|---|---:|---:|---:|
| Pure | 0.94649 → 0.93360 | 0.93434 → 0.93759 | 0.72632 → 0.86307 |
| T2A-only | 0.94649 → 0.93678 | 0.93434 → 0.93730 | 0.72625 → 0.89160 |
| Mix | 0.94649 → 0.93611 | 0.93434 → 0.93695 | 0.72630 → 0.89382 |

T2A-only 的 held-out 单步 T2A loss 在 1000 和 2000 曝光均高于起点，和历史 AudioCaps CLAP 下降方向一致。三组 ReCA 单步 loss 均下降，包括从未接受 audio condition 训练的 T2A-only。要识别音频条件学习，须比较真实重建及 matched/mismatched 差值。梯度 cosine 衡量同一状态下局部梯度方向，无法单独解释生成退化。

## 已支持和待验证的解释

- **H1** 历史 AudioCaps 中 T2A-only 也下降，ReCA 并非退化的必要条件。MusicCaps 的 T2A-only CLAP 同时上升，退化具有任务域依赖性。现有设计无法分离数据规模、分布和优化制度。
- **H2** 10 秒历史 AudioCaps 中 Pure 高于 T2A-only 约 0.00762，低于 Base 约 0.02238。前期条件对齐与 block 恢复研究未发现可直接替代真实 T2A 的代理指标。是否迁移需要统一协议下的双向生成证据。
- **H3** Mix 的历史 MusicCaps CLAP 和分布指标优于 10 秒 Pure，但历史 AudioCaps CLAP 最低。Mix 每个任务只获得单任务组一半曝光，两域成绩不能当作等剂量任务交互证据。
- **H4** 这三组正式 checkpoint 的训练入口将 CFG dropout 固定为 0。常规单卡入口的默认值是 0，DDP 默认值是 0.1；两者都不是这批训练的入口。推理在 DiT 内将 cross-attention 和存在时的 prepend embedding 置零，保留条件掩码、global、input concat 和 timestep。无负条件时采用 `v_u + s(v_c-v_u)`。默认 `scale_phi=0`，所以传入 `rescale_cfg=True` 本身不会触发 std rescale。Base 的上游训练 dropout 缺少运行证据。该配置差异支持测试 CFG mismatch，尚不能确定其影响大小。
- **H5** 707 个唯一训练音频对全 DiT 更新可能偏少。没有规模梯度或同分布更大数据的实验，无法判断规模是否是原因。
- **H6** 当前固定诊断 loss 与真实生成度量的关系依赖模态和采样。现有结果已要求分开报告两者，不能由 loss 变化推出生成收益。

## 本轮证据范围

AudioCaps 取 89 个唯一音频 SHA 各一条 caption，其中 8 个与探索集重叠，主要统计使用余下 81 个 SHA。MusicCaps 使用固定 seed 选择的 89 个唯一 SHA，保留项目的 T2M 文本模板。各组共用按全局索引分配的 seed 和 shard。10 秒窗口实际生成约 9.985 秒。四组 reconstruction 使用固定 crop 的 10 秒原音频、4×低通、静音 mel 输入和 CFG=1。错配用另一音频组的条件，与原目标保持相同初始随机种子。Geometry 分别保留 audio 和 text 两种条件，基于真实 latent 加噪与实际 CFG batch 中的两路输出计算。

完整生成评测、四组参数漂移和 CFG geometry 的结果将在正式运行后填写。
