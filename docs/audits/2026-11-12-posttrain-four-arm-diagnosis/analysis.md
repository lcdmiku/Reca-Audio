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

T2A-only 的训练 T2A loss 从前 50 次更新均值 1.03938 降到后 50 次的 1.02146，held-out 单步 T2A loss 在 1000 和 2000 曝光均高于起点，历史 AudioCaps CLAP 也下降。这与训练集拟合和泛化之间的差距相容。前后训练 batch 不同，训练 loss 差值本身无法定量估计过拟合。三组 ReCA 单步 loss 均下降，包括从未接受 audio condition 训练的 T2A-only。要识别音频条件学习，须比较真实重建及 matched/mismatched 差值。梯度 cosine 衡量同一状态下局部梯度方向，无法单独解释生成退化。

## 已支持和待验证的解释

- **H1** 历史 AudioCaps 中 T2A-only 也下降，ReCA 并非退化的必要条件。MusicCaps 的 T2A-only CLAP 同时上升，退化具有任务域依赖性。现有设计无法分离数据规模、分布和优化制度。
- **H2** 10 秒历史 AudioCaps 中 Pure 高于 T2A-only 约 0.00762，低于 Base 约 0.02238。前期条件对齐与 block 恢复研究未发现可直接替代真实 T2A 的代理指标。是否迁移需要统一协议下的双向生成证据。
- **H3** Mix 的历史 MusicCaps CLAP 和分布指标优于 10 秒 Pure，但历史 AudioCaps CLAP 最低。Mix 每个任务只获得单任务组一半曝光，两域成绩不能当作等剂量任务交互证据。
- **H4** 这三组正式 checkpoint 的训练入口将 CFG dropout 固定为 0。常规单卡入口的默认值是 0，DDP 默认值是 0.1；两者都不是这批训练的入口。推理在 DiT 内将 cross-attention 和存在时的 prepend embedding 置零，保留条件掩码、global、input concat 和 timestep。无负条件时采用 `v_u + s(v_c-v_u)`。默认 `scale_phi=0`，所以传入 `rescale_cfg=True` 本身不会触发 std rescale。Base 的上游训练 dropout 缺少运行证据。该配置差异支持测试 CFG mismatch，尚不能确定其影响大小。
- **H5** 707 个唯一训练音频对全 DiT 更新可能偏少。没有规模梯度或同分布更大数据的实验，无法判断规模是否是原因。
- **H6** 当前固定诊断 loss 与真实生成度量的关系依赖模态和采样。现有结果已要求分开报告两者，不能由 loss 变化推出生成收益。

## 本轮证据范围

AudioCaps 取 89 个唯一音频 SHA 各一条 caption，其中 8 个与探索集重叠，主要统计使用余下 81 个 SHA。MusicCaps 使用固定 seed 选择的 89 个唯一 SHA，保留项目的 T2M 文本模板。各组共用按全局索引分配的 seed 和 shard。10 秒窗口实际生成约 9.985 秒。四组 reconstruction 使用固定 crop 的 10 秒原音频、4×低通、静音 mel 输入和 CFG=1。错配用另一音频组的条件，与原目标保持相同初始随机种子。Geometry 分别保留 audio 和 text 两种条件，基于真实 latent 加噪与实际 CFG batch 中的两路输出计算。

四个 checkpoint 的完整 SHA256、文件大小、任务序列 SHA256、部署版本及确定性样本列表见同目录 `protocol.json`。四组 checkpoint 已在 smoke 中通过 DiT 加载，原始存储精度的参数检查及相对 L2 结果见 `parameter_drift.md`。

正式 geometry 已核验 192 条记录。三组文本条件在 CFG=1 时已经偏离 Base 约 13%，CFG=6 扩大到约 30%。因此当前证据支持 CFG 会放大模型变化，也表明变化并非只发生在 null 分支。该结果来自固定 latent 的单步预测，不能替代生成 sweep。分模态表见 `cfg_sweep.md`。

Smoke 以 2 步通过生成、geometry 和评分。正式评测于北京时间 2026-11-13 06:43 完成，部署版本为 `b85ded1`。核验 4 组 × 8 设置 × 89 样本共 2,848 条生成记录，SHA、caption、seed、帧数和错配源均与合同一致。服务器另保存 89 条 10 秒重建参考音频。完整结果位于 `/home/linchudeng/Reca-Audio/runs/audits/posttrain_four_arm_v1/`，逐条评分和结构化结果见同目录 `results.json`。

## 统一协议的结果

AudioCaps 和重建使用 81 个未参与前期 8 条探索的 SHA 作为主要分析集，MusicCaps 使用确定性抽取的 89 个唯一 SHA。两个时长使用相同 seed。长度改变同时改变 latent 序列形状，因此时长消融衡量整个生成窗口改变的效果。CFG sweep 只在 AudioCaps 执行。所有统计区间为样本 SHA 配对 bootstrap 的边际 95% 区间，未作多重比较校正。

| 模型 | AudioCaps 10 秒 CFG=1 | CFG=2 | CFG=4 | CFG=6 | 11 秒 CFG=6 | MusicCaps 10 秒 CFG=6 | 重建 paired CLAP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 0.29565 | 0.41101 | 0.47207 | **0.49570** | **0.47309** | 0.36946 | 0.18804 |
| Pure ReCA | 0.25657 | 0.38120 | 0.43753 | 0.46101 | 0.43945 | 0.37534 | **0.28958** |
| T2A-only | 0.21548 | 0.34574 | 0.41253 | 0.43435 | 0.42600 | 0.37231 | 0.19502 |
| Mix | 0.24567 | 0.36137 | 0.41724 | 0.43900 | 0.42767 | 0.37523 | 0.25758 |

### H1 与 H6：普通 post-training 的退化与目标错配

T2A-only 在 CFG=1 的 AudioCaps CLAP 比 Base 低 0.08017，95% CI [−0.11349, −0.04675]。CFG=6 仍低 0.06135 [−0.08738, −0.03567]。两种生成窗口的四组排序相同。结合前述训练 loss 降低、固定 held-out loss 升高，以及 CFG=1 条件 velocity 漂移约 13%，可确定退化在普通 T2A post-training 中也发生，并涉及 conditional 映射。小数据上全量更新、训练样本拟合和数据域变化都与这些观察相容，目前没有数据规模、冻结范围或优化配置的单变量对照来分离它们。

Flow MSE 的改善不能直接预测采样语义。例如 T2A-only 固定诊断的 ReCA MSE 下降，但真实重建相对 Base 仅 +0.00698 [−0.02624, +0.04064]。正式 geometry 中三组文本 CFG=6 单步 MSE 低于 Base，实际生成 CLAP 却均低于 Base。

### H2：Pure 保留了较多 T2A 能力，但跨模态净收益仍未建立

Pure 在四个 CFG 点的 AudioCaps CLAP 均高于 T2A-only。CFG=6 的配对差值为 +0.02666 [约 +0.00138, +0.05285]，CFG=1 为 +0.04108 [约 +0.01453, +0.06714]。CFG=4 的区间跨零。Pure 在 10 秒 CFG=6 比 Base 低 0.03468 [−0.05922, −0.01149]，在 11 秒也低 0.03364 [−0.05541, −0.01223]。

Pure 的音频条件重建比 Base 高 0.10153 [0.06076, 0.14284]，CLAP-FAD 从 0.78491 降至 0.64192，LSD 从 30.848 降至 29.155 dB。匹配与错配的 paired CLAP 差值为 0.20925，Base 为 0.15799。两者差值的配对区间约为 +0.05126 [0.00771, 0.09513]。这些结果支持 Pure 更有效地利用音频条件。对 Text Condition 的证据是相对 T2A-only 的保留优势，以及相对 Base 的下降。尚未观测到可归因于 ReCA 的 T2A 净提升，也无法从音频重建增益中分离语义与残留声学信息。

### H3：Mix 的折中及域差异

Mix 的重建相对 Base 为 +0.06954 [0.03109, 0.10836]，比 T2A-only 高约 0.06256 [0.03818, 0.08749]，比 Pure 低约 0.03199 [0.01169, 0.05242]。AudioCaps CFG=6 相对 T2A-only 为 +0.00465 [约 −0.02240, +0.03302]，相对 Pure 为 −0.02201 [约 −0.04805, +0.00263]。本轮没有证据表明 Mix 在这两种能力上同时优于单任务组。

89 条 MusicCaps 中，三个训练组相对 Base 的 CLAP 差值区间均跨零。Mix 的 CLAP-FAD 最低（0.61819，Base 0.64788），而 PANNs FD 高于 Base（45.19 对 43.90）。历史 1004 条 MusicCaps 曾显示 Mix 的 CLAP 优势，本轮抽样和时长协议下只观察到方向相同的均值变化。Mix 的每任务曝光减半，任务交互和训练剂量无法从这一组比较中独立识别。

### H4 与长度：两个因素影响分数，均不足以解释退化

四组 AudioCaps CLAP 从 CFG=1 到 6 逐点增加，最高测量点均为 6。Pure 没有随 CFG 增大而出现绝对 CLAP 下降，相对 Base 的差距始终存在。Mix 的差距从 −0.04998 变到 −0.05670。三个训练组与 Base 的 CFG=6 减 CFG=1 的差中差区间均跨零，当前四点数据未建立某组的 CFG 敏感性显著不同或最优区间移动。固定 latent 中 CFG=6 把文本 velocity 漂移放大到约 30%，但生成时较高 CFG 仍提高了所有组的 CLAP。缺少 null-condition 训练可能影响 guidance，不能由这些 checkpoint 证明其因果效应。

11 秒比 10 秒的 CLAP 均值低约 0.00835–0.02260。Base 和 Pure 的边际配对区间低于零，Base 上界仅约 −0.00004。T2A-only 与 Mix 的区间跨零。各训练组相对 Base 的时长差中差区间都跨零。长度影响绝对分数，统一后训练组在 AudioCaps 仍低于 Base。

### H5 与下一组单变量实验

707 个唯一训练音频与约 3.04B 个更新参数的组合值得关注，现有运行没有数据规模对照，无法确定样本规模是否是主因。现阶段可直接检验的单变量是 **T2A-only 的 CFG dropout 从 0.0 改为 0.1**，用相同 Base、数据、序列、2000 曝光、optimizer、累积和学习率作为控制。沿用 10 秒窗口及四个 CFG 点，预先以 CFG=6 相对原 T2A-only 的配对 CLAP 为主要指标，并保留 CFG=1 与 conditional/unconditional geometry。若 CFG=6 改善而 CFG=1 变化很小，同时 null/guidance 更接近 Base，将支持 null 训练缺失解释部分退化。若 CFG=1 和 6 仍同向退化，优先检查全量更新幅度。此轮只给出实验方案。

每条 prompt 使用一个配对 seed，未扩展到可选的 445 captions。CLAP-FAD 和 PANNs 指标使用各设置全部 89 条，不与历史 445/1004 条绝对值横比。重建错配仍来自同一参考总体，因此错配 FAD 可能与匹配 FAD 接近。逐目标 paired CLAP、LSD 以及匹配减错配的差值更直接反映条件对应关系。组间和差中差补充区间按同一批 SHA 重采样 10000 次（seed 42，经验 2.5% 与 97.5% 分位），均为未校正的描述性区间。
