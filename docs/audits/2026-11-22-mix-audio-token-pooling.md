# 四个音频 token 局部均值的 Mix 对照

[同 caption 预算实验](2026-11-22-caption-budget-results/analysis.md)中，ReCA 使正确音频相对错配音频的重建优势增加 0.06436，但 AudioCaps CFG=6 的文本 CLAP 差值为 −0.00508，95% 区间为 [−0.02058, +0.01043]。本实验检验削弱音频条件在相邻 token 间的变化，能否在相同文本监督下改善文本生成。

## 唯一改动与适用范围

在 ReCA 样本上，从冻结 Qwen Thinker 的倒数第二层读取 hidden states，照常通过冻结的 RMSNorm 和 2048→768 投影。依据 processor 的实际 input IDs 和有效 mask 找出每段连续的音频内容 token。在每段内部，从左至右每 4 个向量取一次均值，并写回原来的 4 个位置。尾段按实际个数求平均。文本、边界符、重建指令和 padding 保留原位置，DiT 继续接收相同形状和有效 mask 的条件。

服务器真实 processor 在历史 Mix 首个四槽组上的有效序列长度依次为 47、337、50、325。两个 ReCA 槽各有一段 250 个音频 token（ID 为 151646），对应 62 组四个和一组两个；两个 T2A 槽均为零个音频 token。这个操作保留全局时序、平均音色和经上下文编码的信息，因此它对局部时间变化的干预不能单独识别所有声学细节或语义。

| 路径 | 新实验 |
|---|---|
| Qwen 输入 | 沿用 10 秒窗口。ReCA 波形从 16 kHz 降到 4 kHz 再升回 16 kHz，文本槽使用原 caption 模板。|
| DiT 条件 | 只有 ReCA 的投影后音频 token 执行分组均值；低层声学提示仍为静音。|
| VAE target | 同一原始片段进入冻结 VAE，使用历史 Mix 的裁剪和采样流程。|
| 训练与生成 | 仍为 rectified flow，更新全部 DiT 权重。T2A 推理沿用原文本通路；匹配和错配重建对新组使用与训练一致的音频处理。|
| 缓存 | 本次训练在线提取条件。文本条件和 VAE 缓存规格保持一致；投影后已完成池化的 ReCA 条件要与原条件区分。|

## 训练对照

从同一 Audio-Omni Base 开始，逐字节复用低 LR Mix 的调度，SHA256 为 `bd5ca9ffa653c06af7d0c1eefb1e36e2fbf6ed09f8bf1b496afb9da1107126a0`。训练池有 7,070 条独立音频，7,072 个槽位由 3,536 次 ReCA 和 3,536 次 T2A 曝光组成。每次更新包含两个 ReCA 和两个 T2A loss，四项之和除以 4，共 1,768 次计划更新。保持 seed 42、Adafactor、梯度裁剪 1.0、fp32 主权重和 AMP fp16。峰值 LR 为 5e-6，warmup 352 个槽位，随后 cosine 衰减，每次更新取该四槽 LR 均值。

## 预定判断

主要终点为 AudioCaps CFG=6 的 81 个既定 SHA，按 SHA 配对计算 CLAP 均值差和 95% bootstrap 区间。第一项比较为**池化 Mix 减原 Mix**，回答改变音频条件后的 T2A 效应；第二项为**池化 Mix 减仅文本槽**，衡量同标注预算下增加这组 ReCA 监督的净收益。超过原 Mix 但未明显超过仅文本槽，只支持相对旧配方的改善。两个对比共享新模型与评测样本。它们的边际区间不提供同时置信保证。

同时记录 AudioCaps CFG=1、MusicCaps CFG=6、CLAP-FAD、FD、KL、IS 和正确／错配音频重建。若仅观察到重建变弱，表明原局部信息可能有助于该重建配方，迁移收益仍待验证。重建比较同时改变新模型训练和重建推理时的条件处理；文本生成比较沿用完全相同的文本推理路径。单一训练 seed 和反复使用的评测集限制外推。

## 服务器入口

- 训练命令为 `bash runs/scripts/train_audiocaps_mix_pool4.sh`。目录为 `/home/linchudeng/Reca-Audio/runs/train/audiocaps_mix_pool4_v1/`。终点路径为 `mix_pool4_7070/dit_exposure7072.pt`，逐步记录为 `mix_pool4_7070/metrics.jsonl`。
- 评测命令为 `bash runs/scripts/eval_audiocaps_scale.sh --mix-pool4`。沿用相同 prompt、seed、10 秒请求、100 步和 fp16，只为新模型生成 445 条音频。输出目录为 `/home/linchudeng/Reca-Audio/runs/audits/audiocaps_mix_pool4_eval_v1/`。
