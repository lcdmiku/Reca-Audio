# Audio-self

自建统一音频生成模型（Path B，保底路线）。

- 目的：在**不依赖上游预训练权重**的前提下，可控地扫「理解编码器 × 压缩率」，画出倒 U 曲线。
- 结构：DiT + rectified flow，条件 = 冻结编码器特征（MERT / CLAP / AudioMAE / Whisper）。
- 状态：待实现（D3 之后）。

上游 Audio-Omni 的改动在 `Audio-Omni/`，两者共享 `runs/` 下的数据处理与评测脚本。
