# 先 ReCA 后 T2A：等总曝光顺序训练

主要问题是纯文本生成。假设连续的 ReCA 更新先适配音频条件，后续真实 caption 梯度能让 DiT 在文本条件下利用这部分学习；另一种可能是末段遗忘音频条件，或文本阶段有效更新过小。两种可能都不能凭训练 loss 单独区分。

唯一新训练臂从原始 Audio-Omni 开始，不加载任何历史 Pure 终点。用低 LR Mix 已验证的逐曝光记录，保留任务内部先后、音频索引、裁剪、提示词与 flow 随机种子，仅重新编排顺序与曝光号。前 3,536 条为 ReCA，后 3,536 条为 T2A。两段合计覆盖原 7,070 条音频；曝光不等于每个任务各有 3,536 条不同音频。训练 Qwen、VAE 冻结，DiT 全量更新。数据、10 秒窗口、音频条件 4× 低通、静音 mel、fp32 主权重、AMP fp16、micro-batch 1、累积 4、Adafactor、seed 42 与 Mix 保持一致。未修改 prompt、特征或目标 latent，不需重算缓存。

阶段各 884 次计划更新，共 1,768 次。一条曝光索引 LR 曲线从头用到尾：warmup 352、峰值 5e-6、cosine，每次更新取 4 个曝光位置的平均值。切换时不重建模型、不重置优化器、scaler 或 warmup。对完整的计划更新求和，前段占名义 LR 总量 80.15%（平均约 4.008e-6），后段占 19.85%（平均约 0.992e-6）；实际参数位移不与 LR 总和简单成比例。原 Mix 的每次更新有两条各任务样本，新方案每次更新四条同任务样本。此实验比较固定全局曲线下的训练制度，不能分别识别任务顺序、组内梯度混合和每个任务的 LR 分配。

切换前在 3,536 曝光增加一次不更新参数的固定诊断，另保留原诊断位置 0/2000/4000/7072。由于存储预算仅保留最终 checkpoint，不主张已得到中点生成能力证据。评测时主要与 5e-6 Mix 对齐输入、seed、采样参数，比较 AudioCaps CFG=6 T2A 的配对 CLAP、CLAP-FAD/PANNs FD 等指标；Base 判断是否有净提升，Pure、T2A-only 为上下文。CFG=1、MusicCaps CFG=6、正确/错配音频重建补充解释。若目标是证明音频预训练本身有益，还需匹配末段文本样本和 LR 的无音频预训练对照。本轮不同时启动反向顺序或其它对照。

启动脚本：`runs/scripts/train_audiocaps_reca_then_t2a.sh`。服务器输出根：`/home/linchudeng/Reca-Audio/runs/train/audiocaps_reca_then_t2a_v1/`。主日志：`sequential_7070.log`；逐步指标：`sequential_7070/metrics.jsonl`；终点：`sequential_7070/dit_exposure7072.pt`。历史组和盲听结果不覆写。
