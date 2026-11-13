# AudioCaps 训练规模对照

既有 T2A-only 训练中，前后 50 次更新的 loss 均值从 1.03938 降至 1.02146，固定 held-out T2A loss 从 0.93434 升至 0.93730。统一协议下 AudioCaps CFG=1 和 6 均低于 Base。这些观察与小训练集上的泛化退化相容。现有结果还无法分离训练集大小、优化制度和数据域。

## 训练数据

从 OpenSound/AudioCaps 的 train 分片获取音频，固定版本 `b29b3243d6ce49c2cd0d48d4b5f0701ae7969ded`。按 AudioCaps 官方 train CSV 核对 `audiocap_id`、YouTube ID 和起始时刻，采用官方 caption。训练池去重后每个 YouTube 视频最多一条。排除官方验证和测试视频、现有 AudioCaps test 清单中的视频和 MusicCaps 评测视频。额外比对现有参考文件的 SHA256。SHA 核对覆盖字节级相同的文件，跨编码排除主要依赖视频标识。

确定性选择 7070 条合格音频，再固定抽取其中 707 条组成小池。音频导出为 PCM16 FLAC，逐个 parquet 分片下载并复用临时文件。源版本、官方 CSV SHA256、拒绝计数、manifest SHA256 和实际曝光的唯一音频数写入 `audit.json`。两个新池都来自相同来源；旧 707 音频实验只作历史参照。

## 固定训练配置

更新后的正式矩阵包括 7070 条的 Pure、T2A-only、Mix 及 707 条的同源 T2A-only。四组均从原始 Audio-Omni 出发，各使用 7072 micro-batch、累积 4、1768 次计划更新。继续使用 10 秒输入、fp32 主权重、AMP fp16、Adafactor、峰值 LR 2e-5、352 曝光 warmup、曝光索引 cosine 曲线和 CFG dropout 0。Qwen 和 Audio VAE 冻结。三个大池组共享同一无放回的音频遍历和逐曝光随机种子。Mix 每次更新含两条 ReCA 和两条 T2A，共各 3536 次曝光。小池以独立的无放回遍历重复约十轮，使用相同逐曝光随机种子和优化配置。

7072 是覆盖 7070 条且整除累积步长 4 的最小曝光量。它用于检验见过一轮大池后的变化，收敛所需训练量仍需验证。0、2000、4000、7072 曝光记录同一历史 held-out 的 loss 和梯度诊断；仅保存最终 checkpoint。7072 曝光矩阵与历史 2000 曝光实验的训练预算和采样方式不同，数据规模效应主要由本轮同源的两组 T2A-only 在等预算下识别。Pure/Mix 是相同大池上的任务对照，Mix 的单任务剂量只有各纯任务组的一半。

正式生成比较沿用固定的 89-SHA AudioCaps prompt、其中 81 个主要分析 SHA、441000 窗口、100 步及配对 seed。优先比较 CFG=6 的大池减小池 CLAP，再检查 CFG=1；Pure/Mix 同时比较音频条件重建。

## 服务器位置

- 数据与 `audit.json`：`/home/linchudeng/Reca-Audio/runs/Benchmark/_data/audiocaps/official_train_scale_v1/`
- 下载日志：`/home/linchudeng/Reca-Audio/runs/train/audiocaps_scale_t2a_v1.launcher.log`
- 新队列日志：`/home/linchudeng/Reca-Audio/runs/train/audiocaps_scale_matrix_v2.launcher.log`
- 训练日志与实际覆盖率：`/home/linchudeng/Reca-Audio/runs/train/audiocaps_scale_matrix_v2/` 下的各组 `.log` 和 `coverage.json`
- 最终权重：各组子目录中的 `dit_exposure7072.pt`

下载完成后由新队列检查磁盘与 GPU 空闲状态。三个大池组先用三对 GPU 并行启动，小池 T2A-only 在大池 T2A-only 完成后复用 GPU。按旧 checkpoint 大小为四组终点权重预留磁盘空间。若预计超过 400 GiB，队列在训练前退出。
