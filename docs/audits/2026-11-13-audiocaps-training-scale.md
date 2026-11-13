# AudioCaps 训练规模对照

既有 T2A-only 训练中，前后 50 次更新的 loss 均值从 1.03938 降至 1.02146，固定 held-out T2A loss 从 0.93434 升至 0.93730。统一协议下 AudioCaps CFG=1 和 6 均低于 Base。这些观察与小训练集上的泛化退化相容。现有结果还无法分离训练集大小、优化制度和数据域。

## 训练数据

从 OpenSound/AudioCaps 的 train 分片获取音频，固定版本 `b29b3243d6ce49c2cd0d48d4b5f0701ae7969ded`。按 AudioCaps 官方 train CSV 核对 `audiocap_id`、YouTube ID 和起始时刻，采用官方 caption。训练池去重后每个 YouTube 视频最多一条。排除官方验证和测试视频、现有 AudioCaps test 清单中的视频和 MusicCaps 评测视频。额外比对现有参考文件的 SHA256。SHA 核对覆盖字节级相同的文件，跨编码排除主要依赖视频标识。

确定性选择 7070 条合格音频，再固定抽取其中 707 条组成小池。音频导出为 PCM16 FLAC，逐个 parquet 分片下载并复用临时文件。源版本、官方 CSV SHA256、拒绝计数、manifest SHA256 和实际曝光的唯一音频数写入 `audit.json`。两个新池都来自相同来源；旧 707 音频实验只作历史参照。

## 固定训练配置

两个 T2A-only 组均从原始 Audio-Omni 出发，各使用 2000 micro-batch、累积 4、500 次计划更新。继续使用 10 秒输入、fp32 主权重、AMP fp16、Adafactor、峰值 LR 2e-5、100 曝光 warmup、原曝光索引 cosine 曲线和 CFG dropout 0。Qwen 和 Audio VAE 冻结。两组共用原 T2A-only 的逐曝光数据、flow 与 prompt 随机种子，按相同均匀随机数映射到各自池中的音频索引。大池在固定 2000 曝光下只会访问其中一部分，比较检验的是固定训练预算下扩大样本池的效果。

训练过程继续使用同一历史诊断 manifest 和 0/1000/2000 曝光检查点。正式生成比较应沿用固定的 89-SHA AudioCaps prompt、其中 81 个主要分析 SHA、441000 窗口、100 步及配对 seed。优先比较 CFG=6 的大池减小池 CLAP，再检查 CFG=1，区分提升是否延伸到 conditional 生成。

## 服务器位置

- 数据与 `audit.json`：`/home/linchudeng/Reca-Audio/runs/Benchmark/_data/audiocaps/official_train_scale_v1/`
- 队列日志：`/home/linchudeng/Reca-Audio/runs/train/audiocaps_scale_t2a_v1.launcher.log`
- 训练日志：`/home/linchudeng/Reca-Audio/runs/train/audiocaps_scale_t2a_v1/t2a_{707,7070}.log`
- 最终权重：`/home/linchudeng/Reca-Audio/runs/train/audiocaps_scale_t2a_v1/t2a_{707,7070}/dit_exposure2000.pt`

数据准备通过后由同一队列检查空闲 GPU 并启动两个训练组。预估用户目录占用约 370–380 GiB，含两组最终 checkpoint。
