# 相同 caption 预算下，ReCA 梯度能否改善文本生成？

比较已完成的低 LR 交错 Mix 与一组新训练。两组从同一 Audio-Omni Base 出发，使用同一份逐槽调度。对每次 DiT 更新，新组仅移除两个 ReCA 梯度。这个对照可以估计，在已固定的文本监督上加入这些音频重建梯度，对该配方的 T2A 终点有何影响。

| 项目 | 低 LR Mix | 新对照 |
|---|---:|---:|
| 调度位置 / 计划更新 | 7,072 / 1,768 | 7,072 / 1,768 |
| 贡献梯度的 T2A 曝光 | 3,536 | 3,536 |
| 贡献梯度的 ReCA 曝光 | 3,536 | 0 |
| 每次更新的 loss | (T₁ + T₂ + R₁ + R₂) / 4 | (T₁ + T₂) / 4 |

其中 T、R 分别是一个样本的 T2A 和 ReCA rectified-flow loss。每组使用历史 Mix 原有的两条 caption、槽位、裁剪、prompt 和 flow 种子。CPU 上冻结的 VAE 编码仍要从全局随机数序列采样，因此新组也在 ReCA 槽重放冻结的 VAE 和条件准备，再丢弃结果。该槽不执行 DiT 训练。诊断阶段两种任务都可以求梯度，诊断梯度不参与更新。

数据池为 AudioCaps 官方 train 中的 7,070 条独立音频。两组保持 10 秒窗口、4× 低通音频条件、静音声学提示、冻结 Qwen/VAE、全量 DiT、micro-batch 1、fp32 主权重、AMP fp16、seed 42。Adafactor 使用外部 LR，warmup 352 个位置，峰值 5e-6，随后 cosine 衰减，每次更新取四个原位置的 LR 均值。裁剪阈值为 1.0，weight decay 为 0.01。加入 ReCA 后，总梯度、裁剪和 Adafactor 状态均可变化；本对照衡量包含这些优化效应的净结果。

## 评测约定

主要比较量为 **Mix 减新对照** 的 AudioCaps CFG=6 配对文本 CLAP，沿用 89 条中预定的 81 个音频 SHA，使用相同 prompt、生成 seed、441000 个请求采样点、100 步和评分器。给出按 SHA 配对 bootstrap 的 95% 区间。AudioCaps CFG=1、MusicCaps CFG=6、CLAP-FAD、PANNs FD、KL、IS 及正确/错配音频重建作为辅助结果。Base 提供绝对能力参照。分布指标覆盖各自完整 89 条；客观指标无法替代听感评价。

若配对差值区间在零以上，支持这批 ReCA 监督在该配置下改善文本生成；若仅重建提升，文本迁移仍缺乏证据；若区间跨零，保留方向和不确定度。单训练 seed 和反复使用的评测集合限制泛化推断。

## 运行和复现

- 启动入口为 `runs/scripts/train_audiocaps_mix_text_slots.sh`，服务器输出根为 `/home/linchudeng/Reca-Audio/runs/train/audiocaps_mix_text_slots_v1/`。日志为 `text_slots_7070.log`，逐步记录为 `text_slots_7070/metrics.jsonl`。终点文件为 `text_slots_7070/dit_exposure7072.pt`，训练结束时写入 `COMPLETE`。
- 训练调度直接复制自 `runs/train/audiocaps_low_lr_v1/mix_7070_schedule.jsonl`，SHA256 为 `bd5ca9ffa653c06af7d0c1eefb1e36e2fbf6ed09f8bf1b496afb9da1107126a0`。启动脚本校验训练 manifest、历史 held-out 交集记录、Base 和模型配置。只保存一个终点，无需重建特征缓存。
- 2026-11-21 为腾出存储，逐一核对旧 b6 合并模型的 2,420 个张量（含 554 个 DiT 张量）的形状、类型和数值后移除了 32.01 GiB 合并副本。保留的权重可以恢复相同模型状态。在服务器项目根目录执行以下命令即可重建。

```bash
PYTHONPATH=src:Audio-Omni ~/miniconda3/envs/audio-omni/bin/python -m reca_audio.train.merge_ckpt \
  --ckpt Audio-Omni/model/model.ckpt --dit runs/train/reca_b6/dit_step002000.pt \
  --out Audio-Omni/model/model_reca_b6.ckpt
```

前序依据见[低 LR 评测](2026-11-18-low-lr-results/analysis.md)和[顺序与交错训练配对评测](2026-11-19-sequential-results/analysis.md)。

## 实际终点与评测入口

训练于北京时间 2026-11-21 16:06:24 完成，部署提交为 `28d590a`。终点文件共 12,162,314,250 字节，SHA256 为 `ee905bd85d4ec5dbe1078105bfba342d6917d62c442423d694d3c0a368972e6b`，训练端记录已通过重载。核对了全部 1,768 条计划更新记录，每条都保留原 Mix 的两个文本位置、四槽损失分母和相同 LR。3,536 次文本曝光覆盖 3,536 条独立训练音频。历史 Mix 的 ReCA 槽覆盖 3,535 条独立音频，与文本槽交集为 1 条。

| 固定 8 样本诊断或训练统计 | Base | 低 LR Mix 终点 | 仅文本槽终点 |
|---|---:|---:|---:|
| T2A flow MSE | 0.934343 | 0.928669 | 0.930800 |
| ReCA flow MSE | 0.946485 | 0.929032 | 0.940335 |
| DiT 相对 Base 的 L2 漂移 | 0 | 0.9486% | 0.8991% |
| 触发梯度裁剪的更新记录 | — | 352 / 1,768 | 99 / 1,768 |

两组诊断 loss 的起点相同，终点均下降。Mix 在两种条件下的终点 MSE 更低，实际裁剪比例也更高。生成能力以配对生成结果判断。

评测命令为 `bash runs/scripts/eval_audiocaps_scale.sh --mix-text-slots`，输出位于 `/home/linchudeng/Reca-Audio/runs/audits/audiocaps_mix_text_slots_eval_v1/`。只生成新组的五套 WAV，复用已校验摘要的低 LR 对照，评分时逐项检查 SHA、prompt、seed、长度和错配源。评测完成以 `COMPLETE` 和包含组间差值的 `results.json` 为准。
