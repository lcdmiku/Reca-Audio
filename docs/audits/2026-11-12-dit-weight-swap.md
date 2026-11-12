# Base / Mix DiT 权重交换

8 个 AudioCaps held-out 音频 SHA，每组取首条 caption，循环使用下一组 caption 作为错配条件。六组共用同一次编码得到的 latent 和条件，使用固定噪声与 `t=0.2/0.5/0.8`。每组 24 条记录，每条包含 36 层 cross-attention、36 个 block 输出和最终 velocity。逐层激活沿 latent 时间轴池化，velocity 使用完整张量。相对漂移以同输入的 Base 输出范数为分母。caption 敏感度以各自匹配文本输出的范数为分母。

| 配置 | 层 0 attention 漂移 | 层 5 attention 漂移 | 层 6 attention 漂移 | velocity 漂移 | velocity caption 敏感度 | 匹配文本 flow MSE |
|---|---:|---:|---:|---:|---:|---:|
| Base | 0% | 0% | 0% | 0% | 7.54% | 0.91747 |
| Base 换入 Mix 条件映射 | 18.73% | 27.92% | 30.10% | 0.96% | 7.53% | 0.91768 |
| Mix | 25.48% | 58.41% | 51.18% | 12.62% | 6.60% | 0.92943 |
| Mix 恢复 Base 条件映射 | 12.38% | 39.51% | 34.93% | 12.59% | 6.64% | 0.92939 |
| Mix 恢复 Base 第 0–5 层 attention | 20.71% | 27.99% | 51.18% | 12.60% | 6.62% | 0.92939 |
| Mix 同时恢复两部分 | 7.42% | 0.22% | 34.93% | 12.58% | 6.66% | 0.92934 |

条件映射指 `to_cond_embed`。早段 attention 交换包括 cross-attention 权重和输入归一化。只换入 Mix 映射能改变 Base 的早层响应，但最终 velocity 漂移为 0.96%。在 Mix 同时恢复两部分后，第 5 层局部响应明显接近 Base，第 6 层仍有 34.93% 漂移，最终 velocity 漂移仍为 12.58%。在这组输入上，两处恢复对最终文本响应的作用很小。剩余权重及其交互仍需通过进一步干预才能定位。

本轮新增了 PyTorch 全局随机种子，比较以本轮重跑的 Base 和 Mix 为基准。样本量仅 8 个音频组，flow MSE 和 caption 敏感度均为诊断指标，生成质量需要单独评测。

服务器原始数据位于 `/home/linchudeng/Reca-Audio/runs/audits/dit_swap_v1/results.json`，执行版本为 `ce2d023`。
