# Base / Mix 完整 DiT block 分段恢复诊断

## 实验与实现

执行版本 `5c8626c`，脚本 `runs/scripts/diagnose_dit_response.py --block-swap`。Base 为原始 Audio-Omni；Mix 为 `runs/train/reca_long_matrix_b_10s_v1/mix/dit_exposure2000.pt`。使用内部 held-out test manifest，8 个不同音频 SHA，各取第一条 caption，下一组 caption 作为错配条件。固定 seed=42，每组共用同一次编码的 10 秒 VAE latent、Qwen 条件、噪声及 t=0.2/0.5/0.8，CFG=1。ReCA 条件沿用 16k→4k→16k 低通瓶颈。没有训练、梯度更新或写出新 checkpoint。

六组各 24 条记录，共 144 条；每条包含 36 个 cross-attention 输出、36 个 block 输出及最终 velocity，已核对完整性。每个交换组均先完整重载 Mix，再恢复指定 Base block，避免组间污染。完整 block 包含该层 self-attention、cross-attention、FFN 和归一化等全部 state_dict 项；区间编号从 0 开始。全部 block 恢复仍保留 Mix 的 block 外参数，因此不等同于完整 Base。

## 结果

表中为 8 个样本 × 3 个 timestep 的均值。velocity 使用完整张量；相对漂移为 `||v_arm-v_base||/||v_base||`；caption 敏感度为 `||v_shuffled-v_matched||/||v_matched||`。两者分母不同。flow MSE 的 target 是 `noise-latent`。

| 配置 | 文本 velocity 漂移 | caption 敏感度 | 匹配文本 MSE | 错配文本 MSE | 音频条件 MSE |
|---|---:|---:|---:|---:|---:|
| Base | 0.00% | 7.54% | 0.91747 | 0.92272 | 0.92082 |
| Mix | 12.61% | 6.60% | 0.92943 | 0.93388 | 0.92898 |
| Mix 恢复 Base block 0–11 | 10.21% | 7.43% | 0.92453 | 0.92871 | 0.92492 |
| Mix 恢复 Base block 12–23 | 8.55% | 6.30% | 0.92282 | 0.92759 | 0.92239 |
| Mix 恢复 Base block 24–35 | 11.33% | 6.68% | 0.92792 | 0.93239 | 0.92761 |
| Mix 恢复全部 Base block 0–35 | 2.77% | 7.32% | 0.91857 | 0.92354 | 0.91972 |

| 配置 | t=0.2 漂移 | t=0.5 漂移 | t=0.8 漂移 |
|---|---:|---:|---:|
| Mix | 12.53% | 13.85% | 11.47% |
| 恢复 0–11 | 10.79% | 11.25% | 8.61% |
| 恢复 12–23 | 7.86% | 9.04% | 8.75% |
| 恢复 24–35 | 10.49% | 12.57% | 10.92% |
| 恢复全部 block | 2.17% | 3.20% | 2.95% |

本轮 Base/Mix 的汇总值与上一轮权重交换诊断一致。恢复中段相对 Mix 降低平均漂移约 32.2%，是三个单独区间中最大者；恢复全部 block 降低约 78.0%。这些是干预后的指标变化，不能作为可加的参数贡献比例。

## 可信解释与局限

1. 在这批固定输入上，完整 block 恢复明显比仅恢复条件映射和前 6 层 cross-attention 更能改变最终 velocity。仅修复早段条件入口不足以恢复 Base 的文本响应。
2. 中段恢复最能降低单段干预下的平均漂移；前段恢复最能提高 caption 敏感度。中段敏感度反而由 6.60% 降到 6.30%，因此“接近 Base”与“更依赖 caption”并不是同一个指标。
3. 恢复全部 block 后仍有 2.77% 漂移，与保留 Mix 的 block 外参数及其交互相容。源码中 block 外路径包括条件、时间和全局嵌入、transformer 输入/输出投影、前后卷积；本轮不能进一步归因到其中某一模块。
4. 单段交换涉及多个子模块，不能据此断言某个 FFN、self-attention 或单层训练不稳定，也不能把漂移变化直接归因为 ReCA 单一任务。当前比较的是已完成 Mix 与 Base。
5. 8 个 SHA 只有 8 个音频组；三个 timestep 不是额外独立样本。错配仅一个循环负样本。没有生成完整音频、运行多步采样或测量 CLAP/FAD，不能证明权重恢复能改善 T2A，也不能由此决定冻结训练区间。音频条件 MSE 同样不是完整 reconstruction 质量。

## 下一步建议

优先做小规模固定 seed 的真实 T2A 生成对照：Base、Mix、恢复中段、恢复全部 block，共用同一 held-out prompt 集、生成时长和采样参数。比较配对 CLAP、试听以及音频条件重建保留情况；小样本阶段不把 FAD 作为主要依据。若单步响应改善能迁移到生成质量，再讨论训练时约束或冻结范围，暂不增加长训练。

## 产物与路径影响

- 远程原始结果：`/home/linchudeng/Reca-Audio/runs/audits/dit_block_swap_v1/results.json`。
- 本次仅修改独立诊断脚本，增加互斥的 `--block-swap` 模式和区间元数据；正常训练、特征缓存、推理入口未改变。
- 无需重新训练或重建缓存；没有保存交换后的模型权重。
