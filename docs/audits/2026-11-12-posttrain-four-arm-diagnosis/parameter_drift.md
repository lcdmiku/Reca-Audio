# 四组 DiT 参数漂移

直接读取 checkpoint 存储精度的参数，对同名张量计算 `||θ−θ_base||₂ / ||θ_base||₂`。各组比较 3,037,311,104 个参数。按模块平方和聚合，扫描核对张量形状和有限值。下表为比例转为百分数，Base 对自身均为 0。

| 模块 | Base | Pure ReCA | T2A-only | Mix |
|---|---:|---:|---:|---:|
| Global | 0 | 2.0209% | 2.0031% | 2.0191% |
| Self-attention | 0 | 1.8242% | 1.8252% | 1.8311% |
| Cross-attention | 0 | 3.3665% | 3.1843% | 3.2945% |
| FFN | 0 | 2.2072% | 2.2131% | 2.2167% |
| Block norm | 0 | 0.0379% | 0.0377% | 0.0375% |
| to_cond_embed | 0 | 2.0908% | 1.6940% | 1.8737% |
| Input projection | 0 | 0.4737% | 0.4765% | 0.4772% |
| Output projection | 0 | 0.6031% | 0.6007% | 0.6069% |
| Timestep embedding | 0 | 1.6942% | 1.7183% | 1.7193% |
| Block 35 | 0 | 3.1428% | 3.0799% | 3.1311% |

三个训练组最大的五个 block 漂移均来自 31–35 层。Pure 的条件映射和 cross-attention 漂移比 T2A-only 更大，Mix 介于两者之间。三个模型其余主要子模块的全局漂移接近。前期完整 block 干预中，恢复 Mix 中段对最终 velocity 的影响大于恢复末段，且恢复中段没有提高 8-item T2A CLAP。参数差异、响应差异和生成分数需要分别报告。

原始逐 block 与子模块记录在服务器 `runs/audits/posttrain_four_arm_smoke_v1/parameter_drift.json`。此处参数比较使用正式 checkpoint，和 smoke 的样本数、采样步数无关。
