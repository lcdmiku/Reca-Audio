# CFG geometry 与生成 sweep

## 固定 latent 的四组响应

正式诊断从 81 个主要分析 SHA 中按合同顺序取前 8 个。每个样本复用同一 VAE target latent、噪声和三个 timestep（0.2、0.5、0.8）。对每个条件类型，均与 Base 在相同输入上的向量比较。每个模型每种条件 24 条记录，共 192 条。独立提取推理 CFG=2 实际批量 forward 的 conditional 和 unconditional 两个输出。CFG=1、4、6 也直接调用推理 forward。

表中漂移为 `||v−v_base||₂/||v_base||₂` 的平均值。Guidance cosine 和 norm ratio 以同条件类型的 Base guidance 为参照。

| 条件 | 模型 | cond 漂移 | uncond 漂移 | CFG 1 | CFG 2 | CFG 4 | CFG 6 | Guidance cosine | Guidance norm ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Text | Base | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| Text | Pure | 0.13028 | 0.13432 | 0.13029 | 0.15638 | 0.23160 | 0.30445 | 0.53426 | 0.68174 |
| Text | T2A-only | 0.13085 | 0.12893 | 0.13085 | 0.15766 | 0.22980 | 0.29906 | 0.56737 | 0.61458 |
| Text | Mix | 0.12864 | 0.12868 | 0.12864 | 0.15644 | 0.23192 | 0.30447 | 0.53147 | 0.66298 |
| Audio | Base | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| Audio | Pure | 0.14043 | 0.13432 | 0.14044 | 0.17109 | 0.24855 | 0.32230 | 0.50962 | 0.91157 |
| Audio | T2A-only | 0.13281 | 0.12893 | 0.13283 | 0.15526 | 0.21490 | 0.27345 | 0.65792 | 0.91602 |
| Audio | Mix | 0.13978 | 0.12868 | 0.13979 | 0.17207 | 0.24815 | 0.31978 | 0.53106 | 0.95320 |

文本条件下，三组 CFG=1 velocity 已偏离 Base 约 13%，所以模型变化不局限在 unconditional 分支。CFG=6 将平均偏离扩大到约 30%。三组文本 guidance 的范数为 Base 的约 61%–68%，与 Base 的方向 cosine 为 0.53–0.57。Audio 条件的 guidance 范数保留较多，但方向同样发生改变。Pure 与 Mix 的 audio guidance 方向更接近彼此各自相对 Base 的漂移水平，T2A-only 在此条件下的方向变化较小。此表分别比较同模态的模型响应，未做 audio 与 text 向量的逐元素跨模态比较。

在相同的固定输入上，Text CFG=1 单步 Flow MSE 为 Base 0.98284、Pure 0.99606、T2A-only 0.99491、Mix 0.99519。Text CFG=6 分别为 1.14989、1.08343、1.05937、1.09006。高 CFG 的大漂移和较低单步 MSE 在这里可以同时出现。生成语义分数须由真实采样得到。

## 100 步生成

四组共用 89 条 AudioCaps prompt、seed 与两分片映射，主要统计排除 8 个探索 SHA。10 秒窗口下使用 CFG 1、2、4、6。生成结果完成后，正式运行目录将写出同名报告，包含各点 CLAP、对 Base 的配对 SHA bootstrap 区间，以及每组实测最高 CFG 点。

原始 geometry 记录位于服务器 `/home/linchudeng/Reca-Audio/runs/audits/posttrain_four_arm_v1/geometry.json`。
