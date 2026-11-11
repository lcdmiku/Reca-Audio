# Qwen / Projection / DiT 条件对齐诊断

日期：2026-11-11  
服务器产物：`runs/audits/condition_alignment_10s_sha_v1/`

## 1. 诊断合同

- 数据：AudioCaps held-out，89 个唯一音频 SHA、445 条 caption，每个音频 5 条正 caption。
- 音频条件：完整 10 秒，按当前 ReCA 配置做 4× 低通，不做额外随机截断。
- 文本条件：实际 T2A prompt 模板。
- Qwen：冻结；读取全部 37 个 hidden-state 条目，当前配置使用索引 35（`layer_idx=-2`）。
- Projection：冻结的 RMSNorm + `2048→768` `proj_features`。
- DiT condition：比较 Base、10 秒 Pure、T2A、Mix 的 `to_cond_embed`（`768→2048→2048`）。
- 指标：匹配/非匹配 cosine、margin、双向 Recall@1、linear CKA、RBF MMD、特征 norm。
- 不进行不同长度 token 的逐位置 cosine；CKA 在音频组级别对 5 条正 caption 求均值后计算。

完整结构化指标见 [results.json](results.json)，自动逐层表见 [report.md](report.md)。

## 2. 协议有效性检查

| 检查 | 结果 |
|---|---:|
| audio groups | 89 |
| text captions | 445 |
| audio `<|AUDIO|>` token / sample | 固定 250 |
| text 中 audio token | 固定 0 |
| audio semantic-mask fallback | 0/89 |
| text semantic-mask fallback | 0/445 |
| Qwen hidden-state 数 | 37 |
| 当前 hidden index | 35 |

早期试运行曾错误按文件路径得到 445 个音频组；根因是五条 caption 对应的文件路径不同但
内容 SHA 相同。该结果和另一次丢弃重复 caption 的 432-caption 中间结果均已删除，未用于
本报告。最终脚本在模型加载前强制校验 89/445 合同。

随机 Recall@1 基准在两个方向上均为 `1/89 = 1.12%`：Audio→Text 有 5/445 个正文本，
Text→Audio 有 1/89 个正音频。

## 3. Qwen Thinker 层间结果

### 当前配置层 35 与最终层 36

| Layer / view | matched cos | unmatched cos | margin | A→T R@1 | T→A R@1 | CKA | MMD |
|---|---:|---:|---:|---:|---:|---:|---:|
| 35 content | 0.8732 | 0.8634 | 0.0098 | 12.36% | 7.19% | 0.5656 | 0.5991 |
| 35 semantic | 0.8543 | 0.8517 | 0.0026 | 3.37% | 2.25% | 0.5112 | 0.7289 |
| 35 full | 0.9329 | 0.9307 | 0.0022 | 11.24% | 1.35% | 0.4406 | 0.7181 |
| 35 last | 0.8019 | 0.7941 | 0.0079 | 6.74% | 4.27% | 0.6738 | 0.6955 |
| 36 content | 0.3386 | 0.3009 | **0.0378** | 12.36% | 2.47% | 0.5244 | 0.6256 |
| 36 full | 0.4711 | 0.4556 | 0.0155 | 12.36% | 1.12% | 0.4292 | 0.6997 |

源码事实：当前 conditioner 使用层 35，而不是最终层 36。层 35 的 matched cosine 很高，
但 unmatched cosine 几乎同样高；绝对 cosine 不能证明模态对齐。它的 content 双向检索
显著高于随机，说明确有语义结构，但表示仍处于高度集中的模态相关分布中。

最终层 36 的 margin 最大，但并非所有指标最好：content T→A R@1 从层 35 的 7.19%
降到 2.47%。content A→T 最佳层为 21（19.10%），T→A 最佳层为 33（17.08%），层 35
则具有最高 content CKA（0.5656）。因此不能仅凭某一个 margin 把 hidden layer 改成最终层。

## 4. Projection 与 DiT 条件表示

### Content view（audio token vs caption token）

| Stage | margin | A→T R@1 | T→A R@1 | CKA | MMD ↓ |
|---|---:|---:|---:|---:|---:|
| Qwen layer 35 | 0.0098 | 12.36% | 7.19% | 0.5656 | 0.5991 |
| Frozen projection | 0.1629 | 25.84% | 24.94% | **0.5999** | 0.3884 |
| DiT Base | 0.1609 | 24.72% | 23.82% | 0.5721 | 0.4387 |
| DiT Pure | **0.2064** | **33.71%** | 23.60% | 0.5851 | **0.3071** |
| DiT T2A | 0.1707 | 24.72% | 23.60% | 0.5690 | 0.4140 |
| DiT Mix | 0.1953 | 30.34% | **24.04%** | 0.5840 | 0.3448 |

### Full valid sequence view

| Stage | margin | A→T R@1 | T→A R@1 | CKA | MMD ↓ | audio/text norm |
|---|---:|---:|---:|---:|---:|---:|
| Frozen projection | 0.1026 | 23.60% | 20.90% | **0.5810** | 0.5281 | 10.47 / 5.64 |
| DiT Base | 0.1619 | 21.35% | 24.04% | 0.5583 | 0.3883 | 1.92 / 0.76 |
| DiT Pure | **0.2122** | **29.21%** | **25.62%** | 0.5728 | **0.2811** | 1.44 / 0.77 |
| DiT T2A | 0.1774 | 22.47% | 23.15% | 0.5581 | 0.3591 | 1.94 / 0.72 |
| DiT Mix | 0.2000 | 26.97% | 23.82% | 0.5704 | 0.3207 | 1.59 / 0.75 |

冻结 projection 并非简单保持 Qwen 空间：它把 content margin 从 0.0098 提高到 0.1629，
双向检索提高到约 25%，同时降低 MMD。说明原始 Audio-Omni 已学习到显著的跨模态投影，
不能把 Qwen raw hidden 与 DiT 实际条件等同。

Pure/Mix 后训练没有破坏 `to_cond_embed` 处的池化跨模态可分性。相反，Pure 最强、Mix
次之、T2A 再次，三者均高于 Base 的 full-sequence margin；Pure/Mix 的 MMD 也更低。

## 5. 与生成结果联合解释

10 秒 AudioCaps T2A CLAP 为 Base 0.4610、Pure 0.4386、T2A 0.4310、Mix 0.4286。
这与条件嵌入的 margin 排序（Pure > Mix > T2A > Base）不一致。由此可以排除一个过强解释：

> “T2A 下降是因为 Qwen projection 或 DiT `to_cond_embed` 把音频和文本表示拉得更远。”

本诊断支持的是：

1. Qwen raw audio/text 分布并不天然接近，只有弱到中等的跨模态检索结构；
2. 冻结 projection 是当前跨模态间距显著缩小的主要边界；
3. Pure/Mix 会进一步强化池化后的匹配结构，但这并不转化为更好的文本生成；
4. 核心失败更可能发生在 `to_cond_embed` 之后：cross-attention 如何读取条件、DiT block
   对 audio/text context 的响应差异，或全模型更新对原始文本去噪映射的干扰；
5. acoustic shortcut 仍可能存在，因为池化检索只验证语义匹配，不能测量音频 token 中
   可供重建复制的时间/音色/环境细节。

## 6. 下一步最小诊断

暂不启动新长训练。下一步应固定同一 latent、noise、timestep 与 caption/audio 对，捕获
Base/Pure/Mix 每个 DiT block 的：

- cross-attention 输出及其 audio/text cosine、CKA、norm；
- block 输出相对 Base 的漂移；
- 匹配与打乱 caption 的响应差；
- audio-conditioned 与 text-conditioned predicted velocity 的差异。

如果条件在 `to_cond_embed` 处对齐、进入若干 cross-attention block 后迅速分离，就能定位
catastrophic interference 的具体层。只有在该诊断后，才决定冻结哪些层或加入 Base-text
distillation；现有结果不支持仅凭 Qwen 最终层 margin 直接切换 hidden layer。

