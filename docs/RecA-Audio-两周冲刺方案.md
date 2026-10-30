# 两周冲刺：在 8×V100-32GB 上拿到 RecA 迁移的初步结果

> 目标不是"做完整个课题"，而是两周后能拿出一句可交付的话：
> **在某个真实音频生成模型上，用 RecA 做无标注后训练后，X 指标从 A 变成 B；并附 100 条定性样例与一张诊断图。**
> 上游方案见 [RecA-Audio-实验思路.md](RecA-Audio-实验思路.md)、[RecA-Audio-执行方案-课程报告版.md](RecA-Audio-执行方案-课程报告版.md)。

---

## 0. 结论

**能，但要把范围砍到"一个基座 + 一个域 + 一组对照"。**

- 算力不是瓶颈：8×V100 峰值约等于 2.3 张 A100，两周可用 2000+ 卡时，本计划只需 **300–400 V100·h**。
- 瓶颈是**依赖能否在 D1 跑通**：V100 没有 bf16、没有 FlashAttention，2026 年的开源仓库默认假定 Ampere+，这是两周里最可能烧掉 3–5 天的地方。
- 因此架构上必须先定死一条：**冻结理解塔 + 离线缓存特征 + 只训生成头**。这一条同时解决显存、速度、依赖三个问题。

---

## 1. V100 的五条硬约束

| 约束 | 事实 | 工程后果 |
|---|---|---|
| **无 bf16** | BF16 需要 sm80+，V100 是 sm70 | 必须 `fp16 + GradScaler`；所有 `autocast(dtype=bfloat16)` 都要改，否则直接报错 |
| **无 FlashAttention** | FA2 官方要求 sm80+ | 强制 `attn_implementation="eager"` 或 `sdpa`（mem-efficient 后端），速度再打 6–7 折 |
| **CUDA 13 不再支持 Volta** | 驱动 580 可向下兼容 | 装 **cu121 / cu124** 的 torch，不要装 cu130 轮子 |
| **32 GB / 卡** | 7B 模型 fp16 权重就 14 GB | 全参训练 7B 不可能；必须冻结 + 分片 + 缓存 |
| **无 TF32** | sm80+ 才有 | 不要指望 `allow_tf32` 提速；fp16 是唯一加速手段 |

**换算**：V100 FP16 峰值 ~125 TFLOPS，A100 ~312；实际训练效率再打 0.6（无 FA、fp16 溢出处理），**8×V100 ≈ 1.3–1.5 张 A100 的有效算力**。

---

## 2. 架构决策：为什么必须"冻结 + 缓存"

真实基座（Audio-Omni 类）的结构是「冻结 MLLM 理解塔 + 可训 DiT」。如果每个 step 都跑一遍 7B 前向：

- 权重 14 GB + DiT 优化器状态 + 激活 → 32 GB 装不下，只能上 ZeRO-3/CPU offload，速度再掉一半；
- 每 step 多一次 7B 前向（约 7 TFLOP/sample），V100 上 batch=8 就要 1.4 s/step，纯属浪费。

**正确做法**：理解塔冻结 ⇒ 它的输出对每个样本是常量 ⇒ **离线跑一遍，把 hidden states 落盘，训练时只读特征**。

- 训练成本从「7B 前向 + DiT 反传」降到「DiT 反传」，显存降到 1/10；
- 缓存体积可控：10 s 片段、池化到 256 token、fp16 ⇒ 约 1.8 MB/条，5 万条约 90 GB；
- 附带好处：换编码器/压缩率只需重算特征，**倒 U 曲线的 12 组实验变成 12 次小训练**，这正是两周内还能塞下机制实验的原因。

---

## 3. 两条路径，共享 70% 代码

| | Path A：预训练基座（加分） | Path B：自建统一模型（保底） |
|---|---|---|
| 模型 | Audio-Omni（冻结 MLLM + DiT）或 UniAudio 2.0 | 300M DiT + flow matching，条件 = 冻结 MERT/CLAP |
| 依赖 | 权重与代码是否公开（**不确定**） | 全部公开，零外部依赖 |
| 两周可行性 | 权重当天能下 → 可行；否则不可行 | 确定可行 |
| 结果说服力 | 高（"真实开源模型"） | 中高（真实数据、真实规模，但不是预训练模型） |

**共享组件（先写，两条路都吃）**：数据管线 → 特征缓存（5 个编码器 × 3 档压缩）→ 训练循环（fp16/flow matching）→ 评测脚本 → 听感测试页。

> 结论：**Path B 先起步，Path A 在 D1 用半天验证权重可用性**。不要为了等权重而停摆。

---

## 4. 十四天日计划

| 天 | 任务 | 产出 / Gate |
|---|---|---|
| **D0** | 解决命令行权限；确认 8 卡无他人占用 | 能跑命令 |
| **D1** | 装 torch cu124；fp16 冒烟测试；跑通一个 DiT 前向反传 | **G1：在没有 bf16/FA 的前提下训练能跑** |
| **D2** | 数据落地：FSD50K + FMA/Jamendo 子集，重采样、切 10 s、响度归一 | 50–100 h 可用数据 |
| **D3** | 特征预计算：CLAP / BEATs / MERT / AudioMAE / Whisper × 3 档压缩，8 卡并行 | 特征库 + 体积报告 |
| **D4** | 基线 DiT + 文本条件；过拟合 100 条 | **G2：100 条能重建到"听得出是同一段"** |
| **D5** | 基线正式训练（8 卡 DDP，约 12 h）；评测脚本跑通 | Base checkpoint + 指标基线 |
| **D6** | **E1 诊断**：用自身特征重建 100 条，人工听 + 指标 | **G3：确认有 headroom，否则换域/换条件** |
| **D7–8** | RecA 训练（首选 MERT 帧级条件）+ h-dropout 变体 | 第一个 RecA checkpoint |
| **D9** | 对照：CLAP 全局条件（信息量≈caption 的替身）同算力训练 | 对照 checkpoint |
| **D10** | 双方评测 + 小规模听感（10 条 × 5 人） | **G4：方向是否成立** |
| **D11** | 编码器三点扫描（CLAP / MERT / AudioMAE，各 8k step，8 卡分 3 组） | 倒 U 的初步三点 |
| **D12** | 复跑最好配置 + 换 seed | 方差估计 |
| **D13** | 出图：诊断图、主对比、倒 U 三点 | 三张图 |
| **D14** | 报告初稿 + 组会 | 可交付 |

**四个 Gate 的意义**：任何一个不过，立刻降级（见 §7），不要硬扛。

---

## 5. 数据与最小评测集

| 用途 | 选择 | 理由 |
|---|---|---|
| 音效训练 | FSD50K | CC-BY，直链下载稳定，~100 h |
| 音乐训练 | FMA-medium 子集 / MTG-Jamendo | CC 许可，直链 |
| 留出评测 | 各自 test split + 200 条自建"细节提示词" | 避免污染 |
| 听感 | 10 条 × 5 人 A/B（自然度、细节、文本一致性） | 课程报告最有说服力的一栏 |

**最小指标集**：FAD（VGGish + CLAP 两套）、CLAP score、beat F1（音乐）、BEATs 事件探针（音效）、人评。

---

## 6. 对照设计的巧思：用 CLAP 全局向量当 caption 替身

音乐/音效**没有 caption**，做不了"caption SFT"对照。但真正要隔离的变量是「条件的信息密度」，不是"有没有文本"。

> **对照组 = 同一模型、同一数据、同一算力，但条件换成 CLAP 全局向量（512-d，一条）**。

这等价于"caption 级信息量"，而 RecA 组是帧级（数百 token）。两组一比，"密度"这个变量就被干净地隔离出来了——比硬凑 caption 更严谨，也省掉标注。

---

## 7. 风险与降级

| 风险 | 概率 | 降级方案 |
|---|---|---|
| 权限/环境卡住 3 天以上 | **高**（已发生） | 先切完全权限；环境问题并行让同学代跑 |
| 预训练权重拿不到 | 中高 | 只做 Path B，报告里注明 |
| 7B 基座在 32 GB 上仍 OOM | 中 | 片段缩到 5 s、batch=1 + 梯度累积、只训 DiT 的 LoRA |
| 数据下载太慢 | 中 | 先用 FSD50K 单域（100 h）跑通，音乐后补 |
| 训练不收敛 | 中 | 先过拟合 100 条；flow matching 的 loss 用 fp32 算 |
| 两周只出"无提升" | 中 | 这仍是结果：写"重建保真↑但 T2A 不动的瓶颈分析"，配 E1/E2 图 |

---

## 8. 两周能交付 / 不能交付

**能**：
- 1 个真实音频生成模型上 Base vs RecA vs 密度对照 的初步对比；
- 1 张重建诊断图 + 100 条定性样例；
- 若顺利，编码器三点扫描的初步倒 U 趋势；
- 一次小规模听感测试。

**不能**（推到第 3–4 周）：
- 完整倒 U 曲线（12 组）、跨架构复现、音频编辑子任务、长音频、大规模人评、与 GRPO 的结合。

---

## 9. D0 环境自检清单（命令可直接用）

```powershell
# 1) 装对轮子（不要 cu130）
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu124

# 2) 确认 sm70 可见
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability(0))"
# 期望：(7, 0)

# 3) 确认 fp16 可用、bf16 会报错（这是预期行为，不是 bug）
python -c "import torch;x=torch.randn(64,64,device='cuda',dtype=torch.float16);print((x@x).sum().item())"

# 4) 确认 SDPA 走得通
python -c "import torch;q=torch.randn(1,8,256,64,device='cuda',dtype=torch.float16);print(torch.nn.functional.scaled_dot_product_attention(q,q,q).shape)"

# 5) 盘位（特征缓存要 100–200 GB）
Get-PSDrive -PSProvider FileSystem | Select-Object Name,@{n='FreeGB';e={[math]::Round($_.Free/1GB,1)}}
```

**代码里要改的三处**（几乎所有 2026 仓库都会踩）：

```python
# (a) bf16 -> fp16 + scaler
scaler = torch.cuda.amp.GradScaler()
with torch.autocast("cuda", dtype=torch.float16):
    loss = model(**batch)
scaler.scale(loss).backward(); scaler.step(opt); scaler.update()

# (b) 关掉 flash-attn
model = AutoModel.from_pretrained(..., attn_implementation="eager")

# (c) 关掉 compile（Triton 在 Volta 上问题多）
# torch.compile(...)  ->  注释掉
```
