# ReCA 后训练

把 Audio-Omni 的生成头（DiT）对齐到理解塔的表征：**给一段音频，让理解塔编码它，
生成头再把它重建成同一段音频**。这是 ReCA 在音频上的直接搬运。

## 结构

```
audio ──44.1kHz──► VAE.encode ──► latent ──┐
                                          ├─► rectified flow loss
audio ──16kHz────► Qwen2.5-Omni thinker ──┘
                       │
                       └─► cross-attn cond ──► DiT(预测 v)
```

- **冻结**：VAE（oobleck）、理解塔（Qwen2.5-Omni thinker 全部参数）
- **训练**：DiT 全部参数（**3037.3 M**）
- 条件走 conditioner 的 `omni_prompt.audio_prompt`，文本用固定指令
  `"Describe the audio."`（保证条件信息只来自音频）

### 防泄漏

`audio_input_prompt`（mel 条件）如果喂原音频，模型可以直接“抄”目标。
所以 ReCA 模式下该字段固定为零音频（10 s，对应 config 里写死的 1723 帧投影层）。
注意：扩散头的 `sample_size` 是 485100（11 s），和 mel 条件的 10 s 不是同一个长度。

## 关键文件

| 文件 | 作用 |
|---|---|
| `src/reca_audio/train/data.py` | manifest 数据集，同时给 44.1 kHz / 16 kHz 两路 |
| `src/reca_audio/train/conditioning.py` | ReCA / T2A 两种 metadata |
| `src/reca_audio/train/flow.py` | rectified flow 损失（t~U(0,1)，回归 x1-noise） |
| `src/reca_audio/train/train.py` | 训练主循环 |
| `src/reca_audio/train/make_manifest.py` | prompts.jsonl + 音频目录 → 训练 manifest |
| `src/reca_audio/train/merge_ckpt.py` | DiT 权重合并回完整 ckpt |
| `runs/scripts/train_reca.sh` | 启动脚本 |

## 显存策略（V100 32 GB）

3B 参数全量训练在单卡上放不下 AdamW（权重 12 + 梯度 12 + 状态 24 = 48 GB），所以：

| 手段 | 效果 |
|---|---|
| **Adafactor** | 优化器状态 O(n) 而非 2n，省约 2/3 |
| **AMP fp16 autocast** | 激活减半，V100 tensor core 加速 |
| **VAE 放 CPU** | 冻结的 VAE 不占生成卡，编码后再搬回 |
| 5 s 片段训练 | 激活随长度线性下降 |

实测：**3.1 s/step**（batch=1、grad_accum=1、5 s、Adafactor+AMP+VAE on CPU），
GPU0 峰值约 30 GB（30.4 亿参数 fp32 + 梯度 + 激活）。

## 多卡训练（实际采用的方案）

和推理时的组件并行一致：每个进程占两张卡，**偶数卡放 DiT，奇数卡放冻结的理解塔 + VAE**。

```
rank i:  DiT (fp16)      -> cuda:(2i)      ~18.5 GB
         理解塔 + VAE     -> cuda:(2i+1)    ~21 GB
```

4 个进程覆盖 8 张 V100，DDP 做梯度同步。实测 **1.83 s/step**（5 s 片段、batch 1），
2000 步约 65 分钟。

```bash
bash runs/scripts/train_reca_ddp.sh                          # 2000 步
SMOKE=1 bash runs/scripts/train_reca_ddp.sh                  # 2 步冒烟
```

关键取舍：

| 手段 | 原因 |
|---|---|
| DiT 权重 **fp16** | fp32 的 12 GB 参数 + 12 GB 梯度放不下；fp16 后稳定在 18.5 GB |
| 关掉 GradScaler | fp16 参数产生 fp16 梯度，`unscale_` 会直接报错 |
| `gradient_as_bucket_view` + `static_graph` | DDP 梯度桶复用，省 1–2 GB |
| VAE 放辅助卡且强制 fp32 | 与理解塔同卡，编码后只搬几十 MB 回 DiT 卡 |

### 提示词多样化

`runs/scripts/const.py` 的 `get_audio_recon_prompt_list()` 提供 **357 条**音频重建指令
（对应 ReCA 原文“多样化提示词防僵化”）。训练时每条样本随机抽一条：

```python
"text_prompt": sample_recon_prompt()
```

找不到 `const.py` 时退回固定指令 `Describe the audio.`。

### ⚠️ 必须用 `build_prompt("Reconstruct", …)` 包装

第一次正式训练（2000 步）把 const.py 的提示词**裸着**当 user text 喂给理解塔，结果模型塌缩：

| 模型 | 重建同一段音频的频谱平坦度 |
|---|---|
| 原始 Audio-Omni | **0.032**（结构化） |
| 裸提示词训练 2000 步 | **0.600**（白噪声） |
| 加 `Reconstruct` 包装训练 300 步 | **0.048**（接近原始） |

原因：Audio-Omni 预训练用的是 `SYSTEM_RECONSTRUCT` 系统模板，裸文本会让理解塔输出
OOD 特征，DiT 学不到条件、退化成输出噪声（CLIP score 甚至为负）。

正确写法（`conditioning.py`）：

```python
"text_prompt": build_prompt("Reconstruct", sample_recon_prompt()),
```

多样性来自 357 条模板文本，任务包装必须与预训练一致。

### ⚠️ 更隐蔽的 bug：flow 方向与采样器相反

修好包装后仍然输出噪声（300 步检查点换种子 0.37–0.52）。定位到 `flow.py` 的时间方向写反了。

`sample_discrete_euler` 从 **t=1 的纯噪声**出发积分到 t=0：

```python
x_t = (1 - t) * x1 + t * eps        # t=1 -> 噪声，t=0 -> 数据
v   = dx/dt = eps - x1
```

而我们最初写成了 `x_t = (1-t)*eps + t*x1`、`v = x1 - eps`，正好是反向参数化。
模型在“错误目标”上 loss 一路下降，用正确采样器就只会得到噪声。

证据（同一批 5 s 音频、同一条件，计算 rectified-flow loss）：

| 模型 | loss（反向目标） | loss（正确目标） |
|---|---|---|
| 原始 Audio-Omni | 3.04 | **0.98** |
| 反向训练 300 步 | 1.24 | 2.91 |
| 反向训练 2000 步 | 1.09 | 2.86 |

正确写法（`flow.py`）：

```python
x_t = (1 - t.view(shape)) * latents + t.view(shape) * noise
target = noise - latents
```

修正后训练 600 步：重建频谱平坦度 **0.026 / 0.069 / 0.073**（三个种子，原始 0.032），
训练 loss 从 1.05 稳定在 ~1.0，不再塌缩。

## 用法

```bash
cd ~/Reca-Audio
source ~/miniconda3/etc/profile.d/conda.sh && conda activate audio-omni
export QWEN_OMNI_MODEL_PATH=$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-3B/snapshots/*/ | head -1)
export PYTHONPATH=$PWD/src:$PWD/Audio-Omni:$PYTHONPATH

# 冒烟（2 步）
SMOKE=1 bash runs/scripts/train_reca.sh

# 正式：2000 次优化，累积 4 → 约 7 小时
STEPS=2000 ACCUM=4 bash runs/scripts/train_reca.sh
```

常用环境变量：`TASK_MODE=reca|t2a|mix`、`LAMBDA_T2A`、`LR`、`SECONDS_PER_CLIP`、
`OPTIM`、`VAE_DEVICE`、`MANIFEST`、`OUT`。

训练产物：`runs/train/<name>/dit_step*.pt`（fp16 DiT，约 5.7 GB）+ `train_log.jsonl`。

## 合并与评测

```bash
python -m reca_audio.train.merge_ckpt \
  --ckpt Audio-Omni/model/model.ckpt \
  --dit runs/train/reca_v1/latest.pt \
  --out Audio-Omni/model/model_reca.ckpt
```

然后用同一套评测脚本换 ckpt 即可对比：

```bash
INPUT=.../prompts1000.jsonl REF=.../ref OUT=.../AudioCaps_reca \
  GPUS_PER_PROC=2 CKPT=$PWD/Audio-Omni/model/model_reca.ckpt \
  bash runs/Benchmark/run_shards.sh --benchmark AudioCaps
```

## 已知限制

1. **训练 5 s、推理 11 s**：DiT 用 RoPE，长度可变，但训练长度与推理不一致可能影响长音频质量；
   显存允许时建议 `SECONDS_PER_CLIP=11`（需要更大显存或梯度检查点，当前 DiT 实现不支持后者）。
2. **未做 T2A 混合**：论文最优设置是 λ_RecA=1、λ_T2A=0，脚本已支持 `TASK_MODE=mix` 但未验证。
3. **没有验证集指标**：训练中只记录 loss；效果需训练后用 AudioCaps/MusicCaps 评测。
4. **merge_ckpt 尚未实跑**：逻辑按 ckpt 键结构写死（`model.model.transformer.*`），
   首次使用请检查 “已更新 N/N” 是否为满。
