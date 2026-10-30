# Audio-Omni 的 RecA 改动说明

上游：`ZeyueT/Audio-Omni` @ `e548fb0`（2026-06-10），SIGGRAPH 2026。
本文记录**每一处改动、位置、原因**，便于审阅与回滚。

---

## 0. 一句话

RecA 在这套代码里几乎不需要改结构：Audio-Omni 的 DiT 本来就是「Qwen hidden states → cross-attn → rectified flow」，
RecA 只是把**条件来源**从「文本」换成「音频自身的理解特征」，把**目标**换成输入音频本身。

所以改动集中在三处：**精度（V100 必需）**、**任务模板**、**API 入口**。

---

## 1. `audio_omni/models/conditioners.py`（V100 兼容，必须）

上游在 3 个地方硬编码 `bfloat16`，而 bf16 需要 sm80+；V100 是 sm70，会直接抛
`Current CUDA Device does not support bfloat16`，**整个模型加载不起来**。

| 位置 | 改动 |
|---|---|
| 模块级 | 新增 `resolve_omni_dtype()`：`auto` 模式下 sm80+ 用 bf16，否则退回 fp32；可用 `AUDIO_OMNI_QWEN_DTYPE` 覆盖 |
| `__init__` | 新增 `self._omni_dtype`；`self.model.to(dtype=torch.bfloat16)` → `self.model.to(dtype=self._omni_dtype)`；`torch_dtype=torch.bfloat16` → `torch_dtype=self._omni_dtype` |
| `forward` | `hidden_states[layer_idx].bfloat16()` → `.to(self._omni_dtype)` |

**为什么退回 fp32 而不是 fp16**：这里是**冻结的前向**，没有 loss scaling 保护；Qwen2.5-Omni 的激活值在 fp16 下容易溢出，
fp32 更稳。代价是 3B 权重占 ~12 GB 显存——V100 32 GB 够用。
如果显存紧张，可以 `export AUDIO_OMNI_QWEN_DTYPE=float16` 试。

---

## 2. `audio_omni/prompts.py`（新增任务模板）

新增：

```python
SYSTEM_RECONSTRUCT = "You are a professional audio engineer. Your task is to faithfully reconstruct ..."
RECONSTRUCT_USER   = "Reconstruct the input audio faithfully."
```

并在 `build_prompt()` 里加了 `"Reconstruct"` 分支。

**为什么单独一个 system prompt**：音频模型对"描述这段音频"和"重建这段音频"的响应完全不同。
如果沿用编辑任务的 system prompt，模型可能走「转写 → 朗读」捷径，恰好把要监督的声学细节全丢掉。
模板只负责**选择任务**，信息全部来自音频特征——这正是原论文 template 的作用。

---

## 3. `audio_omni/api.py`（RecA 入口）

新增 `AudioOmni.reconstruct(source_audio, ...)` 与 `_load_audio_16k_mono()`。

### 3.1 关键设计：低层条件必须静音

Audio-Omni 的编辑任务把源音频塞进 `audio_input_prompt`（100 维 mel，低层条件），
这是一条**强复制通道**。如果 RecA 也照做，DiT 直接从 mel 抄声学细节，
就变成了「mel → 音频」的声码器任务，完全绕过了理解塔——实验就失去意义。

所以 `reconstruct()` 的构造是：

| 条件 | 取值 | 作用 |
|---|---|---|
| `omni_prompt.audio_prompt` | **源音频 16 kHz 单声道** | 进 Qwen 理解塔，产出 hidden states（RecA 的 h） |
| `omni_prompt.text_prompt` | `Reconstruct ...` 模板 | 只选择任务 |
| `audio_input_prompt` | **全零波形** | 与 T2A 完全一致，不泄漏任何声学信息 |
| `sync_feature` | 全零 | 无视频 |
| `speech_prompt` | `None` | 与编辑任务一致 |

这对应原论文的 `L_RecA = L(f_θ([t_template; h]), x)`：h 是理解塔特征，目标就是 x 自己。

### 3.2 采样率

Qwen2.5-Omni 的音频塔按 16 kHz 处理，而 DiT 侧是 44.1 kHz。
所以 `_load_audio_16k_mono()` 单独准备 16 kHz 版本，避免把 44.1 kHz 直接喂给 feature extractor 造成 2.75 倍时长错位。

---

## 4. 新增脚本

| 文件 | 作用 |
|---|---|
| `runs/scripts/reca_diagnose.py` | E1 诊断：批量重建、存 wav、写 manifest（对应论文 Fig.5） |
| `runs/server/setup_audio_omni.sh` | 独立 conda 环境（torch 2.6 + transformers 5.3），HF 走镜像 |

---

## 5. 尚未改动、下一步要做的

**训练脚本还没写**。仓库只有推理代码，RecA 后训练需要新增：

1. `runs/reca/train_reca.py`
   - 冻结 Qwen 与 VAE，只训 DiT（+ 可选的 `proj_features`）
   - 目标：VAE encode 后的 latent，rectified flow 损失
   - 条件：Qwen hidden states 缓存到磁盘（关键——否则每步都要跑 3B 前向）
2. 条件缓存 `runs/reca/cache_condition.py`
   - 对每条音频只跑一次 Qwen，存 `[2048, 768]` 特征 + mask
   - 5 万条 × 2048 × 768 × 2B ≈ 150 GB（fp16）；可截断到 512 token 降到 38 GB
3. 评测脚本（重建 WER/SECS/FAD）后续再加

**为什么先不写**：训练脚本的 tensor 形状必须用一次真实推理的对数来校准
（cross-attn 是 2304 token 还是 2048、global_cond 怎么 flatten），
盲写大概率白跑一天。等 `reca_diagnose.py` 在服务器上跑通、贴回日志，再写训练就一次成型。

---

## 6. 怎么验证改动是对的

```bash
# 1) 只验证 import 与精度分支
python -c "import torch;from audio_omni.models.conditioners import resolve_omni_dtype as r;print(r('auto'))"
# V100 期望：torch.float32

# 2) 跑 3 条重建，看是否出 wav
python runs/scripts/reca_diagnose.py \
  --config Audio-Omni/model/Audio-Omni.json \
  --ckpt   Audio-Omni/model/model.ckpt \
  --input  Audio-Omni/example/example.wav \
  --out    runs/samples/reca_smoke --steps 30 --limit 1
```

**判定标准**：
- 能出声 → 精度补丁与 API 打通；
- 内容大致对（能听出是同一类声音）→ 理解塔确实在传信息；
- 音色/空间/质感明显丢失 → 这正是 RecA 要修的失配（E1 的预期结果）。

---

## 7. 已知风险

| 风险 | 说明 | 兜底 |
|---|---|---|
| `processor(audio=...)` 路径上游未充分测试 | 以前只传过全零波形（会被过滤），真实音频是第一次走 | 失败则改用 `understand()` 那套 `process_mm_info` 直接取 hidden states |
| Qwen fp32 显存 12 GB | 与 DiT/VAE 同卡可能吃紧 | 先 `CUDA_VISIBLE_DEVICES=0` 单卡；必要时换 fp16 |
| 44.1 kHz 生成 vs 10 s 输入 | `sample_size=485100` ≈ 11 s | 评测时按前 10 s 对齐 |
| `_postprocess` 做了峰值归一 | 会改变响度 | 评测重建相似度时先做响度对齐 |
