# UltraEval-Audio 接入说明

上游：https://github.com/OpenBMB/UltraEval-Audio （ACL 2026 demo）

## 为什么用稀疏检出

完整 clone 在 Windows 上 checkout 失败（`audio_evals/lib/Kimi-Audio/...` 路径过深）。
这里用 `--filter=blob:none --sparse`，只保留：

```
audio_evals/{*.py,models,dataset,evaluator,agg,prompt,process,lib/*.py,lib/text_normalization}
registry/  docs/  cli/  pyproject.toml  requirements.txt
```

`audio_evals/lib/<各模型私有实现>` 未包含——我们只接自己的模型，用不到。

## 接入了什么

| 文件 | 作用 |
|---|---|
| `audio_evals/models/audio_omni.py` | 适配器：把 Audio-Omni 包成 UltraEval 的 `Model` |
| `registry/model/audio_omni.yaml` | 注册 `--model audio-omni`，参数含模型路径与双卡设置 |

适配器把 prompt 里的 `audio` 取第一条、文本拼成问题，调用 `AudioOmni.understand()`。

## 运行

```bash
cd ~/Reca-Audio/runs/Benchmark/UltraEval-Audio
source ~/miniconda3/etc/profile.d/conda.sh && conda activate audio-omni
export HF_ENDPOINT=https://hf-mirror.com
export QWEN_OMNI_MODEL_PATH=$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-3B/snapshots/*/ | head -1)

python audio_evals/main.py --dataset sample --model audio-omni --limit 2
python audio_evals/main.py --dataset <dataset> --model audio-omni --prompt <prompt> --limit 20
```

结果写到 `res/audio-omni/<dataset>/<时间>.jsonl` 与 `*-overview.jsonl`。

可用数据集/模型：`python cli/list_availabel.py`。

## 为适配当前环境打的补丁

| 文件 | 补丁 | 原因 |
|---|---|---|
| `audio_evals/lib/coco.py` | caption 指标依赖改为惰性导入，缺失时只在调用时报错 | `pycocotools` 需要编译，理解类评测用不到 |
| `audio_evals/dataset/huggingface.py` | `load_dataset(..., trust_remote_code=True)` 失败时退回不传该参数 | 环境里是 `datasets==2.14.7`，框架要求 3.6.0；旧版会把该参数透传给 builder 报 TypeError |

环境侧另外对齐了：`jiwer==2.6.0`（4.x 删了 `compute_measures`）、`sacrebleu==1.5.1` 的 tokenizer API。

## 验证结果

```
python audio_evals/main.py --dataset sample --model audio-omni --limit 2
→ {'wer(%)': 82.76, 'fail_rate(%d)': 0.0}
```

第 0 条 CER 0%（输出「参与风筝表演的都是世界级顶尖高手。」与参考一致）；
第 1 条 CER 184%——模型在答案后追加了对话性文字，属于**提示词遵循**问题，不是框架问题。

## 已知限制

- `librispeech-test-clean` 等真实数据集需要从 HF 下载，hf-mirror 偶发超时；
- 框架的 `format` 型 prompt 要求模型严格输出 JSON，Audio-Omni 不总是遵守；
- 分词/评分口径以 UltraEval 为准，与我们在 `runs/Benchmark/MMSU` 里的官方脚本口径不同，**不要混用**。
