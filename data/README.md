# data

训练数据占位目录（**内容不进 git**）。

服务器上建议用 `DATA_ROOT` 指到独立磁盘，而不是塞进仓库：

```bash
export DATA_ROOT=/home/linchudeng/data/reca-audio
```

约定：

```
$DATA_ROOT/
├── raw/          # 原始下载（FSD50K / FMA）
├── processed/    # 24 kHz 单声道 PCM16 + index.jsonl
├── features/     # 编码器预计算特征
├── eval/         # 固定评测集与提示词
└── hf_cache/     # HuggingFace 缓存（HF_ENDPOINT 指向镜像）
```
