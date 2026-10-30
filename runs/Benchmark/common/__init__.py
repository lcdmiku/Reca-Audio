"""统一评测框架的公共组件。

目录约定：
    runs/Benchmark/<Benchmark>/evaluate.py   每个 benchmark 一个入口
    runs/Benchmark/common/                   共享的生成、指标、数据加载

设计原则：
- 生成与打分分离：`gen` 只负责产出 wav + manifest，`score` 只读盘上的 wav。
  这样换指标不用重跑几十个小时的采样。
- 所有 benchmark 共用同一套 prompt 格式（jsonl，每行含 `id` / `prompt`）。
"""
