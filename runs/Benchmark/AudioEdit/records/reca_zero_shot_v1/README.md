# AudioEdit zero-shot transfer, v1

正式清单锁定为 120 条。SHA256 为 `5e1760d86105bbf31f2934beace88ea045199f9c1445c7c7686c47928882672a`。其中 64 条来自 32 组互为逆操作的 Add／Remove，16 条属性编辑，16 条音乐风格编辑，12 条固定源换指令，12 条固定指令换源。全部来自过去归档的 96 条不同 held-out 源音频，来源 ID、YouTube ID 和 SHA 对本项目的后训练集均无交集。前景由 caption 筛选，再按 3 秒 RMS 窗口截取。事件阈值应结合目标与源复制校准后的有效数量解释。此清单不含人工的逐片段分离质量标签。

全部六组共 30 条 smoke 音频通过原生编辑推理与 WAV 完整性检查，CLAP 和 PANNs 评分均为有限值。Base 上固定源的换指令只改变 Qwen 条件，固定指令的换源只改变 Mel 条件。

正式任务使用 GPU 1、2，顺序运行六个 checkpoint，100 步 Rectified Flow Euler，CFG=7，提交 `d7b4534`。服务器运行根为 `/home/linchudeng/Reca-Audio/runs/Benchmark/output/AudioEdit_reca_zero_shot_v1/`。该根下的 `formal/` 保存原始音频、日志和评分。公开盲听题本单独放在 `listening/form/`，模型标签保存在题本目录之外。

720 条输出于北京时间 2026-11-28 15:15 完成评分。[研究结论](analysis.md) · [机器指标](machine_summary.md) · [逐条分数与区间](results.json) · [推理参数](inference_protocol.json)

[冻结清单](manifest.jsonl) · [数据审计](dataset_audit.json) · [Smoke 评分验收](smoke_scores.json) · [条件干预检查](condition_checks.json) · [评测协议](../../../audio_edit_transfer_protocol.md)
