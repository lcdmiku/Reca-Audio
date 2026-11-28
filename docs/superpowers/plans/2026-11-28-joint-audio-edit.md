# Qwen 联合音频与文本的编辑推理对照

用原始 Audio-Omni Base 同时评估 A0 原生编辑与 A1 联合条件。两组共用原生命令、未滤波 Mel、10 秒输入、100 步 RF Euler、CFG=7、相同采样窗口和 seed。A1 在用户消息中把音频占位符放在编辑指令之前，把同一段源音频转为 16 kHz，再调用现有 `_apply_bottleneck(..., downsample=4)` 做 16→4→16 kHz。此干预在波形域完成。

- 新增 `runs/Benchmark/AudioEdit/evaluate_joint_edit.py` 和启动入口。在原生编辑调用的 conditioner 边界注入 A1，复用 `_wrap_audio` 和现有采样器。把现有评分器的组名和配对参数提取为可选参数，保留原六模型 CLI。
- 选择 Add、对应 Remove、固定源换指令及固定指令换源各一条。用真实 Thinker pre-hook 验证同一输入含有效音频 token、音频特征和完整指令。逐样本比对源波形、Mel 投影、CFG、采样步数和 WAV 帧数。小规模验证使用与正式评测相同的 100 步。
- 若四条 A0 与已核验的历史 Base WAV SHA256 完全一致，正式组直接引用该目录。否则在独立目录重新生成全部 A0。沿用锁定的 120 条清单，完成两组后才评分。保留逐样本结果、源组配对 bootstrap 和已知源复制对照。
- GPU 1、2 各用一张 V100，推理和评分顺序执行。预留 2 GiB 持久空间，正常复用 A0 时预估运行约 55 分钟，重生成 A0 时约 100 分钟。输出独立放在 `runs/Benchmark/output/AudioEdit_joint_qwen_v1/`。
