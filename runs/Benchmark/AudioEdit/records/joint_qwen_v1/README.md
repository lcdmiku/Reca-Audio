# 联合 Qwen 音频条件与原生 Edit 的同权重对照

正式评分于北京时间 2026-11-28 20:22:44 完成。两组各使用同一份 120 条清单，Base 权重、CFG=7、100 步 RF Euler，使用 GPU 1、2。A1 使用现有 ReCA 的 4× 波形低通和原始 Mel 条件。生成提交为 `207006f`，评分提交为 `beb2e88`。逐条评分 SHA256 为 `fe6c8b347a2ce5b2b7bd3423cfe28106c2631d5768eb8181a294b41f4362c7c8`。

四条接口样本重跑 A0 并逐点核对解码后的浮点样本，经验证后复用了历史 A0。A1 全部 120 条各有 250 个音频 token 和 1,000 帧有效音频特征，且实际 Thinker 输入保留完整指令。四对接口样本的原始 Mel 波形和投影哈希逐对一致，固定源换指令的样本也具有相同 Mel 投影哈希。

独立核验两组 240 条音频的 seed、SHA256 和 WAV 头，并从逐样本记录复算全部按源组配对的 bootstrap 区间。编辑及内容保留均为上一轮的机器指标。编辑区复制倾向另外使用对源、对目标的 Log-Mel 距离差，并统计编辑未达标且更接近源的样本。

服务器输出在 `/home/linchudeng/Reca-Audio/runs/Benchmark/output/AudioEdit_joint_qwen_v1/`。其中 `audio/a0` 引用历史 Base，`audio/a1` 保存本轮生成。首次复用校验的产物位于同级的 `AudioEdit_joint_qwen_v1_wav_header_check/`。

[对照报告](analysis.md) · [逐条评分](results.json) · [真实条件检查](condition_checks.json) · [推理配置](inference_protocol.json) · [固定清单](../reca_zero_shot_v1/manifest.jsonl)
