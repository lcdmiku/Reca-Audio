# ReCA 对 Audio-Omni 编辑能力的零样本迁移

## 代码审查

Audio-Omni 论文第 3 节定义了 Add、Remove、Extract 和 Style Transfer，报告约 110 万个训练样本和 2,000 个测试样本。第 5.2 节和表 4 报告编辑 LSD、FAD 以及 Add、Extract、Style Transfer 的 CLAP。Base 已接受编辑监督，本实验的“零样本”指后训练阶段没有编辑专项监督。上游仓库公开了四类任务的 API、Gradio 与示例。当前 checkout 保留四条 44.1 kHz、10 秒的 demo MP3 和数据导出工具。Benchmark README 中的 `AudioEdit/evaluate.py` 在当前 checkout 和服务器都没有对应文件。Hugging Face 公开数据页面只显示 train，预览为归档音频结构，目前未核实能直接获取论文的带指令、有目标且有来源元数据的 test split。[论文](https://arxiv.org/html/2604.10708v1) · [官方仓库](https://github.com/ZeyueT/Audio-Omni#-audio-editing) · [官方 AudioEdit 数据页](https://huggingface.co/datasets/HKUSTAudio/AudioEdit)

| 输入或中间量 | 原生编辑通路 |
|---|---|
| 指令 | `AudioOmni.edit` 用 `build_prompt` 包装对应操作、声音描述或源／目标类别。Qwen Thinker 从文本得到 hidden states，经冻结投影进入 DiT 高层 Cross-Attention。编辑 metadata 的 `omni_prompt.audio_prompt` 为 `None`。 |
| 源音频 | API 读取音频、转单声道并对齐到 44.1 kHz，最多取前 10 秒。波形进入 `audio_input_prompt`，`TASK_TTS=False`。Mel 特征经序列与通道投影后与零 `sync_feature` 拼成 global condition，由 DiT 的 `to_global_embed` 映射，与 timestep embedding 结合后作为前缀。 |
| VAE latent | 原生编辑让 Rectified Flow 从随机 latent 开始采样，原音频通过 Mel 条件起作用。接口没有传 `init_audio` 或编辑 mask。VAE 解码生成 latent 得到输出波形。 |
| CFG | 正向与无文本分支复用相同的低层 global condition。官方 Python 编辑 API 默认 CFG=7，论文通用推理设置和 Gradio 编辑页默认 CFG=6。 |
| ReCA 重建的区别 | `reconstruct` 把源波形送入 Qwen 音频路，在 `audio_input_prompt` 传静音，用重建模板和 CFG=1。这是另一条前向通路。Pool4 的训练干预针对 Qwen 音频 token，原生 Edit 没有音频 token，所以 Pool4 与 Mix 在 Edit 中使用相同的推理接口。 |

当前 API 可直接调用四个原生任务。评测需要补独立 runner 和数据、评分入口，并处理两个实际接口约束。第一，Mel 的时间投影固定输入为 1,723 帧，而 Python loader 只截断、不把短音频补到 10 秒。runner 应统一将输入规范化为 441000 个 44.1 kHz 单声道样本，记录实际截断或补零及所有有效区域。第二，API 后处理会按输出峰值归一化并转 int16，绝对增益编辑不宜用该信号打分。自由形式属性指令沿用 Gradio 的 `SYSTEM_EDITING` 文本包装，保留相同的源条件，另存解码后的 float 波形用于评分。Add／Remove／Style 可由 runner 直接委托原 API，扩展原始输出时应先核对包装后的 metadata 和 native API 一致。[当前编辑 API](../../Audio-Omni/audio_omni/api.py#L145) · [编辑模板](../../Audio-Omni/audio_omni/prompts.py#L97) · [Mel 支路](../../Audio-Omni/audio_omni/models/conditioners.py#L801) · [条件路由](../../Audio-Omni/audio_omni/models/diffusion.py#L142) · [CFG](../../Audio-Omni/audio_omni/models/dit.py#L284) · [官方 API](https://github.com/ZeyueT/Audio-Omni/blob/main/audio_omni/api.py)

## 模型与严格对照

服务器上的 Base 全模型和 5 份后训练终点均经 CPU mmap 核查。后训练终点的 554 个 DiT 张量名称和形状与 Base 完全一致，保存的 Adafactor 状态均为 1,768 次实际更新。下表完整文件 SHA256 来自既有评测归档，正式 runner 在加载前重算摘要、核对结构并严格加载。每次只在内存替换 DiT，Base 的 Qwen、投影、Mel 条件塔和 VAE 由所有模型共用。

| 模型 | ReCA／文本有效曝光 | 更新 | 训练池化 | SHA256 |
|---|---:|---:|---|---|
| Base | 不适用 | 不适用 | 无 | `8ddfdd6c965b7b1fc5c5af01e306cdfd78e44f56fd584789b8987233438e65b3` |
| Pure ReCA | 7072／0 | 1768 | 无 | `e2047677c252ffe7d9f7d45349171d904388dda1cd2eeaaaaede6d6aa2d8890b` |
| T2A-only | 0／7072 | 1768 | 无 | `35283133c171a2f39b9aef4acb9e8df6c57a5d0b1e4203902b899f538597e0c2` |
| Mix | 3536／3536 | 1768 | 无 | `88901b9b6f52d8f9f6061782f7b0a196c12d8207c03fbc728911d7b0ac08587e` |
| Pool4 | 3536／3536 | 1768 | 仅 ReCA 条件，宽度 4 | `208806160cde769e87aa9dcdc269ac59939639c47f5ac3445a713f6aa3e7c241` |
| 仅文本槽 | 0／3536 | 1768 | 无 | `ee905bd85d4ec5dbe1078105bfba342d6917d62c442423d694d3c0a368972e6b` |

各后训练从同一 Base 开始，使用 7,070 条 AudioCaps 独立训练音频构成的数据池、10 秒目标、累积 4、Adafactor 和峰值 LR 5e-6。Pure／T2A-only／Mix 来自 `audiocaps_low_lr_v1`，训练提交 `deb1e5d`。仅文本槽来自 `audiocaps_mix_text_slots_v1`，提交 `28d590a`。Pool4 来自 `audiocaps_mix_pool4_v1`，提交 `86d1468`。路径和完整训练合同见既有结果。Pure − Base 估计纯 ReCA 后训练相对原模型的变化。Mix − 全量 T2A 固定总训练预算，改变两类监督剂量。Mix − 仅文本槽固定 caption、槽位和文本 loss 权重，是添加 ReCA 监督的首选对照。Pool4 − Mix 固定训练调度，只改变训练时的音频 token 池化。[checkpoint 与训练记录](../../docs/audits/2026-11-25-mix-pool4-results/results.json) · [同 caption 预算](../../docs/audits/2026-11-22-caption-budget-results/analysis.md)

## 预定小规模评测

以 96 个编辑实例为目标。先按固定种子从已归档的 AudioCaps 与 MusicCaps 候选池排序，再由人工只看输入、前景和参考，审核可辨识性、声源与时间边界。筛选完成后冻结清单及 SHA256，再生成任何模型输出。同一源族的反向操作和反事实始终关联到同一统计单元。下列数字是计划配额。若素材无法达到某类配额，将该类降为定性扩展，并记录筛选失败情形。

| 范围 | 数量 | 任务和参考 |
|---|---:|---|
| 原生编辑，预定主要集 | 64 | 32 个独立背景与前景组，每组构造 Add 和 Remove。前景为一段人工核实的短事件，以固定位置和增益混合。Add 输入为背景、目标为混合；Remove 输入为同一混合、目标为干净背景。编辑区间记录实际前景支持，未编辑区间保留背景并在边界留固定余量。原生指令指定声源。Add 的目标时间未写入默认指令时，定位作为辅助指标。Remove 评估是否从实际出现的区间移除了事件。 |
| 属性扩展 | 16 | 16 条独立环境音，8 条固定低频 EQ，8 条固定短混响。用确定性滤波或脉冲响应生成目标，指令通过官方通用编辑包装。记录滤波参数与有效片段，以频带能量变化或干湿／衰减指标评分。 |
| 语义与音乐扩展 | 16 | 16 条独立 MusicCaps 片段，选择主乐器可辨识、节奏或旋律连贯的样本，用原生 Style Transfer 指定源与目标乐器。采用目标音色、原旋律／节奏保留的盲评。仅在可可靠提取时报告旋律轮廓或 onset 相似度。 |

候选记录已做初查。AudioCaps 的 81 个非探索 SHA 与 MusicCaps 的 89 个候选 SHA，均未命中官方 post-training manifest 中已有的 SHA；历史 AudioCaps 审计的 ID、YouTube ID、SHA 交集均为 0。正式固化时对源录音、背景和事件原始素材重算文件 SHA，统一比较原始标识（含 YouTube ID 与时间戳）、训练音频文件 SHA 及可用的规范化 PCM SHA，记录训练与候选 train／val／test 来源和交集。混音或重新编码后的新 SHA 也纳入清单，原素材身份优先于衍生文件名。重叠则先剔除再定稿。由此建立的是对本项目后训练的 held-out，Base 原始训练数据的完整成员关系仍不可审计。

每条记录包含稳定 ID、source group、输入源和前景 stem 的 ID／路径／SHA、目标（若存在）、原生 task 参数、实际渲染指令、结果语义描述、编辑／保留区间、生成 seed、数据出处、规范化和合成参数。人工先标定混合物中的前景类别、强度与可辨识性，并保留一条不编辑的源复制参照和确定性参考目标。

从主要集中预选 12 个 Add 源组，每组再生成两条干预样本：固定源和 seed 换成另一个明确可分的事件指令，固定原指令和 seed 换用另一个已验证的源背景。前者计算目标事件的切换幅度，后者检查输出背景随实际输入切换，并与错误背景的保持得分比较。干预样本配套参考与独立标识，共 24 条，另行报告。原生任务真实编辑的判定需要同时看到指令带来的特定事件变化和背景保留。

## 统一推理与验收

- 顺序使用一对空闲 V100，所有模型共用 Base config、冻结组件、相同 seed、相同 prompt 顺序、相同双卡精度策略和后处理。100 步 Rectified Flow Euler，所有模型统一使用官方 Python Edit 默认 CFG=7。CFG=6 是论文和 Gradio 的固定备选，只有 API smoke 暴露整套协议的技术问题时才统一修订并重新记录。沿用模型配置的 485100 点生成窗口，即 236 个 latent 位置，按架构预计输出 483328 点。保存实际帧数。对齐参考的主指标只用前 441000 点，额外尾段和异常单独统计。原始解码波形与盲听呈现采用统一的声音处理规则。
- 最小 smoke 使用官方 Add、Remove、Extract、Style Transfer 示例，每个模型至少各一次，可加一条短源和属性包装单元检查。确认 10 秒 Mel 输入、真实非零声学条件、指令投影、同 seed 下两种输入干预能改变对应条件张量、正确加载、正常 WAV、日志及评分全链路。demo 输出仅用于验收。若出现整体源复制、指令张量缺失或重叠审计失败，则先查明原因。
- 正式集每组 96 条外加 24 条干预，六组共 720 条。以此前 V100 同模型 100 步的吞吐粗估顺序运行约 4–7 小时，按 smoke 实测修订。服务器用户目录约 389 GiB，建议给运行预留 4 GiB，波形、控制样本及元数据预计低于此规模。用原始 checkpoint 逐组载入，可省去临时合并权重。评分阶段要求每组 manifest 覆盖全部样本、每条语音及 metadata 可读、有限且时长正确、明确列出失败；失败组单独报告并在同参数重试。
- 脚本拟放在 `runs/Benchmark/AudioEdit/evaluate_transfer.py`，独立产物目录为 `runs/Benchmark/output/AudioEdit_reca_zero_shot_v1/`。同目录保存版本、组件 SHA 和配置、清单、每组 WAV、日志、逐样本分数和汇总。

## 评分与结论规则

| 维度 | 主测量 |
|---|---|
| Edit success | Add／Remove 用按时间窗的 PANNs 事件分数（限经审核可映射的类别）、前景保留／抑制量、编辑区 Log-Mel 与配对目标的距离。Add 使用期望结果语义的 CLAP；Remove 用前景下降及背景仍在作为判据。属性任务用目标频带能量比或混响衰减偏差。盲评二元标签分别记录完成编辑、保留原内容与正确对象，报告各自率。 |
| Content preservation | 对预先标注的非编辑区报告 SI-SDR、Log-Mel distance 与源音频嵌入相似度，记录窗口误差与时间偏移。固定允许的全局微小对齐范围，并同时保留未对齐对比。整体 EQ／混响与风格编辑使用结构或目标参考指标；绝对音量与变速不使用源波形 SI-SDR 作主判据。 |
| Audio quality | 可匹配参考的任务计算 CLAP-FAD、PANNs FD／KL／IS，注明编码器与样本量。自由风格任务使用盲听为主。按组打乱输出身份，同一音频的评者可听源并看到真实指令，单独评分自然度、伪影、编辑成功与源内容保留。优先收集至少三名评者，未收齐时将听评标为待验证。 |
| Instruction following | 记录指定对象、变化方向、时间窗（指令明确或 Remove 的已知事件范围）和意外变化。对换指令与换输入分别报告效果。对没有唯一参考的样本，报告盲听及适用的结构测量。 |

主要结果为 Add／Remove 的编辑成功与非编辑区保留，分别报告，任务等权汇总并同时给出分任务结果。固定源的指令干预与固定指令的换源干预分别量化响应。采用按源组的 10,000 次配对 bootstrap，报告相对 Base 和预定对照的差值及 95% 区间。干预和分布指标使用各自的成组统计，附单一训练 seed、约 32 个原生源组、辅助指标多重比较的限制。若源保留提高且指令响应持平，只支持利用输入音频的改善。若编辑与指令成功提高、保留能力同方向且质量指标无明显恶化，才支持这组任务上的零样本编辑收益。模糊结果保持为探索性，保留区间和失败例。
