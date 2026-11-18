# 低 LR 文本生成：音频感知质量盲评

问题：固定文本、生成噪声和播放响度时，Base、Pure ReCA、T2A-only、Mix 的生成音频在自然度和可闻缺陷上有无差异？本次只评价声音质量，隐藏 caption，不收集文本相关度评分。仅覆盖 CFG=6 的低 LR 实验与 Base，不检验高低 LR 间的听感变化。

## 听评者操作

1. 每人只做一份题本。组织者交替分配 A/B，两份音频相同、顺序相反。解压收到的 ZIP（包括 audio 文件夹），用 Chrome 或 Edge 打开 index.html，输入自己独有的匿名编号，例如 5 个随机字母或数字；不必注册。
2. 使用相同耳机，在安静处选择舒适音量后固定。每组含同一条件的 4 条音频。听完一条后给它 1–5 分，可打同分；可勾选失真/爆音、断裂/跳变、异常重复、噪声/模糊、不自然的音色。声音事件本身可能是噪声或重复，只有听起来是生成缺陷时才扣分。
3. 分数锚点：1 缺陷严重；2 明显干扰；3 有缺陷但可接受；4 仅轻微缺陷；5 自然连贯，无明显缺陷。勿按题材喜好打分。16 组共 64 条，净播放约 10 分 39 秒，另需判断与操作时间。可分段完成；重新打开原题本并输入相同编号可续评，必要时先导出进度。
4. 完成后点击“导出完整评分”，将 ratings-*.json 交给组织者。部分进度文件不进入分析。参与者只需要收到一个 ZIP，不应收到来源记录或解盲 key。

## 刺激与质控

- 源为 [本轮机器评测结果](results.json)。AudioCaps 和 MusicCaps 的主分析 SHA 各选 8 个：取 SHA256(“quality-pilot-v1:” + 域名 + “:” + 音频 SHA) 最小的 8 项，不用 CLAP、FD、重建分数或播放试听挑样。四模型逐条匹配 prompt、seed、源 SHA 和时长；MusicCaps 沿用原 T2M 模板。
- 题目和音频名匿名化；每域每个模型在四种播放位置各出现两次。B 颠倒 A 的组序及组内顺序，减少位置和疲劳影响。顺序只生成一次，已发出的题本配其专用 key，切勿用重新制作的包替换。
- 用 FFmpeg 测量整条音频的综合响度与真峰值，仅施加常量增益，不压缩、限幅、裁剪或降噪。响度与真峰值的测量背景见 [ITU-R BS.1770](https://www.itu.int/rec/R-REC-BS.1770/en)。64 条输出实测 −23.13 至 −22.87 LUFS，最高 −1.91 dBTP。PCM16、声道数、44.1 kHz、440320 帧与各自源一致。本试验评估响度匹配后的质量，不评估原文件间的音量差。
- 已检查两份 ZIP 各含页面、盲表与 64 条 WAV，不含解盲文件；本地浏览器完整播放实际 WAV 后可评分，未完成的题本不能导出完整评分。技术检查未产生听评结果。

## 回收与分析

本地运行根为 `runs/outputs/low_lr_quality_pilot_v1/`，含直接打开的 `form-A/index.html`、`form-B/index.html`、两份可分发 ZIP、仅组织者保管的 `private/key-{A,B}.json` 和 `private/sources.json`。听评结果不自动上传。ZIP A 的 SHA256 为 `0415bc08d1491ce82e057d7259f59260f2ae78d66d9bc4ac035989c3c81adf82`，ZIP B 为 `e43a2d796876d19e2c33d3f33482249ccb9ded6e74ca6475d09417dfb3aab388`。

在仓库根目录，运行下列命令；末尾用实际回收的完整评分文件路径替代示例。为保护盲法，等该轮收集截止再解盲。首轮可先收 6 位独立听评者（A/B 各 3 位）；这是小样本探索，不是正式功效保证。

```text
python runs/Benchmark/listening/summarize.py --key runs/outputs/low_lr_quality_pilot_v1/private/key-A.json --key runs/outputs/low_lr_quality_pilot_v1/private/key-B.json --out runs/outputs/low_lr_quality_pilot_v1/ratings_summary.json ratings-1.json ratings-2.json
```

每人每份题本只接受一张完整答卷。每域先在同音频 SHA 和模型内合并听评者，再等权平均各 SHA，报告 1–5 平均质量、缺陷标签计数和模型配对差。优先比较 Pure−Base、Pure−T2A-only；Mix 为联合训练对照。95% 区间按 SHA 做 10,000 次 bootstrap，仅反映所选音频在这些听者条件下的变异；每域仅 8 条，不代表对所有听者/音频的精确估计。没有真人评分时只输出 awaiting_ratings。
