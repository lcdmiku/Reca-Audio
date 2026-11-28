"""Render the paired A0/A1 evaluation in the experiment output directory."""
import json


NAMES = {"a0": "A0 原生 Edit", "a1": "A1 低通音频与文本"}


def render_report(root, result):
    metrics = result["metrics"]
    delta = result["paired"]["a1__minus__a0"]
    checks = json.loads((root / "condition_checks.json").read_text())
    rows = [
        "# Qwen 联合音频与文本的编辑推理诊断", "",
        "两组共用 Base、原始 Mel、编辑指令和逐样本 seed，使用 100 步 RF Euler 与 CFG=7。"
        "A1 把 16→4→16 kHz 的波形和指令送入同一次 Thinker 调用。"
        "四条接口样本的源波形与 Mel 投影哈希逐对一致。"
        f"正式 A0 {'复用了逐点通过验证的历史生成' if checks['reuse_a0'] else '在本轮完整重生成'}。", "",
        "## 编辑、保留与源复制风险", "",
        "| 条件 | 编辑达标 | 保留达标 | 联合达标 | 编辑与保留 | 仅编辑 | 仅保留 | 均未达标 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, name in NAMES.items():
        stats = metrics[arm]["groups"]["native"]
        cases = metrics[arm]["native_contingency_table"]
        percentages = [f"{stats[k]['mean']:.1%} ({stats[k]['n']})"
                       for k in ("edit_success", "preserve_success", "joint_success")]
        counts = [str(cases[k]) for k in ("both", "edit_only", "preserve_only", "neither")]
        rows.append("| " + " | ".join([name, *percentages, *counts]) + " |")
    rows += [
        "", "机器达标率对 Add 和 Remove 等权平均，末四列仅统计同时具备编辑与保留判定的样本。"
        "源复制对照和目标参考校准可查看逐条分数。若 A1 的背景指标改善而仅保留计数增加、编辑响应减弱，"
        "联合指标可量化这种取舍。", "",
        "## A1 减 A0 的逐样本配对估计", "",
        "| 任务与指标 | 个数／源组数 | 差值 | 95% 配对区间 |", "|---|---:|---:|---:|",
    ]
    requested = {
        "native": ("Add／Remove", {
            "edit_success": "编辑达标（百分点）",
            "preserve_success": "保留达标（百分点）",
            "joint_success": "联合达标（百分点）",
            "preserve_embeddings": "未编辑区背景相似度",
            "preserve_sisdr_aligned": "未编辑区对齐 SI-SDR（dB）",
            "edit_region_logmel": "编辑区目标 Log-Mel 距离",
        }),
        "add": ("Add", {"edit_success": "编辑达标（百分点）", "joint_success": "联合达标（百分点）"}),
        "remove": ("Remove", {"edit_success": "编辑达标（百分点）", "joint_success": "联合达标（百分点）"}),
        "swap_instruction": ("固定源换指令", {
            "instruction_switch_success": "达标（百分点）",
            "instruction_switch_margin": "连续响应",
            "instruction_switch_progress": "按参考变化归一化的连续响应",
        }),
        "swap_source": ("固定指令换源", {"source_switch_success": "达标（百分点）",
                                         "source_switch_margin": "背景分差"}),
    }
    for task, (name, fields) in requested.items():
        for key, title in fields.items():
            stats = delta[task][key]
            if stats is None:
                continue
            scale = 100 if key.endswith("_success") else 1
            lo, hi = [value * scale for value in stats["ci95"]]
            change = stats["delta"] * scale
            rows.append(f"| {name} {title} | {stats['n']}／{stats['groups']} | {change:+.5f} | "
                        f"[{lo:+.5f}, {hi:+.5f}] |")
    joint = delta["native"]["joint_success"]
    edit = delta["native"]["edit_success"]
    if joint and edit and joint["ci95"][0] > 0 and edit["ci95"][1] >= 0:
        rows += ["", "本轮联合达标有正向配对证据。训练决策应结合换源、换指令和分任务结果。"]
    else:
        rows += ["", "本轮机器指标尚未建立联合达标收益。现有结果不足以把这项推理改动作为训练收益的依据。"]
    rows += ["", "区间以源组进行 10,000 次配对重采样。各模型仅用同一 Base，"
             "评分描述已有诊断集和机器代理。辅助指标的区间未作多重比较修正。", "",
             "[逐条结果](results.json) · [条件检查](condition_checks.json) · [推理配置](inference_protocol.json)"]
    (root / "analysis.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
