"""Render the four-arm audit from the completed evaluation's results.json."""

from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys


LABELS = {"base": "Base", "pure": "Pure ReCA", "t2a": "T2A-only", "mix": "Mix"}


def table(headers, rows):
    return "\n".join("| " + " | ".join(map(str, row)) + " |"
                     for row in [headers, ["---"] * len(headers), *rows])


def metrics_rows(settings, results):
    for setting in settings:
        for arm, entry in results[setting].items():
            s = entry["summary"]
            primary = s["confirmatory"]
            lo, hi = primary["delta_ci95"]
            yield (setting, LABELS[arm], primary["n"], f"{primary['mean']:.5f}",
                   f"{primary['median']:.5f}", f"{primary['std']:.5f}",
                   f"{primary['win']}/{primary['tie']}/{primary['loss']}",
                   f"{primary['mean_delta']:+.5f} [{lo:+.5f}, {hi:+.5f}]",
                   f"{s['all']['mean']:.5f}", f"{s['shuffled']:.5f}",
                   *(f"{s[k]:.5f}" if k in s else "—" for k in ("fad_clap", "kl", "is", "fd", "lsd")))


def write_report(root, filename, title, content):
    (root / filename).write_text(f"# {title}\n\n{content}\n", encoding="utf-8")


def main(root):
    result = json.loads((root / "results.json").read_text(encoding="utf-8"))
    generated = result["generation"]
    columns = ["设置", "模型", "主要 n", "CLAP", "Median", "Std", "胜/平/负",
               "配对 Δ Base [95% CI]", "全量 CLAP", "错配 CLAP", "CLAP-FAD", "KL", "IS", "FD", "LSD"]
    note = ("AudioCaps 主要统计排除 8 个探索 SHA。全量分布指标与错配 CLAP 使用同一设置的"
            "全部样本。置信区间按 SHA 配对 bootstrap，平局阈值为 1e-6。"
            "逐条分数、seed 和 WAV 路径见 results.json。")
    for filename, title, settings in [
        ("t2a_evaluation.md", "四组 T2A 与 MusicCaps 生成评测",
         ("audio_10_cfg6", "audio_11_cfg6", "music_10_cfg6")),
        ("reca_reconstruction.md", "四组 ReCA 重建",
         ("reca_matched", "reca_mismatched")),
        ("cfg_sweep.md", "四组 AudioCaps CFG sweep",
         ("audio_10_cfg1", "audio_10_cfg2", "audio_10_cfg4", "audio_10_cfg6")),
    ]:
        content = note + "\n\n" + table(columns, list(metrics_rows(settings, generated)))
        if filename == "cfg_sweep.md":
            geo = defaultdict(list)
            for row in result["geometry"]:
                geo[row["condition"], row["arm"]].append(row)
            rows = []
            for (kind, arm), entries in geo.items():
                mean = lambda key: statistics.mean(row["drift"][key] for row in entries)
                rows.append((kind, LABELS[arm], len(entries),
                             *(f"{mean(k):.5f}" for k in ("cond", "uncond", "cfg1", "cfg2", "cfg4", "cfg6")),
                             *(f"{statistics.mean(r[k] for r in entries):.5f}"
                               for k in ("guidance_cosine", "guidance_norm_ratio"))))
            content += "\n\n" + table(["条件", "模型", "记录数", "cond drift", "uncond drift",
                                        "CFG1 drift", "CFG2 drift", "CFG4 drift", "CFG6 drift",
                                        "guidance cosine", "guidance norm ratio"], rows)
        else:
            key = ("duration_11_minus_10" if filename == "t2a_evaluation.md"
                   else "reca_matched_minus_mismatched")
            content += "\n\n" + table(["模型", key, "95% CI", "胜/平/负"], [
                (LABELS[arm], f"{stats['mean_delta']:+.5f}",
                 " / ".join(f"{x:+.5f}" for x in stats["delta_ci95"]),
                 f"{stats['win']}/{stats['tie']}/{stats['loss']}")
                for arm, stats in result["paired_contrasts"][key].items()])
        write_report(root, filename, title, content)
    drift = result["parameter_drift"]
    rows = [(name, *(f"{drift[a][name]['relative_l2']:.8f}"
                      if drift[a][name]["relative_l2"] is not None else "—" for a in LABELS))
            for name in drift["base"]]
    write_report(root, "parameter_drift.md", "DiT 参数相对 L2",
                 "使用 checkpoint 原始精度，按参数平方和计算相对 L2。全零 Base 范数的项保留绝对 L2，见 results.json。"
                 "\n\n" + table(["模块", *LABELS.values()], rows))
    best = []
    for arm in LABELS:
        scores = {cfg: generated[f"audio_10_cfg{cfg}"][arm]["summary"]["confirmatory"]["mean"]
                  for cfg in (1, 2, 4, 6)}
        best.append((LABELS[arm], *(f"{value:.5f}" for value in scores.values()),
                     max(scores, key=scores.get)))
    write_report(root, "analysis.md", "四组统一协议的结果摘要",
                 table(["模型", "CFG 1", "CFG 2", "CFG 4", "CFG 6", "实测最高 CFG"], best)
                 + "\n\n实测最高值限于四个预设 CFG 点。配对置信区间见 cfg_sweep.md。"
                 "10/11 秒差值及 MusicCaps 域结果见 t2a_evaluation.md。"
                 "重建对照及错配干预见 reca_reconstruction.md。")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
