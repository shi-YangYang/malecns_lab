"""Recomputable descriptive differences, kept separate from protocol completion."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .observability_protocol import CONDITIONS, MODES
from ..learning.resource_guard import write_json

PAIRS = tuple((condition, "blank") for condition in CONDITIONS[1:]) + (("left", "right"), ("up", "down"), ("center", "large"))
FIELDS = ("dn_spikes", "dn_v_mv", "dn_g_mv")


def summarize_arrays(branches: dict, protocol: dict) -> tuple[dict, dict]:
    """Pure numeric report helper; it does not certify graph roles or a run."""
    metrics, differences = {}, {}
    for left, right in PAIRS:
        pair = f"{left}_minus_{right}"
        metrics[pair] = {}
        for name in FIELDS:
            a, b = branches[left][name], branches[right][name]
            delta = a - b
            differences[f"{pair}__{name}"] = delta
            changed = delta != 0 if name == "dn_spikes" else ~np.isclose(
                a, b, atol=protocol["difference"]["atol_mv"], rtol=protocol["difference"]["rtol"])
            metrics[pair][name] = {"increased_cells": int(np.count_nonzero(np.any(delta > 0, axis=0))),
                                  "decreased_cells": int(np.count_nonzero(np.any(delta < 0, axis=0))),
                                  "raw_changed_cells": int(np.count_nonzero(np.any(delta != 0, axis=0))),
                                  "above_tolerance_cells": int(np.count_nonzero(np.any(changed, axis=0))),
                                  "max_absolute_difference": float(np.abs(delta).max(initial=0))}
    inputs = {condition: {"input_event_count": int(branches[condition]["input_events"].sum()),
                          "input_spike_count": int(branches[condition]["input_spikes"].sum()),
                          "input_events_by_phase": branches[condition]["input_phase_events"].sum(axis=1).tolist(),
                          "lamina_spikes_by_type": branches[condition]["lamina_spikes"].sum(axis=0).tolist()}
              for condition in CONDITIONS}
    observation = {field: any(metrics[f"{condition}_minus_blank"][field]["above_tolerance_cells"] > 0
                               for condition in CONDITIONS[1:]) for field in FIELDS}
    return {"pairs": metrics, "inputs": inputs, "response_observation": observation}, differences


def summaries_from_files(directory: Path, protocol: dict) -> tuple[dict, dict]:
    summary = {"schema": "cnsbridge.observability_summary.v1", "modes": {}}
    differences = {}
    for mode in MODES:
        branches = {}
        for condition in CONDITIONS:
            with np.load(directory / mode / f"{condition}.npz", allow_pickle=False) as data:
                branches[condition] = {name: data[name] for name in
                                       (*FIELDS, "input_events", "input_spikes", "input_phase_events", "lamina_spikes")}
        summary["modes"][mode], diff = summarize_arrays(branches, protocol)
        differences.update({f"{mode}__{name}": value for name, value in diff.items()})
    return summary, differences


def report_text(summary: dict, protocol: dict) -> str:
    lines = ["# 工程输入可观测性记录", "",
             "本文是原始采集的描述性汇总；协议完整性以父监督和独立 validate-run 结果为准。", "",
             f"时间窗（ms）：{protocol['windows_ms']}。预处理不表示已到稳态。",
             "电位与突触状态按每步返回的复位后状态，以 float64 累加后取 1 ms 桶均值。",
             "计数差为整数精确比较；v/g 容差为 atol=1e-5 mV、rtol=1e-6，仅是数值阈值。", ""]
    for mode in MODES:
        lines += [f"## {mode}", "", "| 比较 | 发放差异细胞 | 电位差异细胞 | g 差异细胞 |", "|---|---:|---:|---:|"]
        for pair, values in summary["modes"][mode]["pairs"].items():
            lines.append(f"| {pair} | " + " | ".join(str(values[field]["above_tolerance_cells"]) for field in FIELDS) + " |")
        lines += ["", "输入事件与 L1–L5 发放总量：", ""]
        for condition, values in summary["modes"][mode]["inputs"].items():
            lines.append(f"- {condition}: 输入事件 {values['input_event_count']}；L1–L5 {values['lamina_spikes_by_type']}")
        lines += [""]
    lines += ["## 解释边界", "",
              "直接输入工程 12×16 掩码，绕过相机和目标识别；映射不是实测视网膜定位。",
              "统一 LIF 与固定 8 mV 早期视觉背景均为模型假设。背景活动本身不证明视觉传播。",
              "零差异只表示本刺激、动力学与有限窗口内未观察到差异，不能推出真实果蝇无响应。",
              "这里没有优化参数、动作读出或设备控制，也未验证生理还原、拓扑优势或飞行能力。",
              "原始 differences.npz 保留所有 DN/时间桶的有符号差异，summary.json 保留增加、减少、原始与阈值计数。", ""]
    return "\n".join(lines)


def write_summaries(directory: Path, protocol: dict) -> None:
    summary, differences = summaries_from_files(directory, protocol)
    write_json(directory / "summary.json", summary)
    np.savez(directory / "differences.npz", **differences)
    (directory / "report.md").write_text(report_text(summary, protocol))
