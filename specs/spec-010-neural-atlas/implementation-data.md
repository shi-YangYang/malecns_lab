# 数据导出实施报告

角色：Implementation。对应 Spec010，验证范围 S2；本报告不替代独立验收。

## 完成范围

- 新增 `src/malecns_lab/visualization/export.py` 和包入口，命令为 `python -m malecns_lab.visualization.export`。默认路径使用现有 `project_asset_path` 定位源码根，不依赖调用 cwd；支持 `--graph`、`--annotations`、`--recordings`、`--out`。
- 参考 NPZ 仅访问 `body_ids` 与 `meta`；官方 Feather 仅投影 `bodyId` 与 `somaLocation`。全部有效坐标统一居中和等比缩放，变换、原始边界和单位限制写入 `coordinate_transform`；不推断缺位置、神经形态或突触连线。
- 校验参考元数据身份、模型来源摘要、图身份/节点/边数量一致、注释文件 SHA256、22 群组的来源映射、每段累计数组的完整节点顺序、帧顺序、原输入时标、模拟窗口、有效性与覆盖统计、有限数值及群组率与累计成员统计的一致性。错误发生在输出写入之前；输出位置不能落入来源目录，文件采用临时文件原子替换，catalog 最后写入。
- 根据原 seed、刺激参数与原输入帧序号调用现有 `GeneratedSourceV2` 重建六种刺激，使用标准库 PNG 编码，不增加依赖。PNG 为按帧竖排的 96 × 72 单帧条带，各段共 50 帧。
- 新增 `tests/test_atlas_export.py`，覆盖精确大整数 ID、坐标缺失、无边数组访问、来源文件不变、PNG 像素重建、输入时间与模拟窗口、错误证据拒绝、零帧/空群组/全缺坐标，以及来源目录写保护。

## 静态输出与兼容字段

目录为 `apps/atlas/public/data/`，目录忽略规则由协调者负责。输出 `catalog.json`、`atlas.json`、`runs/<segment_id>.json` 和 `stimuli/<segment_id>.png`，schema 为 `malecns_lab.atlas.v1`。

catalog 的 atlas/run URL 相对 catalog 目录；run 的 preview URL 为 `../stimuli/<segment_id>.png`，相对 run JSON 目录。没有 `/data` 绝对 URL 硬编码。

已向协调者说明以下向后兼容字段：run 增加 `frame_index`、`simulation_start_s`、`simulation_end_s`、`prediction_valid`、`initialization`、`invalid_reason`、`status` 和 `coverage`。`time_s` 保留原输入帧序号 / 20，真实记录范围 0–2.45 s；对应模拟窗口为 0–2.5 s。无预览或零帧时 `stimulus_preview` 为 null，不制造替代画面。

## 实际验证

在 `/Users/yang/project/malecns_lab` 执行：

1. `.venv/bin/python -m pytest tests/test_atlas_export.py -q`：**12 passed in 0.40s**。
2. `.venv/bin/python -m malecns_lab.visualization.export`：退出码 0，导出成功，命令耗时约 0.57 s。

实际导出摘要：

| 字段 | 数量 |
|---|---:|
| 参考节点 | 166,700 |
| 有胞体坐标 | 139,662 |
| 缺胞体坐标 | 27,038 |
| 参考连接数量（来源元数据） | 25,582,938 |
| 群组 | 22 |
| 回放段 | 6 |
| 总记录帧 | 300 |
| 展示文件 | 14 |

既定检查通过后停止；没有执行全项目测试、build、lint 或 typecheck，没有重复已通过的检查。前端构建与浏览器检查由其他任务负责。

## 证据边界与剩余限制

- 回放来自 `data/diagnostics_v2/standard_stimuli_20261016` 的固定 LIF 诊断模型，未训练；不是最新训练策略，也不是活体记录。
- 逐帧颜色只代表记录的群组均值，同组节点共享均值；累计神经元计数只用于核验，不作为单元逐帧活动。未记录的节点没有虚构动态，动作与实体反馈未接入。
- 已核对实际导出的六段 JSON：`generator_hash_matches` 均为 true，当前与原记录的生成器源码 SHA256 均为 `f46d25892b55b904a2792896f6b9d74c2f864ced7f7104e82ef33290df455386`。每段 provenance 仍同时保存 `original_generator_sha256`、`current_generator_sha256`、原/重建 source_id 和原参数；未来摘要不同时会明确标识。预览始终为按原参数生成的刺激重建，不是原录像。
- 为避免加载大图，图拓扑 SHA256 只比较来源元数据中记录的身份，没有重新读取边数组计算拓扑摘要。该范围在 catalog 的 `topology_hash_validation` 明示；没有修改旧科研加载或来源校验。
- 旧模型、训练、诊断记录与其他数据保持只读，没有启动仿真、训练或设备。未修改 pyproject、README、gitignore；未提交或推送。

数据实施未发现待解决问题，交协调者整合与独立验收。
