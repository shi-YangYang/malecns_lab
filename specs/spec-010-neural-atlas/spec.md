# Spec 010 — 本地 WebGL 神经活动图谱

状态：Done；[独立验收](acceptance.md) AC1–5 全部 PASS。2026-10-21 用户要求先完成可视化，并询问 WebGL、原生、Electron、Tauri 选型。采用本地浏览器 WebGL；Python 只读导出已有记录，暂不增加桌面壳、云发布、训练或设备控制。该选择可逆，沿用已有完整 MaleCNS 资产和证据边界。

## 交付

1. `apps/atlas/`：Vite + TypeScript + Three.js 本地页面。中文研究展示界面，三维脑胞体点云、群组活动、刺激预览、暂停/播放/时间轴/速度、刺激切换、群组选取、视角复位和解释面板。
2. 简洁的三维果蝇示意模型，明确不是神经驱动的肢体仿真；无动作记录时不编造动作。
3. `src/malecns_lab/visualization/`：将完整参考图的节点名单、官方注释的 somaLocation 与 V2 诊断记录导出为浏览器数据。只加载必需数组/列，不加载大图边矩阵，不启动神经仿真。
4. 数据导出到忽略的 `apps/atlas/public/data/`，前端依赖本地安装、锁定版本；离线演示无需 CDN。缺数据/格式错误给出明确提示，不回退到合成脑。
5. 使用说明、项目入口与技术栈同步。保留现有训练、诊断和序列化契约。

## 数据和展示语义

- 所有参考节点保留在计数与覆盖统计中；缺失位置明确计数，不虚构空间坐标。点云是胞体坐标，不冒充完整神经形态或全部突触线。
- 默认复用 `data/diagnostics_v2/standard_stimuli_20261016` 的六种已记录刺激；它是固定 LIF 诊断模型的仿真回放，不是最新训练策略，也不是活体测量。
- 逐帧颜色对应记录的群组平均放电率；同一组节点共用组均值，清楚标注。单元累计计数不冒充逐时刻活动；没有记录的节点保持静态背景。
- 显示原记录帧与有效性、预热区间、仿真时间、来源摘要。插值只用于显示时明确注明，优先按原帧步进。
- 刺激预览按记录参数重建并标注“刺激重建”；导出文件记录当前生成器摘要与原摘要。没有来源记录则不生成假视频。
- 仿真动作与实体反馈均显示未接入，不从神经率硬编码造动作。

## 前后端静态契约（schema `malecns_lab.atlas.v1`）

`catalog.json`：`schema`, `graph: {node_count, edge_count, source_id, graph_sha256}`, `atlas_url: "atlas.json"`, `model: {name, source_id, trained: false, description}`, `recording: {name, source_path, frame_hz}`, `runs: [{id, stimulus, label, url, frame_count, duration_s, warmup_s}]`, `provenance`, `notes`。

`atlas.json`：`schema`, `node_count`, `positioned_count`, `missing_count`, `positions`（扁平 xyz，统一居中缩放、变换记录在 `coordinate_transform`）, `body_ids`（与坐标同序，字符串避免整数精度损失）, `coordinate_transform`, `groups: [{name, point_indices, total_neurons, positioned_neurons, centroid}]`。没有成员的 centroid 为 null。

每个 run JSON：`schema`, `id`, `stimulus`, `label`, `groups`（顺序对应 rate 列）, `time_s`, `valid`, `group_rates_hz`（帧×组，缺失数值 null）, `network_spikes`, `network_active_neurons`, `warmup_s`, `frame_hz`, `source_id`, `model_source_id`, `stimulus_preview: {url,width,height,frames,layout:"vertical",kind:"reconstructed"}`, `provenance`, `notes`。PNG 精确按当前已有生成器及原 texture_seed/时序重建，非摄像头原录像。

前端读取 catalog，支持 404/损坏数据、零帧与空群组的合理状态。时间轴严格对应记录时间，切换刺激后不留旧数值。前端以共同色标比较刺激，图表提供 Hz 单位。

## 验收（S2；新前端构建属于必要构建检查）

- AC1：用现有完整图/官方注释/真实诊断记录成功导出，图身份、位置覆盖、群组和数组一致。
- AC2：网页成功加载真实点云及六种回放，交互控件有效；无网络 CDN 依赖。
- AC3：模型身份、群组均值、缺位置、刺激重建、仿真回放与未接入动作等边界可见。
- AC4：无修改旧模型/训练/记录；不启动训练或硬件，不触碰无关未跟踪审计文件。
- AC5：导出器定向测试、前端构建和一次浏览器交互/截图检查（若可用）通过；没有浏览器能力须明确记录限制。不运行全仓库测试。

## 实施责任

数据实施 Agent 独占 visualization 与对应 tests，前端实施 Agent 独占 apps/atlas；两者通过本契约协作。协调者维护文档与治理文件，负责本地预览交接；实施结束后独立验收者只读审查，不重复已通过检查。
