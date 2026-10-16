# Acceptance — 离线神经活动记录与刺激分析

## Result

PASS

由未参与实现的独立验收 Agent 审查。实现满足 AC1–AC5 的软件与文档要求；后续状态收尾及用户要求的 commit 由协调者完成。

## Spec Coverage

| 标准 | 证据 | 判断 |
|---|---|---|
| AC1 | `activity.py` 延迟加载可选依赖，`collector.py` 只复用会话读取，不读取动作标签或实例化设备/动作读出；逐帧顺序推进。新目录以 `exist_ok=False` 独占创建，失败删除本次目录；已有目录拒绝覆盖。定向测试覆盖无标签、帮助依赖隔离、短/长视频失败与覆盖保护。 | PASS |
| AC2 | 群组计数使用全组成员均值并按 50 ms 换算 Hz；编码驱动独立保存。缺失群组为 NaN/空值/N/A，存在但无脉冲为 0 Hz。神经元轮流确定性抽样、去重及 1–4096 硬上限明确；测试用已知计数及超过浮点精确范围的 body ID 检查映射与单位。 | PASS |
| AC3 | `stimuli.py` 验证有限、非空、不越界、不重叠区间，以 AVI 播放时间筛选完整窗口；暖机仅排除比较、不跳过神经推进。先计算区间均值，再等权汇总同名刺激；只有显式有效 baseline 才计算差值。测试覆盖时基、暖机、跨界、重复刺激及无基线。 | PASS |
| AC4 | 实施报告记录 MiniFly 合成视频端到端生成 NPZ/CSV/JSON/HTML 和三张 PNG；保存的数组无 object/pickle，三种时间轴与计数换算有断言。协调者已经一次目检图像，轴、图例和单位可读、无截断。图表和文档明确限定为模拟信号、有限采样及合成软件检查。 | PASS |
| AC5 | `analysis` 可选依赖和 README、诊断指南、数据/架构文档同步；30 秒刺激安排、实际视频时间标注和新 CLI 一致。核心四文件相对 HEAD 无差异，旧序列化标识、控制与学习实现未修改；新格式独立为 `cnsbridge.neural_activity` version 1。输入来源、配置/bias、群组、采样索引、图与源码哈希可追踪。 | PASS |

## Validation

- Risk Level：S3，仅涉及新增持久化格式与可选依赖；检查限制在诊断子系统及其直接依赖契约。
- 已阅读 AGENTS、constitution、项目规则、验收流程、交接、Spec/Plan/Decision 004、实施报告、全部新增诊断源码与测试、相关现有 Brain/Encoder/会话读取实现，以及本轮文档和配置差异。
- 本验收实际执行 `git diff --check` 一次，退出码 0；执行核心四文件的 `git diff --name-only HEAD -- ...`，无输出，确认无修改。
- 审阅实施者已有定向验证证据：`.venv/bin/python -m pytest tests/test_neural_activity.py -q -s`，18 passed，退出码 0；未机械重跑。
- 接受协调者对 `data/neural_activity_demo/synthetic_32459262ba/report/` 三张静态图的一次目检记录；未重复出图或查看。
- 未执行全项目测试、构建、lint、真实录像分析、全量 MaleCNS 仿真、训练或硬件操作；本轮边界隔离且没有具体问题需要扩大验证。

## Issues

未发现与本次需求直接相关的未解决问题。

## Regression Risks

新增诊断入口只读取原会话并写新目录，不改变既有控制或学习链路。结果仍受固定相机、零 IMU、视觉近似和人工 bias 影响；有限样本及描述性刺激统计不能视为全脑代表或行为有效性证据。真实 MaleCNS 录像响应仍待用户实际采集后观察，当前未宣称已验证。

## Required Rework

无需返工。协调者更新完成状态并提交后停止，不追加重复验证。
