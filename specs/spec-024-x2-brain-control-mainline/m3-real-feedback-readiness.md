# M3 真实执行反馈准备度

2026-10-23，只读评估。结论：**M2 已有真实身体反馈素材，但没有完整互动训练片段；下一次有界 M4 qualification 是取得任务—脑决策—实际反馈连续片段的最小入口。** 本文不改数据、训练或评分协议，不启动运行。

## M2 能保留什么，不能补造什么

实际来源为 `data/x2_sim/agibot/data/var/skills-20261023-151800-93a1e497/`。HOLD 的 `probe-00-hold/events.jsonl` 有280条 odom、12条 MC、3条 execution；ADVANCE 的 `probe-01-advance/events.jsonl` 有2458条 odom、78条 MC、3条 execution。二者分别确认站立、前进0.442807883米后停稳；这是执行器探针，不是脑选择成功。原始来源与结果见 [m2-actual.md](m2-actual.md:29)。

| Observation/Feedback 字段 | 可用依据与边界 |
|---|---|
| `yaw_rad/vx_m_s/wz_rad_s/mc_mode` | 接受的 `odom.message.pose.pose.orientation`、`twist.twist.linear.x/angular.z`，以及 `mc.message.action_info.action_desc`；保留实际值、单位与来源。不能拿目标速度替代测量值。 |
| `status/behavior/execution_id/request_id` | `execution.event` 的原始阶段、行为与归属；HOLD 在行815–817，ADVANCE 在行2564/2845/7143依次记录 ACCEPTED、STARTED、COMPLETED。只有阶段事件到达之后，后续观测才能使用它；没有 RUNNING 事件就不补造该阶段。 |
| `standing/valid/age_s` | `standing_confirmed` 是当时的站稳证据；其他采样点需按当时源码、仅此前的 odom/MC、持续窗口重建。不能用最终 summary 的成功标记填满整段。原版 `transport_src/malecns_lab/control/x2_skills.py:230–249` 要求新鲜反馈、模式/姿态/速度及1秒稳定窗口，单个 `body_status=1` 不够。 |
| 时刻与来源 | `received_monotonic` 是 Linux 探针记录接收时刻，tick另有源 `monotonic/odom_age/mc_age`，ROS时刻保留在 message。记录器回调时刻略晚于内核接收时刻，不能冒充精确内核或 macOS 入脑时刻。导出必须保留时钟域、原文件hash/行号与重建规则。 |
| `perception` | 没有相机，保留 `valid=false/source=missing` 和零视觉值；不由仿真画面或机器人位置生成“识别到人”。 |

**尚不能无伪造构造完整 Observation：** 两个 probe 的命令均为 `decision_seq=0/request_id=isolated-skill-probe`，执行ID也均为 `skill-0`，没有 Task、神经消费序列、request_changed、策略状态或行为头输出。`ADVANCE` 不能倒推成用户曾发 `SHOW_FORWARD`，HOLD也不能倒推成 `END_INTERACTION`；两段独立进程不能拼成同一脑互动。导入身份需按原run/probe命名空间区分并保留原ID。探针只启用 `(HOLD, 当前技能)`，不能填成五技能均可用；两项也没有相对朝向任务目标，不能从最终姿态构造 `heading_error_rad`。契约的必填字段见 [`control/x2_behavior.py:112–160`](../../src/malecns_lab/control/x2_behavior.py:112)，探针命令来源见 [`experiments/x2_skills.py:266–291`](../../src/malecns_lab/experiments/x2_skills.py:266)。

同run左转失败必须保留为真实失败轨迹；其停稳不改写为转向成功。启动超时的空probe不能生成神经缺反馈样本。现有素材缺少普通结束、真实重复/切换请求、多行为完成链、拒绝/过期反馈的入脑反应、右转及挥手。可用于反馈解析和实际时延研究，不能直接当完整五任务示范。

## 下一次 M4 qualification 如何留下可训练片段

已有链路足以保留真实行为片段的核心：

- `task_requests.jsonl` 记录所有任务请求与host接收时刻；`behavior_registration.json`、`sources/`、`official_assets.json`绑定候选和来源（[`x2_behavior_desktop.py:108–163`](../../src/malecns_lab/experiments/x2_behavior_desktop.py:108)）。
- `behavior_events.jsonl` 中 `observation` 保存传给模型的完整对象；`raw_decision.decision.observation_seq`、状态ID及三头输出关联本次消费。**按 raw_decision 关联挑出已完成入脑计算的观测，不能将所有 observation 日志当训练帧**：推理后新请求刷新门禁也会记录同种日志，但未进入这次脑计算。异常中止的计算保留为不完整段（[`x2_behavior_session.py:194–273`](../../src/malecns_lab/learning/x2_behavior_session.py:194)）。
- `execution` 与 `bridge_events.jsonl` 保存真实MC发布、odom/MC、接受/完成/拒绝及停止证据。`rejection_neural_input`同时记录真实拒绝和替换前身体反馈；训练使用实际送入脑的对象，不把它改写成另一技能成功。`observation_clocks`保存nonce、host发送/接收、RTT上界及Linux源age；朝向目标保存原始和当前odom，不能把完成标记替代误差（[`x2_behavior_bridge.py:106–127`](../../src/malecns_lab/experiments/x2_behavior_bridge.py:106)、[`227–277`](../../src/malecns_lab/experiments/x2_behavior_bridge.py:227)、[`x2_behavior_session.py:370–390`](../../src/malecns_lab/learning/x2_behavior_session.py:370)）。
- `telemetry/` 的manifest和批次关联session、checkpoint、input_map、observation/decision/request及已有技能执行ID；遥测只是同实例证据，不提供教师标签（[`x2_behavior_telemetry.py:16–24`](../../src/malecns_lab/learning/x2_behavior_telemetry.py:16)、[`59–67`](../../src/malecns_lab/learning/x2_behavior_telemetry.py:59)）。

原始快照存在一个需限定修复的血缘缺口：`on_odom/on_mc` 无论 `accepted` 真伪都覆盖 `raw_latest`，但不能据此断言旧有效测量会混入新拒绝原文——当前内核拒绝时会把对应 `core.odom/core.mc` 置空，`feedback()`先拒绝缺失（[`control/x2.py:167–203`](../../src/malecns_lab/control/x2.py:167)）。**能够由调用顺序确认的错配是 execution 的附带原始传感器**：`SkillExecutor.update_odom()`可在返回前同步发出STARTED，`update_mc()`可同步发出UNKNOWN；此时bridge的 `execution()`已读取尚未更新的 `raw_latest`，事件触发测量与附带原文可相差一帧（[`x2_skills.py:156–179`](../../src/malecns_lab/control/x2_skills.py:156)、[`x2_behavior_bridge.py:173–192`](../../src/malecns_lab/experiments/x2_behavior_bridge.py:173)）。这是源码可达路径，本次未运行复现，不能称已发生的真实run故障。

最小后续实施限定为两项，均等待当前训练退出、另行登记来源后进行：

1. **补齐采样血缘日志。** 每个callback先固定不可变原始快照ID、接收时刻和message，再记录core接受结果；有效Feedback只引用与core序号/接收时刻一致的已接受odom及MC，拒绝原文另留日志，execution另关联触发callback，避免引用上一帧。现有 `observation_response.json` 是覆写邮箱；`observation_sent`没有完整Feedback和全部原始传感器引用，非转向任务也没有heading_target中的原始odom。每次接受快照追加完整packet或不可变引用，并绑定文件/记录位置；明确 `neural_input` 与 `gate_refresh`，保留实际入脑前拒绝投影。修改限于实际快照来源/归档关联，不改模型、Observation schema、控制或执行语义。连续关节回调目前只更新 `raw_latest['arm']`，未逐条落盘；本轮wave禁用，不把这些零散快照当挥手轨迹。
2. **离线只读导出器与新数据登记。** 从不可变run提取完整session的入脑观测链、任务边界、决策、实际结果和保护标记；教师标签单独离线产生，只看当前可部署观测及此前已到达反馈，绝不读取未来COMPLETED、最终summary或模型预测来补正确答案。没有获得的阶段留缺，不重采样伪造CONTINUE；保护归零不算普通END成功。当前v1校验假定帧序号连续且session_id等于episode_id，导出若重编号须另存原始序号映射并登记新数据身份，不能修改原档案凑格式（[`x2_behavior_data.py:166–183`](../../src/malecns_lab/learning/x2_behavior_data.py:166)）。

一次qualification只在已登记任务/预算内收集自然发生的行为，候选拒绝或失败也全部保留；不通过直接MC脚本强行补足成功。当前wave不开放，turn仍需能力资格，因此不能保证取得完整五任务链。脑跨请求保留状态，拆分边界至少为完整session/run，同一连续记录的相邻片段不得分到训练和验证；普通结束与独立急停分别存档。最终新锁定评估须有独立真实反馈采集，不能把用于调参的qualification再称未见评估。

## v1 锁定集边界

v1明确为 `simulated_execution_feedback`，train200/validation50/locked-test100按完整片段和参数族隔离；教师目标位于Observation之外，评分是离线教师序列一致性（[`x2_behavior_data.py:145–159`](../../src/malecns_lab/learning/x2_behavior_data.py:145)、[`200–220`](../../src/malecns_lab/learning/x2_behavior_data.py:200)）。**保持现有文件、manifest、hash与评分规则不变；不读取锁定集选样、调参或建立真实数据转换规则。** 新采集另建数据身份和预登记拆分，不把真实记录插入v1后仍沿用原身份；v1未来得分也不能单独满足Spec024要求的真实反馈P3。Spec要求见 [spec.md:101–110](spec.md:101)。

本次仅检查上述源码和已发生M2记录，文档改动S0；未读取locked-test文件、未运行测试/模型/仿真、未改变任何数据与Python来源、未commit/push。
