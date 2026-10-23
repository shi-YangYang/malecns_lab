# M4 实际反馈来源关联独立验收

2026-10-23，**PASS（限定工程范围）**。独立审查者 `roadmap_m4_feedback_review`，按 [Ready 契约](m4-feedback-provenance-contract.md)及[实施报告](m4-feedback-provenance-implementation.md)审查当前 bridge、session 和相关测试；改动风险 S2。未发现本次来源关联修复的阻断项，未修改业务代码。

## 核对结果

1. **同步事件关联本次输入：PASS。** `SensorEvidence.update()` 在进入执行器前固定传感器名、唯一快照 ID、接收时刻和原文副本，设置 `latest/trigger`；`wrap_execution()` 对附带证据再次深拷贝。因此 `update_odom()` 同步触发 STARTED、`update_mc()` 同步触发 UNKNOWN 时，事件引用本次 callback，而不是上一帧。此时接受结果明确为 `pending/accepted=null`；回调返回后，以相同快照 ID 追加接受/拒绝、core 序号，既有事件不会被回写。另一传感器保留自己的时刻。证据：[`bridge:112`](../../src/malecns_lab/experiments/x2_behavior_bridge.py:112)、[`bridge:178`](../../src/malecns_lab/experiments/x2_behavior_bridge.py:178)、[`bridge:254`](../../src/malecns_lab/experiments/x2_behavior_bridge.py:254)；对应测试 [`test:503`](../../tests/test_x2_behavior_desktop.py:503)。
2. **Feedback 只引用已接受的对应 core 样本：PASS。** `accepted()` 同时要求接受为真、序号相同、接收时刻完全相同；`feedback_snapshot()` 对 odom 和 MC 均执行该核对，再从这两份 core 对象生成反馈。缺失或不匹配直接失败，不借用旧原文。已核对底层无效输入仍清空对应 core 的原有行为，未把“旧有效测量混入新拒绝原文”当作已发生问题。证据：[`bridge:143`](../../src/malecns_lab/experiments/x2_behavior_bridge.py:143)、[`control/x2:167`](../../src/malecns_lab/control/x2.py:167)；对应测试 [`test:527`](../../tests/test_x2_behavior_desktop.py:527)。
3. **完整响应先于覆盖邮箱归档：PASS。** `RemoteExchange.observe()` 为所有任务携带两种反馈来源、实际 Feedback、nonce、远端序号、采样/发出时刻和 host 查询时刻；非朝向任务也保留来源。`publish_observation()` 先追加完整 `observation_response`，记录器 flush 后才原子替换邮箱，再记录既有 `observation_sent`。固定文件名、记录种类和 nonce 构成引用；缓存和归档使用副本。host 的 `observation_clocks` 保留引用、各时钟和来源，传输 age 增量与远端原始 Feedback 可以区分。证据：[`bridge:66`](../../src/malecns_lab/experiments/x2_behavior_bridge.py:66)、[`bridge:169`](../../src/malecns_lab/experiments/x2_behavior_bridge.py:169)、[`bridge:230`](../../src/malecns_lab/experiments/x2_behavior_bridge.py:230)、[`session:158`](../../src/malecns_lab/learning/x2_behavior_session.py:158)；对应测试 [`test:549`](../../tests/test_x2_behavior_desktop.py:549)。
4. **实际入脑尝试与门禁刷新可区分：PASS。** `infer()` 先完成真实拒绝投影，再保留旧 `observation` 记录并新增 `neural_input(status=attempted)`，绑定实际传给 policy 的完整对象、observation/decision 序号及 request ID。只有完成计算才有后续 `raw_decision`；policy 中止不会伪造完成，也不会消费该请求边缘或待投影拒绝。`gate_refresh` 区分只接收执行事件的轮询与确实更新门禁的较新 Observation；前者不冒充入脑，也不改变门禁观测。证据：[`session:197`](../../src/malecns_lab/learning/x2_behavior_session.py:197)、[`session:252`](../../src/malecns_lab/learning/x2_behavior_session.py:252)、[`session:383`](../../src/malecns_lab/learning/x2_behavior_session.py:383)；对应测试 [`test:579`](../../tests/test_x2_behavior_desktop.py:579)。
5. **范围与控制边界：PASS。** 新原文证据位于传输/日志字段，没有加入 Observation schema 或神经读出。新增用途记录接在已有 snapshot/refresh 调用点，没有增加订阅、后台轮询或动作。普通拒绝投影、请求边缘消费、相对朝向、控制权和保护停止仍由原有逻辑执行；本验收不对这些既有能力重复全面验收。

## 检查与限制

实施者已记录一次最相关 CPU 检查：`test_x2_behavior_desktop.py -k 'provenance or old_task_raw_output_is_archived_and_state_never_rolled_back'`，结果 **7 passed、24 deselected、0.75s**。本次核对相应回归内容与生产调用路径，未发现需要新复现的具体疑点，因此没有重复该检查，也未运行全量测试、build 或 typecheck。当前相关文件尚未入库，验收依据为工作区现有实现、契约及实施报告，不将整份未跟踪文件视为本次新增差异。

本审查只新增此报告。未启动 MPS、完整脑、ROS、仿真、Electron 或硬件，未读取锁定测试，未改训练来源/旧 run，未 commit/push。同步事件错配是源码路径及 CPU 回归覆盖，不是历史实测已发生错配的证明。真实采集时延、进程/机器崩溃下的持久化耐久性、真实反馈训练、P3/P5 任务成功及真机表现不在本次通过结论内；新的实际 M4 run 仍需训练退出后登记来源并验证。
