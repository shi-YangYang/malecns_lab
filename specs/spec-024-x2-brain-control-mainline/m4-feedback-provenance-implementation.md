# M4 实际反馈来源关联实施

2026-10-23。按 [Ready 契约](m4-feedback-provenance-contract.md)实施，风险 S2。实现与定向检查完成，等待新的独立验收；不据此宣称实际反馈训练或任务行为通过。

## 变更

- `experiments/x2_behavior_bridge.py`：每个 odom/MC 回调在调用执行器前固定独立快照 ID、原始 message 副本和一次 Linux 单调接收时刻。同步 STARTED/UNKNOWN 的 `triggering_snapshot` 与 `sensors` 引用本次输入，接受结果明确为 `pending`；回调返回后，在既有 odom/MC 追加记录中保存接受/拒绝结果及实际 core 序号。后续接受结果不会修改已经归档的事件。其他传感器各自携带接收时刻；arm 仍只保留最近快照，不新增逐关节流归档或订阅。
- `feedback_snapshot()` 只使用与当前 core 序号、接收时刻完全相同的已接受 odom/MC 快照；缺失或不匹配报错。拒绝时 core 清空的既有语义不变，不回退到上一份原文。非朝向任务也保留两种来源。
- `RemoteExchange` 在现有传输包中附加来源与时钟证据，原始 message 不进入 Observation 或读出。`observation_response` 先将完整响应追加到 `bridge_events.jsonl`，再更新邮箱；既有 `observation_sent` 保留并指向固定文件、记录种类和 nonce ID。完整响应包括实际 Feedback、各来源时刻、nonce、远端序号和 host 查询时刻；主机 `observation_clocks` 保留对应记录引用、采样时刻与来源。缓存及归档使用副本，避免调用方后续修改改变证据。
- `learning/x2_behavior_session.py`：保留 `observation` 记录并附加用途。实际计算前的 `neural_input` 位于真实拒绝投影之后，记录完整 Observation、observation/decision 序号、request ID 和 `status=attempted`。完成计算继续由 `raw_decision` 关联；计算中止不补造完成。`gate_refresh` 区分更新门禁观测与只接收执行事件的轮询，并记录对应 Observation；不增加轮询或消费请求边缘。
- `tests/test_x2_behavior_desktop.py`：补充真实执行器的同步 STARTED/UNKNOWN 来源、接受/拒绝及序号/时刻匹配、非转向响应归档、拒绝投影后的入脑尝试、门禁刷新和中止记录回归。已有一项因新增明确日志而调整尾部日志序列断言。

没有改 `control/x2_skills.py`、Observation schema、模型、训练数据、训练来源或桌面模块；动作、门禁、相对朝向、控制权与保护停止规则保持原逻辑。未改历史 run。

## 实际检查

只运行一次定向 CPU 命令：

```sh
PYTHONPATH=src .venv-train/bin/python -m pytest -q tests/test_x2_behavior_desktop.py -k 'provenance or old_task_raw_output_is_archived_and_state_never_rolled_back'
```

结果：**7 passed, 24 deselected in 0.75s**。没有构造完整脑、启动 MPS/ROS/VM/Electron、读取锁定测试或运行全量检查；已通过后停止测试。已核对本次编辑片段及范围。未 commit/push。

## 限制

本次 CPU 回归复现的是同步事件来源关联路径，不是历史 run 已发生错配的证据。旧记录不补写；新的实际 M4 run 需在当前训练退出后登记新来源。实际采集时延、真实反馈训练、P3/P5 与真机表现仍未验证；数据导出器和教师标签不在本次范围。
