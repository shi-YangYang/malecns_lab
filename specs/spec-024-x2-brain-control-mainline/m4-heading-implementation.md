# M4 方向任务目标输入实施

日期：2026-10-23。依据 [方向合同](m4-heading-contract.md)，范围仅为 `experiments/x2_behavior_bridge.py`、`learning/x2_behavior_session.py` 与直接 CPU 接缝测试。风险等级 **S2**：目标建立、跨进程证据与真实推理观测之间的相关模块交互。待新的独立验收；本报告不表示实际方向任务通过。

## 实施结果

- `RemoteExchange` 按 run/session/request 建立目标；首份有效且未过期的实际里程计给出起始 yaw，FACE_LEFT 为 +0.45 rad，FACE_RIGHT 为 −0.45 rad，目标和每次误差均按角度环绕。目标缓存不由请求轮询、gate-only refresh、实际推理或执行 COMPLETED 状态更新；同任务的新 request 从新的实际 yaw 建立新目标。非方向任务显式携带空目标，误差为 0。
- bridge 从同一个已接收的实际里程计提供原始 ROS 消息、序号、远端接收时刻，并保留建立目标的远端观测序号/时刻。`heading_target` 明确标记来源为用户请求的相对朝向，包含起始 yaw、固定目标、当前 yaw 和当前误差。它没有视觉/人物跟踪含义。消息在采样到发送之间经过的时间加入原身体反馈年龄，host 仍仅用本机 RTT 上界叠加年龄，没有相减 macOS/Linux 的不同单调时钟。
- host 核对目标身份、左右偏移、原始四元数、里程计时刻与序号、误差计算，以及同一 request 的目标不漂移；缺少目标元数据、缺少原始里程计或超时反馈不会获得伪造目标。原始目标和每次测量在 bridge 日志及 worker 独立 `heading_target` 记录中保留，不只依赖最后一个 mailbox 文件。
- `NeuralSession` 按 observation 序号保留对应目标。把先前拒绝投影进下一次实际脑输入时，额外归档当前请求的目标与当前身体反馈；拒绝归属仍来自被拒决策。目标不挪用旧请求，gate-only 轮询不消耗拒绝或 request_changed 边沿。
- 本层只更改已有 `Feedback.heading_error_rad` 输入与目标证据，不提交行为/速度/MC 请求、不调用 teacher、不决定下一展示动作或成功。M1 编码、M3 数据/模型/共享脑文件、严格候选恢复、独立急停与 wave 禁用保持既有实现。

## 真实检查

执行一次 `PYTHONPATH=src .venv-train/bin/python -m pytest -q tests/test_x2_behavior_desktop.py`：**24 passed，1 failed，0.78s**。唯一失败是新测试 fixture 将已观察的同一序号再次用修改后的拒绝反馈送入 gate，触发正确的 `Reordered observation` 拒绝；生产代码未因此改变。测试修正为实际路径中的下一份新观测，再仅执行 `-k rejection_projection_keeps_new_request_goal`：**1 passed，24 deselected，0.36s**。没有重跑其余已通过检查。

这些检查覆盖既有 R1–R4 接缝、左右符号与跨 ±pi 环绕、轮询固定目标、同任务新 request、非方向空目标、无效/过期/缺失里程计不建立目标、完成事件不归零、host 拒绝自洽但移动的目标，以及拒绝投影保存当前目标且只消费一次。最后阅读了限定修改段，未发现超出本任务的修改。未启动完整脑、MPS 运算、VM、ROS、仿真、训练或硬件，未 commit/push。

## 尚未验证

实际达到 0.45 rad 目标、净转向与剩余误差、脑三头原始选择、同次真实神经遥测仍须有界仿真记录。现有离线教师在一次转向 COMPLETED 后可能选择挥手；本次没有改写该数据或基线身份，因此输入修复不等于已学会准确对齐。真实资格记录必须分别报告原始脑决策、实际转角、剩余目标误差及保护情况。
