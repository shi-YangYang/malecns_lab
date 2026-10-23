# M4 方向目标输入独立验收

日期：2026-10-23。角色：新的独立只读验收者。依据 [方向合同](m4-heading-contract.md)与[实施报告](m4-heading-implementation.md)，审查限定为 bridge、session 及其直接 CPU 检查；风险等级 **S2**。

**结论：工程准入 PASS。未发现本次方向输入接缝的确定缺陷。该结论不表示实际准确转向、P3 或 P5 通过。**

## 核对结果

| 要求 | 审查证据与结论 |
|---|---|
| 首份有效里程计建立请求目标 | `RemoteExchange.observe()` 在原身体反馈有效、年龄未超限且原始四元数校验通过后，才提交目标缓存与已发观测。方向目标绑定 run/session/request/task，左 +0.45 rad、右 −0.45 rad；无效、过期或缺少原始里程计会拒绝，不写入目标缓存。实际 bridge 仅对有效身体快照应答。 |
| 固定目标与角度环绕 | `heading_goal()` 排除当前测量字段，保留目标及建立证据；bridge 与 host 分别检查同一 request 的不可变目标。后续误差只由固定目标减当次实际 yaw 并经 atan2(sin,cos) 环绕得到。相同任务的新 request 重新建立目标，正负方向及跨 ±pi 情形有直接检查。 |
| 完成状态不伪造对齐 | 目标更新不读取执行状态；RUNNING、COMPLETED 与普通轮询共用相同计算。已有检查明确保留 COMPLETED 后仍大于 0.2 rad 的实际剩余误差。非方向任务显式空目标与零误差，仅表示该任务不要求朝向。 |
| 原始测量、时钟与归属 | bridge 的身体 yaw 与 `current_odom` 来自同一次已接受里程计；回调拒绝里程计时底层清空 odom，不能继续借旧记录建立目标。host 由原始 quaternion 重算起始/当前 yaw、固定目标和误差，核对序号、远端时间及来源。采样至发送耗时累加到原反馈年龄，传输仅累加本机 RTT 上界，不相减主机与 VM 单调时钟。 |
| 归档与拒绝投影 | bridge 持久记录发送的目标；worker 为每个 observation 记录独立 `heading_target` 与时钟证据。拒绝投影保留被拒决策归属，同时保存当前 request 的目标和当前身体反馈，不复用旧目标；实际脑输入仍保留当前 yaw/heading error。 |
| gate-only 不消费输入 | `snapshot()`、`heading_evidence()` 与执行事件轮询仅采样、存档或更新 gate。`last_consumed_request_id` 与待消费拒绝只在实际策略调用成功返回后更新/弹出；目标不随轮询移动。既有同任务新请求边沿检查与本次拒绝投影检查相互对应。 |
| 不绕过神经选择 | 新逻辑只修改观测的 `Feedback.heading_error_rad` 及审计元数据；没有新增 task 到速度/行为/MC 的分支、teacher 调用或任务成功判定。MC 提交仍经过已有原始神经决策与 gate。 |

## 验证范围

已阅读并采纳实施者的一次相关子系统检查：**24 passed、1 failed**；失败来自测试对同一观测序号重复注入修改后的反馈，修正 fixture 后仅对应检查 **1 passed、24 deselected**。本验收未机械重跑已通过检查，静态阅读没有出现需要新增 CPU 复现的确定疑点。

另外用标准库 SHA-256 对照 r2 源码快照与登记，确认 `brain/mps_trainable.py`、`learning/x2_behavior_policy.py`、`learning/x2_behavior_training.py`、`experiments/x2_behavior_training.py`、`learning/x2_behavior_data.py`、`senses/x2_behavior.py`、`control/x2_behavior.py` 七个相关 M1/M3 文件仍与 `data/x2_behavior_probe_20261023_r2` 完全相同。首次定位快照目录未命中，改用实际 `sources/malecns_lab/` 后完成比对；没有将未命中当作源码差异。

本验收只新增本报告，未修改业务或测试；未启动完整脑、MPS、VM、ROS、仿真、训练或硬件，未 commit/push。

## 实际资格边界

可使用严格身份匹配的登记候选开展有界实际方向任务联调。必须分别记录脑的原始行为/阶段输出、实际净转角、固定目标的剩余误差、执行保护与同次真实神经遥测；运动方向正确不等于达到 0.45 rad 目标。现有离线教师在一次转向 COMPLETED 后可能选择挥手，本次没有修改该数据或模型，也未验证实际目标精度。wave 取消资格、完整 P3/P5 和现场 P6 仍由对应阶段验收。
