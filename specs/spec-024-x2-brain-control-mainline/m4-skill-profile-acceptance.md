# M4 固定执行时长 profile 独立验收

2026-10-23。结论：**PASS（限定工程范围）**。验收者未参与本次实施；风险等级 S2。没有发现本次改动范围内需要返工的问题。

## 范围与证据

依据 `m4-skill-profile-contract.md` 与 `m4-skill-profile-implementation.md`，检查新增的 `control/x2_skill_profile.py`、`tests/test_x2_skill_profile.py`，并用 `/tmp/m4-skill-profile-before/` 中的实施前副本逐一检查 `control/x2_skills.py`、`experiments/x2_behavior_bridge.py`、`experiments/x2_behavior_desktop.py` 的限定 diff。其余已有 M4 朝向、反馈来源与 M2 兼容工作不重新验收。

- **脑选择与时长职责分离**：固定 profile 为 ADVANCE 1 秒、TURN_LEFT/TURN_RIGHT 各 2 秒。执行器只按已接纳的行为查询时长，生产路径仍经过 `DecisionGate.admit` 再提交脑决定；新增配置代码没有读取任务或选择下一动作。`VelocityCore.request_velocity` 的 `.2 × intensity`、方向、完成阈值与停止保护没有改动。
- **期限在接纳时锁定**：构造器严格检查完整连续行为键集合和有限 `(0, 2]` 数值，复制后以只读映射保存。合法 START 固定 `active_command_seconds` 和 deadline，再将该时长传入原速度 TTL；CONTINUE 在此前返回，不修改期限、TTL 或原命令强度。调用者随后修改原字典不能改变执行配置。
- **配置在动作前一致**：宿主同时写入 host/bridge registration，纯配置模块进入来源快照与 transport manifest。bridge 在导入 ROS、构造 Node 或发布命令前核对会话、仿真资格、关闭挥手及完整固定 profile；执行器再次核对后构造。readiness 和双方 summary 保留该 profile。
- **记录区分预算与实测**：基础 `skill_pack_id` 保持原身份，单独记录 `skill_profile_id`。执行事件与 core summary 保存本次固定命令上限、原强度、接纳时间和 deadline；该时长没有被称为测得的运动时间。拒绝事件的 `command_seconds` 为 null，附带的旧 core summary 明确标为上下文，避免当作被拒动作的执行证据。
- **旧入口与限制保持**：未提供逐行为配置时，原统一 1 秒默认和单技能显式 1/2 秒仍贯通。HOLD 保持站立确认语义，命令时长为 null。生产候选能力集合仍不含 WAVE_RIGHT，`wave_probe=False` 且 preset 回调拒绝执行；没有放开设备或 renderer 调整入口。

## 验证与停止

继承实施者已通过的唯一 CPU 子系统检查：

```text
.venv/bin/python -m pytest -q tests/test_x2_skill_profile.py tests/test_x2_skill_duration.py
8 passed in 0.27s
```

已阅读这些检查对生产登记/构造、1/2/2 秒期限、CONTINUE、外部字典变更、非法配置、动作前拒绝、事件记录和旧单技能入口的覆盖。本次独立验收只做代码与 diff 审查，没有重复运行已通过的检查，没有新增测试、build、lint、typecheck，也没有启动脑、ROS、VM、仿真或设备。仅新增本报告，不修改业务代码；PASS 后停止本范围工作。

## 结论边界

此 PASS 允许固定时长 profile 进入后续已授权的候选联调，不代表 M4 实际联调或 P2/P5 展示通过。现有时长依据仅为 intensity=1 的单次前进与双向转向实测；模型其他强度下的有效运动、相对 ±0.45 rad 目标和完整多行为任务仍待实际验证。挥手取消资格继续关闭，真机待上海现场接入。
