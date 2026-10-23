# M4 候选执行时长绑定实施

2026-10-23。实施完成，待新的独立验收。风险等级 S2；本次只覆盖连续技能预算、M4 登记与事件证据，不代表候选仿真实测通过。

## 实际改动

- `control/x2_skills.py`：增加可选完整 `command_seconds_by_behavior` 配置，只接受 ADVANCE/TURN_LEFT/TURN_RIGHT 三个键及有限 `(0, 2]` 秒数值。构造时复制并只读保存；每次合法 START 固定本次时长、deadline 和原始强度，CONTINUE 不重新申请 TTL、不续期、不覆盖原强度。未提供配置时仍沿用原统一 1 秒默认及显式 2 秒入口。HOLD 原停稳语义和挥手原边界保持。
- `control/x2_skill_profile.py`：新增固定仿真资格配置 `malecns.x2.sim.advance1-turn2.v1`，基础技能包仍为 `malecns.x2.skills.v1`，分别记录。配置为前进 1 秒、双向转向各 2 秒、挥手关闭；严格核对注册内容，不接受界面或任务动态改时长。
- `experiments/x2_behavior_desktop.py`：宿主与 bridge 登记同一 profile；将新纯配置模块纳入 sources/transport 哈希快照；bridge readiness 及宿主结果带 profile。
- `experiments/x2_behavior_bridge.py`：ROS 导入、节点创建、命令发布之前，核对宿主/bridge 的会话、资格模式和固定 profile；生产执行器按登记构造。配置、readiness、执行事件及最终 summary 保留 profile，事件保留本次 `command_seconds` 与原强度。拒绝事件时长明确为 null；其中旧动作 summary 标记为上下文，不能当作被拒动作的执行证据。
- `tests/test_x2_skill_profile.py`：增加本次配置与边界的 CPU 回归，不改训练、数值模型、M2 runner/host、前端或旧运行资产。

`command_seconds` 表示本次连续技能的固定命令上限，不表示测得运动持续时间；停稳仍由原实际反馈与保护逻辑确认。HOLD 的该字段为 null，deadline 为接纳时刻。

## 验证

改动前文件临时副本位于 `/tmp/m4-skill-profile-before/`，用于本次未跟踪源码的限定 diff。已检查三个现有模块的前后 diff。

唯一 CPU 子系统检查：

```text
.venv/bin/python -m pytest -q tests/test_x2_skill_profile.py tests/test_x2_skill_duration.py
8 passed in 0.27s
```

覆盖生产宿主登记及 transport 拷贝、bridge 生产构造、1/2/2 秒 deadline/TTL 与实际命令到期归零、原强度保留、CONTINUE 不续期、调用者字典变更不影响已接纳动作、非法配置、宿主/bridge 不匹配在 ROS 导入前拒绝、拒绝事件没有执行时长，以及直接受构造参数变更影响的原单技能 CLI 1/2 秒贯通。默认 1 秒、HOLD 与关闭的挥手也在该范围检查。

检查通过后停止。未运行全量测试、build、lint、typecheck、脑计算、ROS、仿真或设备动作；未 commit/push。

## 结论边界

固定时长依据第四轮 ADVANCE 1 秒及第七轮左右转向 2 秒的单次、intensity=1 实测；一秒左转失败保持不变。尚未证明模型其他强度下产生有效位移、达到相对 ±0.45 rad 目标、完整多行为展示或实机能力。profile 只限制脑已选择动作的持续上限，不读取任务、不选择下一动作、不改变速度幅度或停稳判据。挥手取消资格仍关闭。
