# M2 记录器修复独立验收

日期：2026-10-23。结论：**限定修复 PASS，可由协调者按第三轮原登记预算，在新目录开展实际技能探针。** 本次只验收记录器字段冲突，不代表 HOLD 或其他技能已实测通过，不改变 P2 和挥手取消的待验收状态。

本审查者未参与此次实施。风险 **S1**；仅静态检查修复、当前记录调用、真实 tick 返回路径及直接回归测试，不重新审查已 PASS 的 M1/M2 控制逻辑。

## 核对结果

- 已核对失败目录 `data/x2_sim/agibot/data/var/skills-20261023-141343-a8172c4d/probe-00-hold/` 的原始摘要与日志：0.153247667 秒失败，`behavior=null`、`events=[]`，异常为重复的 `monotonic` 关键字。该轮没有技能接受证据。
- 将当前 `experiments/x2_skills.py` 与该 run 的 `transport_src` 源码快照对比，差异只有 `_event_record()` helper 和 `write()` 使用该 helper；未修改控制目标、阈值、SDK、生命周期或旧失败资产。
- `write("tick", **core.tick())` 是当前唯一将动态载荷展开到顶层的记录调用。实际路径为 `SkillExecutor.tick()` → `VelocityCore.tick()` → `X2MotionAdapter.tick()`；返回字段含源 `monotonic`，并增加角向停稳字段，不含 `kind`、`phase`、`wall_time` 或 `received_monotonic`。新记录器的固定字段与该载荷互异，因此原来在 JSON 写入前触发的冲突已解除。
- 已逐一核对配置、预设请求/响应/查询、注册、里程计、MC、关节、回调异常、技能事件、探针命令、异常与最终零速的 `write()` 调用。其顶层显式字段均未占用上述保留名；执行器事件嵌套于 `event`，即使其自身含 `received_monotonic` 也不冲突；ROS 原始载荷嵌套于 `message` 或 `response`。
- helper 保留源载荷值并构造新字典，`write()` 将其交给原 `_EvidenceLog.record()` 的逐行 JSON 写入与 flush 路径。直接回归使用真实 `SkillExecutor.tick()` 启动载荷和 `_EvidenceLog`，核验完整载荷、两种时间、JSON 可写与输入未改，没有伪造接受或运动结果。

## 时间语义

tick 的 `monotonic` 是适配器在该 probe 进程内采样 tick 时的单调时间；顶层 `received_monotonic` 是同一进程进入记录函数后采样的记录接收时间，不能当作 ROS 消息的精确到达时刻。`wall_time` 是该记录调用采样的 Unix 时间。执行器事件自身的时间仍保留在嵌套事件中，ROS 源时间戳仍保留在原始 `message`/`response` 中。此次修改没有建立 macOS、Linux VM、ROS 之间的时钟对齐，也没有覆盖源时间。

## 验证依据与停止

复用 [实施报告](m2-runtime-fix.md) 中唯一相关检查：`.venv/bin/python -m pytest -q tests/test_x2_skill_records.py`，**1 passed in 0.01s**。本次已阅读该测试和真实写入路径，没有重跑、追加测试、build 或 typecheck，也没有启动 VM、容器、ROS、仿真、完整脑、训练或真机。

协调者后续须用修复后的新源码快照与新输出目录，保持原预算和原技能判据；旧失败日志继续保留。此 PASS 只解除已定位的记录器阻断，不预判实际探针结果，不自动放行用户挥手能力，不把第三轮运行准入写成 P2 完成。本次未改业务源码，未 commit/push，范围内复核完成后停止。
