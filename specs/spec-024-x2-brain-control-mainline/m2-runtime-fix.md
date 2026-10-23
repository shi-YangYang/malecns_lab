# M2 首次探针记录器运行错误修复

日期：2026-10-23。风险 **S1**；仅修独立技能探针记录器的时间字段冲突，未改控制器、SDK、能力阈值或桌面文件。

## 事实与原因

实际目录 `data/x2_sim/agibot/data/var/skills-20261023-141343-a8172c4d` 保留不变。首个 HOLD 探针在 0.153 秒报 `TypeError: dict() got multiple values for keyword argument 'monotonic'`，`events=[]`，尚未有技能接受记录。

`control/x2.py::tick()` 返回含源采样时刻 `monotonic` 的字典，`control/x2_skills.py::tick()` 原样返回它。`experiments/x2_skills.py` 将它展开到 `write("tick", **core.tick())`，而记录器又传入同名 `monotonic` 关键字，因而在写入 JSON 前失败。

## 修改与时钟含义

- `experiments/x2_skills.py` 提取无设备依赖的 `_event_record()`，记录器时间改为 `received_monotonic`；源载荷的 `monotonic` 保持原值，不被写入时刻覆盖。
- `monotonic` 是适配器采样时刻；`received_monotonic` 是记录器收到待写记录时的本进程单调时刻；`wall_time` 是同次记录的 Unix 时间。ROS 原始时间戳仍在 `message` 中。三者不混作跨主机统一时钟。
- 检查同一记录函数全部调用：只有 tick 展开动态字段，实际 tick 不含 `kind/phase/wall_time/received_monotonic`；其他调用使用明确的互异字段，执行事件及 ROS 原始消息仍嵌套保存。未发现第二处当前调用的明确字段冲突。
- 新增 `tests/test_x2_skill_records.py`，以真实 `SkillExecutor.tick()` 启动载荷接入记录 helper 和 `_EvidenceLog`，核验 JSON 可写、源时间和记录器时间分别保留、原载荷未修改；不伪造技能接受或运动。

## 验证与边界

唯一运行检查：`.venv/bin/python -m pytest -q tests/test_x2_skill_records.py`，结果 **1 passed in 0.01s**。已将 runner 与该失败 run 的源码快照作 diff，差异仅为 helper 和记录调用。检查通过后停止，没有重跑已通过的控制器测试或扩大为 build/全量测试。

未启动 VM、容器、ROS、仿真、完整脑、训练或真机，未 commit/push。旧 run、源码快照与失败证据未改。此结果只修复已定位的记录错误，**不代表 HOLD 或任何技能实测通过**。须由新的独立验收者复核，后续实际运行由协调者另行安排新目录。
