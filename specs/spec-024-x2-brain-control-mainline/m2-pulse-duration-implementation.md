# M2 连续技能脉冲时长参数实施

日期：2026-10-23。范围：只接通显式 `--command-seconds`，供新的有界转向实验登记使用。风险为 S2：host CLI、远端 probe CLI 与运行记录跨模块传递；未改核心控制或官方组件。

## 修改

- `experiments/x2_skill_host.py` 与 `experiments/x2_skills.py` 均接受 `--command-seconds`，默认 1 秒，只允许有限值 1 或 2；在 preview 和实际启动前校验。host 构造器也校验，不能通过直接构造绕过边界。
- host 构造器以默认值为 1 秒的新增 keyword 保存时长，向远端 CLI 显式传递；runner 将解析值交给实际 `SkillExecutor`。host preview、运行事件、host summary、runner preview、configuration 与 runner summary 记录选中的实际值；runner configuration/summary 在核心已构造时从核心读取。
- 保留 M4 的四参数 `SkillRuntime` 构造方式及 1 秒默认值。`control/x2_skills.py`、MC、AimDK、方向、速度上限、成功与停稳阈值、生命周期时限均未修改。
- 此参数只影响原有连续动作脉冲。WAVE_RIGHT 继续使用原核心的预设轨迹与 12 秒观察期限，不会被改成 1 秒或 2 秒挥手；CLI help 已明示该边界。

## 定向验证

唯一验证命令：

```text
PYTHONPATH=src .venv-train/bin/python -m pytest -q tests/test_x2_skill_duration.py
```

结果：**1 passed in 0.03s**。通过后未追加或重复运行其他检查。

该检查用隔离的 host/ROS transport doubles，从真实 host CLI 经远端命令解析到真实 runner 和真实 `SkillExecutor` 构造，分别核验默认 1 秒及显式 2 秒、两层 summary 与 configuration 的一致性。服务替身在动作源注册阶段立即拒绝，测试只发布最终零命令，不执行动作或定时等待；同时核验 0、负值、1.5、超过 2、NaN、正负无穷被拒绝，M4 默认构造保持 1 秒，原速度与 wave/wall 常量不变。

## 限制与下一步

本次未启动 VM、ROS、仿真、完整脑或硬件，未修改历史 run，未 commit/push。测试只证明参数与证据贯通，不证明 2 秒左转能够成功。由新的独立验收者审查后，协调者登记新的实际条件探针；原 1 秒左转失败证据与成功/停稳判据继续保留。
