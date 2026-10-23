# M2 实施报告 — 官方单技能执行与有界核验入口

状态：业务实现与定向检查完成，待新的独立审查和协调者实际仿真。风险 S3（控制互斥、有效期、所有权与停止）。本报告不表示 P2 已通过，未启动 VM、仿真、训练、完整脑或硬件。

## 实际修改

- `src/malecns_lab/control/x2_skills.py`：无 ROS 的单技能执行核心，保留原 `control/x2.py` 不动；复用原来源 `malecns.x2_mvp` 的注册、一次空来源认领、0.5 秒来源确认、反馈有效期、直立及有界速度守卫。局部 `VelocityCore` 增加左右转目标与独立角速度/角漂移停止证据。HOLD 的初始稳定站立允许空来源，但不会因此发布运动；只有显式移动请求能进行受限认领。
- `src/malecns_lab/experiments/x2_skills.py`：单个官方技能的 ROS 核验进程；默认 preview 不导入 ROS/Torch/NumPy、无 IO。必须 `--execute --sim-only`、ROS domain 47、localhost 1 和新输出目录。真实 odom、common state、手臂关节以及预设请求/查询原文归档。
- `src/malecns_lab/experiments/x2_skill_host.py`：主机独占私有 GPU 仿真生命周期，复用现有桌面 Runtime 的守卫/停止/关闭方法；不启动脑。持有桌面锁及完整脑互斥锁，至少保留 1 GiB 系统内存；不接管已启动的 VM/容器，不切换默认 context，不触碰 Noria。源码快照和官方文件哈希在启动前落盘，并在容器内复核。
- `tests/test_x2_skills.py`：假时钟/假反馈测试，不启动 ROS 或设备。

`SkillExecutor.submit_decision(decision, observation_received)` 仅转换已经通过 M1 `DecisionGate` 的真实神经决策。执行器再次拒绝重复序号、过期、非法行为/阶段和在途冲突，CONTINUE 不续期也不发新预设请求。执行器不含任务到动作选择表；独立 probe 是显式设备能力核验，不称脑控。

## 冻结的运行预算与判据

协调者已确认：连续目标最大 `vx=0.2`、`|wz|=0.2`，FLU 正 wz 为左；默认发 1 秒，核心上限 2 秒。目标值不是实测速度。startup 12 秒；连续动作停止观察最多 12 秒；波形自然完成观察最多 12 秒；单 ROS probe 总墙钟 40 秒。主机总墙钟 240 秒，提前留 30 秒退出/资源关闭，不在剩余不足 41 秒时启动下一 probe。

停止沿用原水平速度 `<0.02 m/s`、新鲜双主题持续 1 秒和位移漂移限制，并新增 `|wz|<0.03 rad/s` 持续 1 秒，累计角漂移不大于 `0.03 rad`。实际转动需观测绝对角变化超过 0.02 rad 且方向正确；实际前进需观测位移超过 0.02 m。发出目标或 MC ACK 均不计作真实运动。数据缺失、所有权丢失、反馈过期、时钟错误、异常退出都先归零；停止未确认记 UNKNOWN/失败，不能写 STOPPED。

## 挥手证据与明确限制

已读取当前本地 AimDK `SetMcPresetMotion.srv`、`GetMcPresetMotionState.srv`、`McMotionStatus.msg`、`McPlayerState.msg`、官方预设动作文档及 MC `animation_player.yaml`：

- 右挥手使用 motion 1002、area 2，且 `interrupt=false`。CSV 2141 行、末尾 4280 ms，是 4.28 秒官方轨迹资源；这不是实际运动记录。
- 预设动作无连续速度的来源优先级。查询只报告 400 执行中/1 完成；不能把 `GetCurrentInputSource` 当成手臂归属。执行核心要求一次请求在途、实际 task_id 回应匹配、common motion/area/type 匹配，出现外部打断/错误/无法归属记 UNKNOWN。
- 自然完成需要：ACK、查询 RUNNING、common PLAYING、右臂七关节实际相对初态至少 0.1 rad 的变化、查询完成、player IDLE、回到初态误差不超过 0.15 rad、关节速度小于 0.08 rad/s 且恢复稳定观察。实际值全部落盘；此处阈值只用于有界仿真核验，不是现场安全保证。
- 没有找到独立取消预设动作服务。`interrupt=true` 是用另一个动作替换，绝不发送任意 motion=0 或新动作来伪装停止；零速度不代表手臂停止。`McAction.SOFT_EMERGENCY_STOP=2` 只有枚举依据，尚未证明具体取消/姿态语义，因此本实现未调用。
- 协调者后续核对[官方过渡说明](https://x2-aimdk.agibot.com/zh-cn/latest/faq/temp_works.html)：`SOFT_EMERGENCY_STOP`自v0.8起不再作为二开运动模式，即使枚举仍在也不能直接调用。官方列出的零力矩/阻尼切换不等于已验证的挥手取消后稳定站立；后续验证另行登记，不扩展此次工程修复。
- 普通脑 HOLD/FINISH 可以等待当前有限挥手自然结束；保护/紧急取消记 UNKNOWN 并继续有限观察，不伪称 CANCELLED。波形仅 `--wave-probe` 开放隔离仿真核验；普通用户执行能力默认禁用。即使自然完成 probe 成功，P2 的挥手紧急取消及可部署能力仍未通过。

## 协调者操作

主机入口在项目目录执行（必须使用新名字）：

```sh
PYTHONPATH=src .venv-train/bin/python -u -m malecns_lab.experiments.x2_skill_host \
  --execute --sim-only --wave-probe \
  --skill HOLD ADVANCE TURN_LEFT TURN_RIGHT WAVE_RIGHT \
  --output data/x2_sim/agibot/data/var/skills-20261023-HHMMSS-xxxxxxxx
```

`HHMMSS` 必须是六位时间，`xxxxxxxx` 是 8–32 位十六进制随机值。主机先验证旧 CPU 容器 `exited`、私有 lighter 停止且无 socket，然后取得本次资源所有权、启动同一 GPU Sim/MC 并等待官方 SD。`ready.json` 表示正在等待可见官方界面的 Reset/Run，不表示动作已可执行。

协调者使用 CUA 在 `http://127.0.0.1:6081/vnc.html?autoconnect=true&resize=scale` 操作官方 Reset、Run，看到仿真运行后，写入该新目录的 `continue.json`，内容只为 `{"command":"continue"}`。此后主机逐一运行选定 probe；任何 probe 未完成就停止后续动作，保留原始失败。每个子目录保存 `events.jsonl` 和 `summary.json`。主机无论成功/异常/超时都尝试让本次 probe 执行 finally，再停止本次私有容器和 lighter，保存 `host_summary.json`。未确认清理不得称资源已关闭。

若只观察左右转，可显式 `--skill TURN_LEFT TURN_RIGHT`；需要单独核验取消路径时登记新的输出目录及 `--cancel-after`（0.1–4 秒），不用失败结果覆写前一次。未因行为失败自动重跑/改阈值。运行预算包括启动和 GUI 等待，UI 等待最长 80 秒且为后续动作保留时间。

源码快照包含旧传输依赖和新核心/ROS runner；`official_assets.json` 记录已配置版本及八个实际官方文件的 SHA256，容器运行前逐项比对。不会修改 SDK、MC、Sim 或旧档案。主机源另存 `host_source.py`。

## 真实检查

1. 初始纯执行与 ROS preview 的 10 项定向检查通过：`10 passed in 0.06s`。
2. 随后按协调者要求新增主机生命周期入口，并补普通结束、主机无副作用预览、实际资产路径和失败仍清理的检查；同时收紧挥手反馈新鲜性并增加新契约转换。因相关代码变化重新运行同一小文件：`13 passed in 0.10s`。
3. 差异空白检查通过。没有运行完整仓库测试、build、训练、仿真或硬件；不重复已通过检查。

独立审查应静态核对控制/归属/资源退出及本报告实际结果，不机械重跑上述检查。仍需协调者真实仿真、取消路径证据和新独立验收，不能把这 13 个离线测试写成 P2 通过。
