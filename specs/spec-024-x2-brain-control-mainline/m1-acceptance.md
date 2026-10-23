# M1 独立工程验收

日期：2026-10-23。结论：**P1 FAIL，等待限定返工**。本报告只验收新增契约、编码、数据及已存在的 M2 消费接口，不评价模型学习、实际技能或真机表现。审查者未参与 M1 实施，未修改业务源码。

风险 S3：新序列化和控制交互契约。实施者记录的 16 项定向检查已通过，本次没有机械重跑。只对实际发现的协议接缝执行了两条纯对象最小复现命令，没有加载完整图、Torch、ROS、VM、仿真或设备，没有 commit/push。

## 阻断项

### R1：空闲 HOLD 的生命周期无法通过门禁

`control/x2_behavior.py:274-277` 仅为 START 保留在途项；HOLD 必须为 FINISH，因此空闲 HOLD 被 admit 接受后没有在途项。`execution_event()` 的无在途分支（329 行）只接受终态。

实际消费者 `control/x2_skills.py:272-276` 对合法空闲 HOLD 依次记录 ACCEPTED、STARTED、COMPLETED。首个事件就被门禁以 `Event without active skill` 拒绝。正常待命及 END_INTERACTION 回待命路径不能按目前接口完成事件归档。

最小复现结果：

```text
HOLD FINISH admission= True
ACCEPTED REJECTED: Event without active skill
```

限定返工：统一 HOLD 生命周期及归属，允许真实接受/站稳/完成事实通过合法状态转换，保持 HOLD/FINISH 不自动证明停止；若是结束已有技能，也必须保留原技能与结束决策的可追溯关系。不能靠丢弃真实事件解决接口不一致。

### R2：挥手的两次 ACCEPTED 与门禁状态机冲突

`control/x2_skills.py:272` 首次发出 `ACCEPTED/skill_admitted`，官方预设服务回复后 `preset_ack()`（203 行）再次发出 `ACCEPTED/preset_ack`。门禁 `control/x2_behavior.py:317` 不允许 ACCEPTED → ACCEPTED，所以实际服务 ACK 到达时被拒绝，后续开始/完成证据不能连续消费。

最小复现结果：

```text
WAVE_RIGHT START admission= True
ACCEPTED accepted
ACCEPTED REJECTED: Invalid execution transition; ACK is not completion
```

限定返工：明确本地 admission 与官方服务接受的不同事实，统一执行器和协议对 ACK 的表达、去重与合法转移；不把本地门禁接受当作官方接受，不把 ACK 当作实际开始或完成。

## 已核实的额外语义差异

UNKNOWN 目前在门禁中属于 TERMINAL，清除 active 后允许下一 START，即使 `stop_evidence.stop_confirmed=False`。针对这一疑点的第二条纯对象复现得到：

```text
after UNKNOWN active: None
next START admitted: True
executor while unresolved accepts next START: False
executor after terminal accepts next START: False
executor failure latch: ['wave_cancel_unverified:test']
published commands: []
```

当前 M2 有独立拒绝：`_unknown()` 锁存 `motion.failures`，未终态时拒绝另起技能，终态后 `submit()` 的 failures 检查仍拒绝。因此本审查**没有发现当前 M2 会因此并发发出 MC 命令**，不把门禁差异夸大为已发生的不安全动作。限定返工应同时冻结 UNKNOWN 的未确认停止/恢复语义，使上层不会把门禁空闲误认作可重新执行；不能只靠状态名称“终态”推断机器人已停稳。

## 通过的静态边界与已有检查证据

- 新 Observation、Decision、ExecutionEvent 分开；原始 logits 与解码行为/阶段、参数 sigmoid 一致性有检查，拒绝不改写神经原始输出。身份、单调序列、观测过期/未来时间以及新请求使旧推理失效的路径存在。
- 任务没有直接调用 MC 的路径。teacher 仅在离线数据模块生成标签；感觉编码只接 Observation，没有教师目标、未来帧或动作读出输入。正常 END_INTERACTION 与执行保护分开。
- 固定 57 通道覆盖任务、反馈、身体量、视觉及有效性；无相机时明确缺失。编码生产入口绑定现有 PolicyNeuronMap，检查完整参考角色、166700 节点、3377 个输入和 1314 个 DN，以及数组/身份一致、输入输出分离；码本覆盖全部输入。
- 数据按完整 episode、分离参数族划分 200/50/100；标签与观测分离。train/validation 读取只打开指定文件，locked-test 需显式命名。评分声明是教师序列一致性，未冒称实际任务完成率。
- 实施者 16 项定向检查的结果来自 `m1-implementation.md`；本次审查了相应测试源码，没有再次执行。
- `git diff --name-only -- src/malecns_lab tests` 没有已跟踪文件差异；当前 M1 为新增文件，旧数值内核、旧 schema 和旧源码未被此次实现覆盖。此项不等价于扫描或重校验全部本地科研资产。

## 限制与恢复条件

真实完整图映射和 MPS 输入刺激没有在 M1 运行验证，仍由 M3 生产入口核验；这里的静态绑定审查不能替代该证据。模拟反馈样本不证明真实反馈覆盖或学习通过。P2—P6 均不在本报告通过范围。

由新的实施者只返工上述契约/事件接缝，记录最相关定向检查；随后交另一位新的独立验收者。修复前不宣称 P1 完成，也不把本次 FAIL 扩大为重写已有脑内核或重新执行已通过的旧项目检查。
