# Spec023 冻结脑任务能力实测

2026-10-23。一次预登记、有界离线完整图评估；风险 S2。

## 结论

**技术执行完成，任务行为 FAIL。** 5 场景共 15 帧全部输出 FORWARD；正确 6/15（40%），FORWARD 命中 6/6，STOP 命中 0/9。唯一通过场景为远目标保持；模型在近目标和无目标输入下没有产生预期的停止意图。

结果与恒 FORWARD 策略相同，低于恒 STOP 的 9/15（60%）。当前冻结脑加既有纵向意图映射尚未满足本轮 X2 最小任务，不能以已有“能前进并由控制流程停下”的演示证明模型会判断何时停止。

本次没有训练、修改门限或数值后端，没有动作执行或实际停车测量；未因行为失败重跑。评估程序的完成与模型能力通过是两项独立结论。

## 执行与身份

```sh
PYTHONPATH="$PWD/src" .venv-train/bin/python -u -m malecns_lab.experiments.x2_task_eval --execute --output data/x2_task_eval_20261023_r1
```

- 原始目录：`data/x2_task_eval_20261023_r1/`。父监督返回码 0，`status=completed`、`behavior_pass=false`。
- 登记时间：`2026-10-23T04:14:59.304862+00:00`；登记身份 `25b8c611528439c5200a5fef61db8a172cd2f133fd595b43e881d95657e5040f`。
- 任务执行身份：`690d1164d778f90098e3546bbbdde524ef9501bdd09939083fb28558fcc82e95`；冻结后端执行身份 `3f30a7ba3cb6842389f383dde535125729ffdf9c92808f219d3cad3790a25890`。
- policy：`a04eb5f35eb591fc000d632964ed52b70427928b1324dbb28741f2c441680ca4`；checkpoint：`30b09f6a7d4a2a879472648fa49b504d918e8a1225fe797532327bb937198f63`。
- 同一完整 MaleCNS：166700 节点、25582938 非零边、124177617 原始连接计数。固定 checkpoint 为 `data/neural_training_gpu_recovery_20261019_r1/segments/000042/windows/000160/checkpoint/`。54 个登记历史源码、当前包装、协议、图与参数来源均随 run 留存。

## 逐场景结果

F=FORWARD，S=STOP；期望、顺序与严格 `raw vx > 0.03` 规则在首次完整图运行前固定。

| 场景 | 三帧距离（m） | 期望 | 实际 | 正确 | 场景通过 |
|---|---|---|---|---|---|
| 远目标保持 | 2.0, 2.0, 2.0 | F F F | F F F | 3/3 | PASS |
| 近目标保持 | 0.5, 0.5, 0.5 | S S S | F F F | 0/3 | FAIL |
| 始终无目标 | 无, 无, 无 | S S S | F F F | 0/3 | FAIL |
| 目标接近 | 2.0, 1.5, 0.5 | F F S | F F F | 2/3 | FAIL |
| 目标消失 | 2.0, 无, 无 | F S S | F F F | 1/3 | FAIL |

混淆计数：期望 F→实际 F 为 6，期望 F→实际 S 为 0；期望 S→实际 F 为 9，期望 S→实际 S 为 0。目标接近至 0.5m 的帧和消失后的两帧均继续给出前进意图。

## 原始三轴输出

下表保留原始 `[vx, vz, wz]`（仅展示六位小数；完整精度、logits 与 features 在原始 JSON/NPZ）。仅 vx 用于本轮意图评分，三轴均未发送至机器人。

| 场景 | 帧 | vx | vz | wz | 期望 | 实际 |
|---|---:|---:|---:|---:|---|---|
| far_hold | 1 | 0.224106 | 0.106927 | 0.213602 | FORWARD | FORWARD |
| far_hold | 2 | 0.280164 | 0.147226 | 0.332987 | FORWARD | FORWARD |
| far_hold | 3 | 0.256655 | 0.140807 | 0.338380 | FORWARD | FORWARD |
| near_hold | 1 | 0.223794 | 0.106763 | 0.213189 | STOP | FORWARD |
| near_hold | 2 | 0.279181 | 0.144158 | 0.325555 | STOP | FORWARD |
| near_hold | 3 | 0.266461 | 0.146625 | 0.333110 | STOP | FORWARD |
| absent_hold | 1 | 0.224151 | 0.106969 | 0.213626 | STOP | FORWARD |
| absent_hold | 2 | 0.282310 | 0.150274 | 0.338421 | STOP | FORWARD |
| absent_hold | 3 | 0.260059 | 0.142008 | 0.321421 | STOP | FORWARD |
| approach | 1 | 0.224106 | 0.106927 | 0.213602 | FORWARD | FORWARD |
| approach | 2 | 0.280164 | 0.147227 | 0.332988 | FORWARD | FORWARD |
| approach | 3 | 0.254595 | 0.139226 | 0.334996 | STOP | FORWARD |
| disappearance | 1 | 0.224106 | 0.106927 | 0.213602 | FORWARD | FORWARD |
| disappearance | 2 | 0.280143 | 0.147202 | 0.332943 | STOP | FORWARD |
| disappearance | 3 | 0.257984 | 0.141514 | 0.336924 | STOP | FORWARD |

vx 范围约 0.223794–0.282310，均高于固定 0.03 门限。各场景原始值有差异，但未形成任务需要的前进/停止区分；不能把数值变化或脉冲活动等同于正确决策。

## 状态、遥测与冻结证据

- 5 场景分别调用原 initial_state，600 步预热后的完整状态摘要一致；各场景时钟为 600→1100→1600→2100，场景内递归状态连续。完整运行合计 10500 神经步。
- 保存 10 份完整初/末状态 NPZ、20 个状态边界 JSON、15 份输入 NPY 与 15 份输出 NPZ；帧记录包含像素/数组/文件 hash、前后状态及执行身份。
- 5 个独立遥测流均为 42/42 批，总 210/210 批；complete=true、丢批 0、error=null、partial_batch_steps=0。每批 50 神经步，沿用真实 22 群组和 32 个选样输出神经元脉冲，不是全脑逐步记录。
- `parameters.json` 的 theta、readout_weight、readout_bias 前后 hash 一致；`unchanged=true`，未构造 optimizer，推理关闭 autograd。
- 原始连接计数前后 SHA256 均为 `4ce30eec9ecc64c7bb2723f9dee084b40cea08793375f40ab71cacc2592b0565`。
- 关键记录为 `registration.json`、`execution_identity.json`、`actual_policy.json`、`parameters.json`、`worker_result.json`、`scores.json`、`summary.json` 及 `scenarios/*/`；独立验收见 [acceptance.md](acceptance.md)。

## 实际资源

- 含准备的总墙钟 55.143s，父监督 54.916s，均在 240s 上限内。15 帧 forward 平均 2.371s，范围 2.336–2.421s。
- 有效 RSS 预算 5.461GiB；监督采样峰值 RSS 1.539GiB，采样最低系统可用 4.316GiB，高于 1GiB 预留。
- MPS 545 次采样，已分配峰值 1.012GiB、driver 峰值 1.087GiB；采样无错误。统一内存 RSS 与 driver 值不能相加，采样峰值不是连续瞬时峰值保证。
- 运行前只读预检 GPU VM 和旧 CPU 仿真均停止，本阶段没有启动仿真；完整脑子进程已正常退出。没有改变默认 Docker context、Noria、SDK 或其他服务。

## 解释边界与后续建议

这是居中虚拟红方标的五条固定、相互相关的短序列；不是仿真相机或真实相机，不具备统计泛化证据。每帧 50ms 是神经时间，不是实际控制延迟；每场景只有三帧，不能据此断言更长输入永远不能停止。

历史 checkpoint 来自既有无人机三轴离线模仿训练，本轮标签是新拟定 X2 任务。FAIL 不等于历史训练完全无效，也未定位问题属于视觉编码、内部动力学、读出或任务标签不匹配。

下一阶段建议先核对旧训练目标与 X2 输入/读出需求，制定包含前进、近距停止、缺目标停止的专项学习与独立留出验证协议，再按协议开展有界训练。评估集不可直接充当训练集或调门限的依据；任务决策达标后再恢复仿真动作验收。本次仅提出后续方向，没有启动下一阶段。
