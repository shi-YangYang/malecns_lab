# Implementation Report — P3d 有状态神经闭环评估

## Summary

实现 Ready [P3d 协议](p3d-closed-loop-evaluation.md)的独立评估子系统。生产准入严格要求 joint 与独立 frozen_brain 的真实最终 160 窗封存链；本批没有启动完整图评估、训练、实体设备或任何服务操作。既有 29 个训练绑定源码保持不变。

新增工具支持最终参数固定、五个条件的冻结顺序、任意帧神经状态恢复、执行动作的廉价环境重放、完整 DN 到动作的父复算和逐场景独立评分。原始资源失败与数值/协议失败分开；物理失败作为正式行为终局保留，不能用段完成或部分 MAE 冒充成功。

## Files Changed

- `src/cnsbridge/learning/evaluate_snn.py`：register / run / summary；共享训练 OS 锁；显式 1–40 帧段；资源监督与父结果。
- `src/cnsbridge/learning/evaluation_contract.py`：两种最终训练准入、全部来源绑定、实际参数数值身份、acute 原始 theta 派生、固定条件和场景。
- `src/cnsbridge/learning/evaluation_checkpoint.py`：无 Adam 的完整状态格式；frame 0、奇数及终局；Torch-free 读取、实际数组/时钟校验、worker 内恢复。
- `src/cnsbridge/learning/evaluation_rollout.py`：独立读出转换、检测与监督、环境重放、评分；无神经前缀重算。
- `src/cnsbridge/learning/evaluation_evidence.py`：唯一封存链、恢复前实际值、完整轨迹和父复核、失败诊断、val 封存门槛和固定场景配对汇总。
- `src/cnsbridge/learning/evaluation_worker.py`：生产完整图参数加载、先干预后场景初始化、只接 BGR 与 PolicyState 的 no_grad 前向、十帧块封存。
- `tests/test_evaluation_snn.py`：只使用明确数值电路和独立 OS 子进程的定向检查。
- `docs/EVALUATION_SNN.md`：准入、显式命令、首次 1+1 帧实际门槛、结果解释和失败续段规则。
- 本实施报告。

## Important Decisions

- 没有修改新协议或训练核心。`policy_id` 不含训练后数值，因此 theta/读出摘要形成独立 `parameter_set_id`，在条件、恢复、块和状态档逐层绑定。
- 原始 theta 来自已封存 r2 数组；acute 干预发生在 `initial_state()` 前，不假定原始数组恒为零。
- 场景状态恢复支持奇数帧，不复用训练档的偶数帧/正 Adam 步约束；基线状态档没有虚构神经数组。
- 环境只通过重放已验证动作恢复；父逐帧复核图像、检测、介入、truth 和 next_truth。神经前向不接这些信息。
- 父对全部保存 DN 的标准化/线性层/tanh/动作尺度独立复算，不能据此声称独立验证了生物神经活动的真实性。
- 数值/协议失败禁止自动恢复；summary 保留诊断类别、不给无法复核前缀生成指标。资源失败可保留采样覆盖的封存前缀，原始 supervisor 结果仍 failed。
- 首两帧代码限制为两个独立一帧调用；其后的新独立实际验收属于研究工作流，不伪造为程序已完成的门槛。

## Validation

- Risk Level: **S3**，涉及新的状态持久化、段监督和研究证据链。范围限新增评估子系统及直接接口；未运行旧完整套件、build 或全量 lint。
- 根协调者先确认生产训练 session 9089 已退出、无模型进程，随后实施者执行：
  `PYTHONPATH=src .venv-train/bin/python -m pytest tests/test_evaluation_snn.py -q`
  session **76640** 已收取 exit 1：**24 passed、1 failed，3.47 秒**。
- 唯一失败来自数值测试假设：完全遮挡的小电路原始动作也恰为零，无法观察监督是否抑制非零输出。仅将该 fixture 的读出偏置在套件登记前显式设为 0.1；业务实现未因此改变。
- 仅重跑失败用例：
  `PYTHONPATH=src .venv-train/bin/python -m pytest tests/test_evaluation_snn.py::test_pixel_only_forward_action_changes_observation_and_lost_target_advances -q`
  exit **0**：**1 passed，0.49 秒**。其余已过 24 项未重复执行；25 项当前均有通过证据。
- 覆盖两个真实 OS 进程的 1 帧＋3 帧恢复，与同参数连续 4 帧的全部神经/感觉状态、环境轨迹和指标精确一致；不是同进程假恢复。
- 覆盖 frame 0/1/80、非法相位/计数/shape/非有限数组/布尔游标、同结构错参数、acute 时序、BGR-only 接口、动作改变图像、丢目标仍推进神经、基线无 DN、独立读出/环境/评分拒假、坏父链、资源失败有效前缀、数值失败不自动续跑、val/test 门槛、无生产 fixture 回退、物理提前终止的 80 帧分母、空失败与假成功、显式配额及错恢复状态。
- 实际 `git status --short` 检查显示业务改动仅六个新增评估模块；没有任何既有 29 个绑定源码的修改条目。现存 `p3c-continuation-20261019.md` 修改属于根协调者的训练进展记录，实施者未编辑。
- 新 OS 进程确认导入父入口不会导入 Torch。仅有既有 PyTorch Sparse CSR beta 警告，无设备或完整图执行。
- 测试结束已通知根协调者恢复训练；实施者无存活测试/模型作业，未触碰 Noria/Docker。

## Known Limitations

- 两种真实最终 160 窗训练链尚未同时存在，本批没有实际套件登记或完整图闭环成绩；不关闭 AC3–AC6。
- 数值 fixture 只验证软件契约，不是生产图后备模型，也不证明目标跟随成功。
- 600 秒下最多 40 帧的生产成本仍待实际测量；保留资源监督及保守封存后采样检查，未放宽时间偏移边界。
- 场景为同类红色标记的简化速度级仿真；拓扑贡献、实时部署和实飞属于后续独立子批。

## Remaining Questions

无阻止本批提交独立软件验收的已知问题。请由新的独立验收者检查协议覆盖、来源与证据边界；本报告不代替最终验收。实施者未 commit，提交由协调者完成。
