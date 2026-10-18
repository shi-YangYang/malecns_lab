# Acceptance — P3a 教师序列与策略参数存档

## Result

**FAIL**。本报告由未参与实施的新独立验收者完成；两个具体缺口违反 [P3a](p3a-data-checkpoint.md) 的训练来源隔离及轻量存档语义校验要求。未修改业务实现。

## Spec Coverage

- A：**未通过**。采集路径确实保留全部帧、独立真值教师标签、行为/执行动作及无效帧掩码；数组、时间轴、登记 metadata 和字节 hash 有校验。但是实际轨迹/图像与登记场景之间缺少验证，可通过替换实际数据并重封存绕过 train split。
- B：**未通过**。参数集合、精度、有限值、固定 rho、完整目标身份比较及先校验后拷贝的恢复顺序符合预期；生产入口仍使用 `NeuralPolicy.from_reference`，数值角色不会进入该入口，未自动初始化状态。但是轻量读取仍接受与当前源码不兼容的动力学声明。
- C：本轮独立验收已执行，须返工后由新的验收者复核；尚不满足提交完成条件。
- 文档明确区分部署参数与优化器/隐藏状态恢复，也没有把软件检查称为真实训练或实飞成果。

## Validation

- Risk Level：**S3**，范围仅限本批持久化和训练数据来源。
- 阅读了本批两模块、两份定向测试、两份使用说明、实施报告，以及被调用的策略、环境、场景与 rollout 接口。
- 实施者记录的 `49 passed, 1.20 s` 已审阅，未重复执行该测试命令。
- 针对代码审查发现的两项具体风险，执行了一次临时 Python 复现命令（`PYTHONPATH=src:tests .venv-train/bin/python`）。使用 0.3 s 场景 fixture 和已有 5 节点数值电路，仅临时目录；未加载完整图、运行训练实验、触及设备或服务。
- 实际输出：

```text
VAL_DATA_ACCEPTED_AS_TRAIN train-sine-combined True
INCOMPATIBLE_GAIN_METADATA_ACCEPTED exp(theta); UNBOUNDED incompatible dynamics
```

## Issues

### P1 — 实际 val 数据可冒充 train 序列

位置：`src/cnsbridge/learning/neural_data.py:199`，尤其第 206 行从记录自行提供的 `truth` 计算标签。当前只核对记录内部关系；没有验证图像、truth、next_truth 与登记场景及连续状态相符。

最小复现过程：

1. 将四个登记训练场景缩短到 0.3 s，并显式使用 `numerical_test_only` 数据角色；正常收集一次。
2. 用相同相机和 dt 运行同样缩短到 0.3 s 的 `val-triangle-combined`，行为策略仍为既有 `visual_teacher`。
3. 将该 val rollout 的实际图像、记录、指标和对应监督数组放入 `train-sine-combined` 目录；保留原训练场景的 summary metadata 和 manifest 场景定义。
4. 重新计算三个文件 hash 和 dataset ID。`read_neural_data` 成功返回 `train-sine-combined`，而返回记录的第一帧目标位置与 val 场景完全相同。

影响：调用方只改变来源声明并重算 hash，就能把非训练场景的实际观测/标签混入训练，违反教师数据契约第 6 项。当前测试只改 manifest split/定义，未覆盖实际数据替换。

### P2 — 轻量参数读取只校验部分模型语义

位置：`src/cnsbridge/learning/policy_checkpoint.py:121`—139 的模型校验。`model.gain` 没有与当前模型契约比较，嵌套 canonical hash 只能证明内部摘要一致，不能证明描述与已绑定源码一致。

最小复现过程：保存正常数值 fixture 策略；将 `metadata.policy.model.gain` 改为 `exp(theta); UNBOUNDED incompatible dynamics`；依次重算 `model_id`、`policy_id`、`checkpoint_id`，不改参数文件或源码 SHA。`read_policy_checkpoint` 仍成功返回该声明。它与当前 `1 + 0.5*tanh(theta)` 的有界增益语义矛盾。

影响：不构造图的轻量读取会把不兼容参数解释声明作为已验证存档返回，违反参数契约第 3 项。恢复到实际策略时的完整 identity 比较仍会拒绝此存档；本次没有证明或声称发生了错误参数恢复或部分写入。

## Regression Risks

返工必须维持只读加载、完整场景时间顺序、无效帧保留和教师/行为分离；不得为校验来源调用神经模型或把来源 hash 当作语义证明。checkpoint 必须保持完整图生产入口、固定背景和恢复失败不变性，不得通过放宽身份解决兼容性。

## Required Rework

1. 用登记场景、相机和环境时间参数建立可验证的实际来源关系。可通过现有轻量环境按记录执行动作确定性推进，核对每帧 pixels/truth/next_truth 及连续时间；不能仅核对调用方提供的 truth 与调用方提供的标签相互一致。新增实际 val 数据替换后重封存的拒绝检查。
2. 补齐轻量读取对固定模型语义的校验，至少覆盖已复现的 gain 声明及相应严格字段契约；对结构 hash 自洽但语义不兼容的 metadata 应直接拒绝。保留原有恢复全部预验证的顺序。
3. 针对以上具体改动执行相关定向检查，由新的独立验收者复核。无需重复完整图实验或未受影响的旧套件。
