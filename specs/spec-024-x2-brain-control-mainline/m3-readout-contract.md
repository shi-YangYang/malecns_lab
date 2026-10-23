# M3 多任务读出基线登记

状态：准备就绪，待独立核对后由协调者串行运行。2026-10-23。第二次完整脑探针已完成一次有限联合更新；本阶段建立冻结内部参数、仅学习新三头的对照，不用该单样本更新充当专项训练成果。

## 固定样本与初始化

- 使用相同seed2401的新schema数据；训练拆分按既有task-round-robin选择前50个完整episode（每任务10），416帧。验证使用全部50个完整episode（每任务10），402帧。覆盖rejected、missing_feedback、waiting_stand、already_aligned、repeat_request、switch_request、slow_feedback和normal全部8类；normal每拆分15个，其余每类5个。
- 选择依据是任务/情境覆盖与成本，不依据模型得分；不使用默认5条，因为它们全部是rejected情境。锁定100条测试不开启，不用于选择、调参或续训。
- 初始化固定`data/x2_behavior_probe_20261023_r2/initial-checkpoint`，checkpoint ID `394e39ec8eed0b622712fe2ffeca20d247e1ed437a9e6945479716eb007f3959`。它是精确旧theta迁移+seed2401新头，未包含探针那次单样本更新。新反向身份gain0.1沿checkpoint继承；本阶段theta冻结，因此不发生内部反向更新。
- 完整166700节点/25582938非零边，3377输入、1314DN及2628维特征不变。每episode重置，600步预热；内部跨决策状态持续，每帧500步，无人为替换或降采样。训练和验证的实际完整脑特征只计算一次并归档，后续仅复用本次冻结脑特征训练头。

## 优化与选择

- 新Adam，lr0.001、betas0.9/0.999、eps1e-8、weight_decay0、amsgrad=false；三个头最多200次更新。每次按50条固定顺序取一完整episode求平均损失，最多4轮，不因得分继续增加。
- 损失保持行为CE+阶段CE+0.1×有效连续动作参数MSE；没有任务/标签直通头部或推理教师。
- 每次更新后只用这50条验证episode的缓存神经特征计算验证loss；选择最低loss候选，相同loss取较早更新。记录全部梯度审计、实际改变、验证原始预测与完整序列/行为/阶段指标。
- theta必须逐参数完全不变；任何非有限、资源超限或身份改变立即保留失败并结束，不清洗数值、不自动调参/重启。

## 时间与资源

- 单次总墙钟上限5400秒，最多200次更新。第二次探针两输出归档间隔5.301秒（含有梯度前向/写盘），818个决策帧按此量级约4336秒，加100次预热与头优化/归档预留后定5400秒；无梯度缓存可能更快，但当前不把它当已测吞吐。
- 同时仅一个完整脑，仿真关闭，沿用独占锁。CPU4线程/interop1；启动重测可用内存，RSS/MPS预算min(6GiB, 可用内存−1GiB)，至少1GiB系统预留。采样RSS与MPS driver分别报告，不相加。
- run新目录`data/x2_behavior_readout_20261023_r1`，不覆盖。准备时可用磁盘约571GiB；状态、特征、source、checkpoint和资源记录按现有归档保存，不删除旧科研资产。
- 本登记授权一次有界读出基线；后续联合训练预算需依据本轮真实吞吐与验证结果另行冻结，不在运行中扩大5400秒或200次上限。资源运行结束即关闭。

```sh
PYTHONPATH=src .venv-train/bin/python -u -m malecns_lab.experiments.x2_behavior_training --execute --mode readout --checkpoint data/x2_behavior_probe_20261023_r2/initial-checkpoint --train-episodes 50 --validation-episodes 50 --max-updates 200 --wall-seconds 5400 --output data/x2_behavior_readout_20261023_r1
```

## 结论边界

这些是明确标注的离线工程教师与模拟执行反馈，不是真实X2轨迹。验证拟合只用于候选与成本判断，不能替代P3真实反馈、P4干预、P5桌面动作或P6真机。现有数据的情境名称也不自动证明所有真实事件边沿均有覆盖；真实同任务新请求、步态转换瞬态、连续状态与执行拒绝等差异需在后续闭环数据中核验。若读出基线不足，保留结果分析原因，不用无限续训或打开锁定测试寻找配置。
