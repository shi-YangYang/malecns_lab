# M3 第二次完整脑探针实测

2026-10-23。按[m3-stabilized-contract.md](m3-stabilized-contract.md)固定方案B，在[新独立准入PASS](m3-stabilized-acceptance.md)后只运行一次。结果为**该固定片段的一次联合更新通过**，不是P3/P4行为能力通过。仿真已关闭后才运行，worker已正常退出。

目录：`data/x2_behavior_probe_20261023_r2`。命令：

```sh
PYTHONPATH=src .venv-train/bin/python -u -m malecns_lab.experiments.x2_behavior_training --execute --mode probe-stabilized --output data/x2_behavior_probe_20261023_r2
```

## 关键实测

| 项目 | 实际结果 |
|---|---|
| 原始前向配对 | 3个状态、2帧输入/神经特征/三头输出逐数组dtype/shape/字节完全相同，paired-forward.json all_exact=true |
| 损失 | 12.290372848510742，与首轮相同；loss scale=1 |
| 第一窗末v梯度 | 166700项中165467非零，全有限；L2=15220.6295，max_abs=8586.1035 |
| theta梯度 | 11752项全有限，3427非零；L2=5.13722846633e11，max_abs=4.76142895104e11 |
| 50步边界审计 | 400个登记张量，381个被损失遍历并获得梯度；全部观察到的梯度有限。最终不影响本损失的g/pending无梯度按缺失保存，不伪造成零 |
| Adam准入与更新 | fresh Adam，所有原始梯度可安全平方，实际step=1，更新后Adam状态有限 |
| 实际参数改变 | theta3427项；行为权重13140/偏置5；阶段权重7884/偏置3；参数头按mask为0项 |
| 时间 | worker37.886秒，监督38.192秒，总38.575秒；两输出归档时刻差5.301秒，含前向与写盘，不是纯推理基准 |
| 资源 | 实际预算5279465472字节；采样RSS峰1627668480字节，MPS current1512644352、driver1602945024；统一内存指标不相加 |
| 数据与图 | 仅打开train，未打开validation/locked-test；完整图原始连接计数未变 |

原始梯度、400个边界登记、更新前后参数、源快照和资源CSV均在run内保留。新的policy_id为`cf621475ebe520cb23b05c57b8cc4f9d24c04952e57b1009b59e26d745ba1647`，input_map_id为`6250f324b20a9a28498153347ca90b93b1d6c870d72c0ae53530bac4c446a2b8`；final checkpoint为`4362279ba2c23e3488695249cda696f301a1aba872bb40ead0f16248a5d4ae18`，明确标为probe_only_not_selected_for_deployment。

固定0.1代理梯度使这次更新可执行，但theta梯度幅度仍大，不能外推为所有场景或长期训练都稳定。完整跨窗信用有实测非零证据，但训练估计器已经变化，不声称与原surrogate梯度等价。下一阶段从本轮initial-checkpoint（未做这次单样本更新）建立读出基线，另行登记场景覆盖、更新/时间上限；不把探针候选直接送机器人。
