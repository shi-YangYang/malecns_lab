# P3c2连续训练进度 — 2026-10-19

冻结run为`data/neural_training_joint_20261019_r1`，身份/模型/数据/训练规则沿用[首次实测](p3c-run-20261019-r1.md)。首两窗已有[独立实际PASS](acceptance-p3c-run-r1.md)；此记录按真实后续段追加，不重写既有运行。当前不是完整epoch或学习收益结论。

## Segment 3：首次单段四窗

在首两窗独立验收并提交`f34c56d`后执行一次：

```bash
PYTHONPATH=src .venv-train/bin/python -m cnsbridge.learning.train_snn \
  --run data/neural_training_joint_20261019_r1 --windows 4
```

终端session75106已收取exit0。原监督与worker均completed；父复核返回`registered_windows_verified`，累计6窗。新段从combined frame4继续至frame12，窗口3–6各两帧，实际Adam步3–6；神经/感觉step连续为2600→3600→4600→5600→6600。没有新增场景预处理。原图有序计数摘要前后保持`4ce30eec9ecc64c7bb2723f9dee084b40cea08793375f40ab71cacc2592b0565`。

| 窗口 | 更新前动作MSE | 本窗theta改变数 | 原始theta梯度L2 | 两帧前向s | 反向/裁剪s |
| --- | --- | --- | --- | --- | --- |
| 3 | 0.00660218 | 4,814 | 8.96269e29 | 18.887433 | 76.826822 |
| 4 | 0.12427571 | 4,852 | 4.20599e30 | 19.350813 | 79.791428 |
| 5 | 0.14176721 | 4,870 | 1.74455e29 | 19.453579 | 81.114935 |
| 6 | 0.04076099 | 4,894 | 2.42297e29 | 19.429825 | 82.144452 |

所有更新、保存数组和状态经过父数值/来源链复核。损失波动全部保留，窗口输入和神经状态各不相同，不能据此直接推断训练收益。第四个新窗口DN特征范围约−576.98至−45.62mV；这仍是工程模型状态，不是已验证生理测量。

资源登记：启动available=5,936,644,096 bytes，额外预留1,073,741,824 bytes，实际RSS上限4,862,902,272 bytes，600秒/0.1秒监督。实际3592个样本，采样峰值3,193,143,296 bytes（2.973846GiB），最低available=3,877,142,528 bytes；监督墙钟401.859676秒，worker墙钟400.927180秒。未触发停止条件，未操作Noria/Docker。四窗约402秒是这一次实际观测，不保证后续场景成本相同。

段ID为`7556780de150bcadbbfac75542960b2d7f1eea1ae96099252a528762ffe39af1`。最后完整checkpoint ID为`f85f5376019778d40a50db68f0ab10b58b92d00ece21d51616dd2637a2cb7be9`，位于`segments/000003/windows/000006/checkpoint`。游标next_frame=12、completed_windows=6、optimizer_steps=6；epoch_complete=false，joint仍余154窗，frozen尚未开始。没有活跃模型进程。

该首次多窗段已获新的[独立实际PASS](acceptance-p3c-four-windows.md)，完整报告已读取处理，无返工项。独立复算一阶矩和参数全部exact；readout.weight二阶矩在step4/6有4.44e−16/5.68e−14最大差，满足冻结容差，未宣称所有矩逐位一致。后续保持同run，从frame12按1–4窗显式推进，不重跑已完成窗口、缩图或放宽阈值。完整训练、跨场景实际边界、独立对照与闭环仍待完成。
