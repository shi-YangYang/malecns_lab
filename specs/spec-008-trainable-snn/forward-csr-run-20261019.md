# 共享 SciPy 前向的完整图实际对照

日期：2026-10-19北京时间。软件提交 `13716c6` 后按[冻结子规格](forward-csr-cache.md)串行运行P1 r5及配对诊断r2，两者均已正常退出0并通过各自父进程证据校验；实际独立验收见 acceptance-forward-csr-actual.md，不以进程退出单独认定通过。

## P1 r5：完整图一次内部更新

```sh
PYTHONPATH=src .venv-train/bin/python -m cnsbridge.learning.probe \
  --graph data/malecns_v2/full_reference.npz \
  --out data/trainable_probe_20261019_p1_r5
```

登记UTC `2026-10-18T19:32:30.960880+00:00`，registration ID `d6695f8356d4c327237acb64894b8e1d22b0ecaca7f60278f8afb65fd499f6ca`，protocol ID保持 `c93a1fcd3fe90e4a0bda9a258f4c50b3bd5dd75956358b209ff08ba1ecd058bb`。新model ID `a4266ff08805c46141aac8f5d36dd2e494fdfb8f67fc167b41d354785ec45cfb`，新checkpoint ID `57e3e0cbff60ebd8b2893508e17f4e809d59c384bf124435444e118fcec5b885`。身份变化来自后端和源码，不绕过严格检查。

完整166,700节点、25,582,938边、124,177,617突触计数，原图摘要 `4ce30eec9ecc64c7bb2723f9dee084b40cea08793375f40ab71cacc2592b0565`；200步/20ms、CPU float32、4线程、同样R1–R6输入与L1目标、Adam一次更新，无动作读出。

| 指标 | r4 Torch前向 | r5 SciPy前向 |
|---|---:|---:|
| 总墙钟 | 83.690591 s | 32.391075 s |
| 更新前前向 | 15.720709 s | 3.678343 s |
| 更新后前向 | 16.509187 s | 3.847775 s |
| 归零恢复前向 | 16.653713 s | 3.897782 s |
| checkpoint重载前向 | 16.630378 s | 3.877232 s |
| 反向 | 13.786124 s | 13.443898 s |
| 采样峰值RSS | 1,873,248,256 bytes | 1,881,767,936 bytes |

r5采样峰值约1.753GiB。启动available 5,789,122,560 bytes，保留1GiB后RSS预算4,715,380,736 bytes；保持6GiB封顶、600s、0.1s监督。父进程返回 `passed / numeric_internal_update_verified`，正常退出0。

协调者直接比较已保存的baseline.npz、responses.npz、update.npz全部非运行身份字段，原协议与图身份一致；所有电位、梯度、theta和gain逐元素精确相同，最大差0。仍只R1–R6组变化，809个L1电位改变、最大0.1765289306640625mV，损失163.01246098981304→161.55835529495025；归零恢复、checkpoint重载和原始图检查通过。派生比较保存在r5的comparison_r4.json，不替代绑定的原始证据。

## 配对诊断 r2

沿用原协议JSON内容及全部两模式七条件，在新目录运行：

```sh
PYTHONPATH=src .venv-train/bin/python -m cnsbridge.diagnostics.observability \
  --graph data/malecns_v2/full_reference.npz \
  --protocol data/observability_20261019_r1.protocol.json \
  --out data/observability_20261019_r2
```

登记UTC `2026-10-18T19:33:19.642316+00:00`，registration ID `d9c814338eba6eb90ef6d5f1de0ff95b27250a15aefd3ab443474404be54949b`，protocol ID保持 `3b35bdf6e99d443b5d2f928c086e560fd0f29ddd79127e3cf3bece13d82e8726`。两模式、全部七条件、60/20/10ms窗口、5400神经步及选取顺序均不变。

| 指标 | r1 Torch前向 | r2 SciPy前向 |
|---|---:|---:|
| 总墙钟 | 468.513886 s | 108.558011 s |
| 采样峰值RSS | 1,610,334,208 bytes | 1,617,035,264 bytes |

r2采样峰值约1.506GiB，启动available 6,108,037,120 bytes，扣除1GiB后预算5,034,295,296 bytes。正常退出0，父校验返回 `protocol_complete`，没有触及资源门槛。

协调者直接比较两次全部22个NPZ的465个可比字段：输入/输出选择顺序、全图基态v/g/不应期/18槽pending、参数、全部1314个DN逐桶脉冲/电位/g、L1–L5逐桶信号、全部166,700节点分相脉冲、输入事件/累计数、驱动相位/时间、派生差异数组，全部逐元素精确相同。无数值字段超过预登记容差。派生证据在r2的comparison_r1.json。

运行ID、模型ID、包含模型ID的起止状态哈希及实际墙钟不作为跨版本相等条件，已明确列为排除字段；各自运行内的身份、分支状态和摘要由既有证据链核对。没有保存的每0.1ms全节点轨迹不能由上述比较声称逐位相同；结论限定为全部已保存数值与语义字段一致。

因此此前观察边界保持：零背景本窗DN无差异，固定8mV背景有刺激相关电位差异，差异仍在撤除后才出现；所有条件逐细胞30ms总脉冲数相同。提速没有改变这些已记录结果，也没有增加新的持续刺激响应或动作学习证据。

## 解释与边界

P1单次总耗时比约2.58，前向约4.27；反向仍为原Torch完整伴随，耗时基本相近。配对诊断单次总耗时比约4.32。这是同机非隔离对照，不是稳定加速保证。新前向仍约3.9s/20ms神经窗，不能解释为实时性。软件检查、完整图数值更新、视觉通路响应和三轴任务学习是不同证据；目前未因此完成目标跟随、拓扑贡献或实飞。

没有修改或裁剪完整数据、Noria/Docker与硬件；不重复已通过的46项数值及37项身份检查，也不重跑旧实验。实际验收须核对原始产物、源码及资源记录，而非只信任结果状态。
