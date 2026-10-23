# M3 冻结读出独立恢复运行登记

2026-10-23，最终独立工程验收PASS（m3-cache-recovery-final-acceptance.md）；协调者开始执行以下唯一登记命令。新目录固定为 `data/x2_behavior_readout_20261023_r2`，不能覆盖。原r1继续保留wall_limit失败。

参数沿m3-cache-recovery-contract.md：从r2探针initial-checkpoint（ID 394e39ec8eed0b622712fe2ffeca20d247e1ed437a9e6945479716eb007f3959）恢复全部七组参数，fresh Adam，theta冻结；显式复用读出r1的完整缓存，50train/50validation，固定200更新，seed2401，每步402帧验证，最小验证loss、最早平局。loss/LR/优化器与原登记相同，gain0.1继承原checkpoint。没有locked-test、联合训练或实际控制。

唯一执行命令：

```sh
PYTHONPATH=src .venv-train/bin/python -u -m malecns_lab.experiments.x2_behavior_training \
  --execute --mode readout \
  --feature-cache data/x2_behavior_readout_20261023_r1 \
  --checkpoint data/x2_behavior_probe_20261023_r2/initial-checkpoint \
  --train-episodes 50 --validation-episodes 50 --max-updates 200 --wall-seconds 600 \
  --output data/x2_behavior_readout_20261023_r2
```

600秒包括父准入、worker复核、完整图身份/参数恢复、200次输出头更新与封存。禁止brain forward/initial_state/step/warmup；实际结果须所有神经调用计数0、200更新/200history、theta逐元素不变、正式selection与validation-scores。特征及数据均从worker本次复验的同一字节取得。

资源由启动入口现场测量并登记：CPU4/interop1、MPS无CPU fallback、RSS/MPS min(6GiB,available−1GiB)，至少1GiB系统预留、单完整脑锁。仿真/私有VM保持关闭；不动Noria、旧CPU容器或Docker context。来源快照与cache_manifest按实际启动时封存，不手改旧run哈希。

失败保留，不自动增加步数/预算或选择原r1诊断候选。结束后记录真实状态、耗时、资源峰值、选择结果和证据限制；成功只表示冻结读出实验执行完成，不自动通过P3/P4/P5。联合训练仍待独立冻结方案。
