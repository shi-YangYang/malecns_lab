# 完整脑冻结推理基线实测

2026-10-23，风险S2。代码实施及21项定向检查见 [implementation-code.md](implementation-code.md)，独立结论见 [acceptance.md](acceptance.md)。

新增默认只预览的显式CLI、旧来源/参数校验与受监督worker、虚拟标记投影。原训练数值源码与官方X2组件未改。原 `requires_grad=True` 标志保留以满足旧校验，推理全程no_grad，无优化器，无参数更新。没有加载历史神经状态或Adam。

## 唯一实际基线

```bash
PYTHONPATH="$PWD/src" .venv-train/bin/python -m malecns_lab.experiments.x2_inference --execute --output data/x2_neural_inference_20261023_r1
```

退出0，完成三个预登记输入及预热后/每帧后共四份完整神经与感觉状态。登记ID `fccc36a328c77e96d9337b40f270ede534b84e89778ae25a2db626873d8383db`。最终训练checkpoint、54个登记来源hash、实际policy、完整图身份均匹配；旧源码从Git `732ddc0` 导出为该新目录只读快照，未修改历史数据。当前runtime与旧记录仅backend标签不同，Python/NumPy/SciPy/Torch及线程设置相同。

实际图为166700节点、25582938非零连接、124177617突触计数。初始600步空白预热，每帧500步、50ms神经时间；新场景状态跨三个输入连续，不重新reset。参数前后摘要相同、无梯度，原始边计数摘要前后相同。

| 阶段 | 同步后墙钟 |
|---|---:|
| 模型构造及参数恢复 | 2.6720秒 |
| 600步预热 | 7.2759秒 |
| 2.0m居中标记前向 | 6.5410秒 |
| 1.5m居中标记前向 | 6.0110秒 |
| 1.0m居中标记前向 | 5.9469秒 |
| 整次父进程 | 30.4477秒 |

三个原始 `(vx,vz,wz)` 输出依次为 `(0.224106,0.106927,0.213602)`、`(0.280164,0.147227,0.332988)`、`(0.254718,0.139239,0.335098)`。三个vx均超过预登记0.03门限，但这是离线输入探针；未向X2发送这些输出，其他轴也未执行，不证明模型具备目标跟随或停止能力。

共享锁与0.1秒监督有效，申请/有效RSS限额6GiB；实测RSS峰值1641971712 bytes（约1.53GiB），最低系统available约5.34GiB，高于1GiB预留。MPS current峰值1083982848、driver峰值1166753792 bytes；两者与RSS为重叠口径，不能相加。采样284次、无采样错误、未越限，worker退出0。240秒上限未触发。

## 证据与限制

`data/x2_neural_inference_20261023_r1/` 保存registration、来源快照与manifest、历史provenance、actual_policy、frames三组输入/输出、states四个完整边界、parameters、timing、resources.csv、mps_memory.csv、worker_result和summary。默认CLI/帮助无重计算及来源/投影/失败路径的21项软件检查已经通过，没有重复；仅真实基线一次。

预检确认私有GPU VM关闭、socket不存在、CPU容器exited；本阶段未启动仿真，未操作Noria或重启Docker，无训练、真机、push。整个脑worker已退出，无常驻计算。报告数值读取已有JSON/CSV，没有再次运行前向。

目前每帧约6秒，明显不能按原0.1秒控制时钟实时执行；GUI 60FPS与该数字无关。下一阶段[Spec020](../spec-020-x2-frozen-optimization/spec.md)只减少冻结推理的重复校验/同步，按全部边界状态和输出精确一致、计算耗时至少下降20%的预登记门槛比较。后续X2闭环需采用独立动作发布和停稳后推理，不把模型慢计算期间的旧观测当作新鲜反馈。
