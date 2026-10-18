# P3b — 完整图三轴动作损失的一次联合更新

状态：软件已实现并获新的独立[PASS](acceptance-p3b.md)，[实施报告](implementation-p3b.md)及验收报告已全文读取处理，36项定向检查通过，无返工项；实际完整图运行尚待本软件提交后进行。此前独立[协议PASS](acceptance-p3b-protocol.md)、P3a软件65b35ca及[四训练场景封存](p3a-data-run-20261019-r1.md)均已完成。本文件冻结研究选择，不代表模型已经训练；实际运行与独立实际验收另行记录提交。

## 问题与边界

P1已证明内部增益可数值优化，但目标是L1电位。P2e接通全部DN到三轴动作的可微接口；持续输入实测表明固定8mV背景在21–25ms开始有DN差异，20–50ms窗口已有位置/大小差异。本次要在**完整图**上验证一次真实动作模仿损失可共同更新theta与读出，且用内部急性恢复检验动作是否依赖此次内部变化。

仅使用训练场景的两帧固定教师轨迹，不是神经策略闭环，也不证明未见场景泛化、性能优势或果蝇学习机制。没有超参数搜索、验证/测试选优或自动重试。阴性结果仍保留；不因信号存在而预判梯度可用。

## 固定协议

- 完整参考图、原始边/突触计数不变；生产NeuralPolicy、全部3,377输入/1,314DN、CPU float32、4线程/interop1；背景固定L1–L5 8mV/rho0，不训练背景。原始模型默认LIF配置dt0.1ms，空间seed1701、驱动seed1702、150Hz/68.75mV，读出seed1703。
- `frame_ms=50`、`precondition_ms=60`、`control_dt_s=0.1`。50ms依据已观测持续期信号选择，不是最优或生理时钟声明；两帧共同形成100ms反传窗。第一帧后不detach/reset；60ms空白预处理在no_grad下，梯度不穿过该阶段。
- 使用P3a封存的四场景数据集，**仅train-sine-combined索引0、1**，控制时刻0和0.1s。必须相机/时钟匹配、两帧valid，否则以原样失败记录，不改选其他帧。标签为对应帧teacher_target，使用限幅后的SI三轴值；不能取下一帧标签或executed_action。所有选定输入/标签副本与数据集ID、文件摘要纳入运行证据。
- 参数集合与顺序：theta组`brain.theta`；读出组`readout.weight,readout.bias`。Adam两组lr分别0.001和0.0001，betas=(0.9,0.999)、eps=1e-8、weight_decay=0、amsgrad=False、foreach=False、fused=False；恰好一次optimizer.step。
- 动作损失为两帧×三轴全部元素等权平均的 `((action-target)/[0.3,0.2,0.5])**2`；另记各轴MSE。正则为 `1e-4*mean((gains-1)**2)`，两者分开记录。初始theta严格为0、gain严格为1，首步正则和其theta梯度必须为0，因此不能用正则解释非零内部任务梯度。
- 全参数统一clip_grad_norm_=1.0，error_if_nonfinite=True，固定foreach=False；记录clip前/后全部梯度、总范数及实际clip系数。任何非有限值停止，保留到失败点的证据。
- 五个分支：baseline(θ0,r0，带图)、updated(θ1,r1)、acute_restore(θ0,r1)、full_restore(θ0,r0)、checkpoint_reload(θ1,r1)。后四次no_grad；每个分支fresh initial_state、相同两帧，不能把另一分支隐藏状态复用。急性恢复包含参数改变预处理状态的影响，不据此定位唯一刺激期间通路。
- 动作、特征的恢复与重载容差均rtol1e-6/atol1e-7（按各自SI/mV单位），另报精确相等与最大绝对误差。内部功能影响用updated与acute_restore全部2×3动作按同一容差比较，另记录全部2×2628DN特征变化；不事后挑神经元或轴改变阈值。

## 子进程执行与资源

参考现有probe/observability轻量父进程结构，但使用独立协议/结果schema，不将P1电位目标语义混入新动作试验。新CLI拟为 `python -m cnsbridge.learning.joint_probe --graph <reference> --dataset <sealed-dataset> --out <new-run>`。协议固定在版本化源码/JSON中，真实路径与数据集ID在启动前登记。

父进程先创建新目录，记录完整协议ID、源码SHA、数据manifest字节摘要/身份、当前预算和环境，再启动worker。所有完整图加载、Torch模型、反向及参数重载必须在受监督子进程内。用既有resource_guard：额外预留1GiB、申请RSS6GiB、有效上限min(6GiB,available−1GiB)、600s、0.1s采样；Noria/Docker不变。禁止并行其他全图作业。

已有数据的成本估计为 `1.2*(40*3.897782291 + 5*13.44389775)+35 ≈303s`：五次160ms前向、一次100ms反向与余量。这里只是线性量级估计，策略聚合/完整图加载/存档也计入实际墙钟。100ms带图内存未知，不用无梯度RSS估计冒充；达到资源门槛即保留失败并停止，不缩窗、裁图或重复到阳性。

更新后先保存完整策略参数checkpoint，再做干预；完整恢复后释放旧模型、optimizer及所有带图Tensor，才生产重载训练后checkpoint，避免双份完整图常驻。所有分支输出转换成独立NumPy数组后释放图引用；记录phase和已完成帧，终态须由父进程实际收取。不得把新参数存档称为优化器/隐藏状态续训存档。

## 必须保存与独立校验

保存完整policy/model/graph/输入输出与时钟身份，原始图计数更新前后摘要；登记源版本与实际环境。NPZ均allow_pickle=False，字段/shape/dtype/有限值/ID固定，封存后文件SHA写入清单。

- 选定两帧的输入、标签、valid/time、数据集与场景引用；全DN bodyID顺序。
- 全部theta/gain、读出weight/bias的before/after值；三参数clip前后梯度、范数/系数、正则梯度及优化器超参数。
- 五分支的两帧三轴动作、两半窗全部DN特征、动作每轴/总MSE、正则和总loss；每分支预处理与末状态的完整值摘要、驱动/神经步号。不把只保存摘要说成已记录全节点每步轨迹。
- 独立策略checkpoint ID及文件；保存参数与after逐位一致。恢复/重载数值比较包括exact、allclose、最大误差。acute_restore保留r1、theta=0，full_restore与baseline参数逐位一致。
- 监督器原始终态、资源CSV、worker终态和阶段耗时。父校验不启动模型，不只信worker自报PASS，独立复核全部身份/产物/损失/更新/对照与资源。

结果分三层，不能混成一个成功布尔值：

1. `protocol_complete`：完整证据、图不变、有限数值、恢复/重载通过和预算终态有效。
2. `joint_numeric_update`：非零有限theta任务梯度、theta/gain实际变化，读出权重/bias实际更新；固定背景未改变。记录每项成立与否；不要求loss下降来隐藏阴性。
3. `internal_action_effect`：updated与acute_restore动作存在超登记容差差异。另报动作MSE改变方向；即使内部影响损害模仿，也如实保留。若参数改变而动作无可观察差异，只报告内部数值更新。

协议无效/资源失败不能升级为领域通过；协议完整但学习/功能条件阴性不是缺失证据，不删除。一次loss下降、单训练片段以及数值更新均不关闭总AC3–AC6。

## 实施与验收范围

建议单一实施者负责新 `learning/joint_probe.py`、按职责拆出的 `joint_probe_evidence.py` / worker模块和定向测试；不复制大型既有核心或修改旧工具语义。读取/复用P3a数据与checkpoint，保留旧数值探针与研究基线。工具文档、implementation-p3b.md和独立acceptance-p3b.md另行记录。

S3检查限本协议：小电路实际两帧梯度/联合更新、同帧教师与无中途reset、完整恢复/急性恢复/重载、参数clip与非有限拒绝、轻量登记和子进程边界、独立证据拒绝缺失/改写/阴性分类。仅显式数值fixture，生产CLI无假图回退；不得用fixture报告完整图运行。

软件独立PASS并commit后，先验证P3a实际数据，再新目录登记一次完整图试验。新的实际验收者读取原始证据，不重复模型或已通过检查。下一阶段是可恢复的长场景联合训练及闭环/对照，不把此pilot当作路线终点。
