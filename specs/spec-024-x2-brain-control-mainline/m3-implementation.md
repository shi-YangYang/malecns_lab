# M3 新策略与有界学习入口实施报告

状态：代码与定向轻量检查完成，等待新独立审查及协调者运行首次探针。2026-10-23。没有在实施者进程启动完整脑、MPS计算、训练、仿真或硬件。

## 实际交付与风险

风险S3：新增策略/状态/checkpoint序列化以及训练与资源监督入口，影响新学习运行身份；保留旧契约、历史数值文件及模型资产，不改旧冻结路径。

- `learning/x2_behavior_policy.py`：真实完整图166700节点、25582938非零边、124177617计数，与迁移源图/类型分组/全部3377 R1–R6和1314 DN映射严格相等；仅迁移theta，新建5行为/3阶段/1参数头。M1工程编码只进入感觉刺激，三头只读2628维DN特征。
- 完整LIF继续调用现有`MPSLIF.step`，不改方程、原内核或旧冻结优化器。600步无输入预热，500步决策，50步非重入checkpoint；一次会话内保留所有神经状态、感觉相位及计数。训练只在明确的两决策更新窗口后detach，未把每次动作重置。
- 新policy/input_map/checkpoint/state身份、源码依赖闭包快照、参数与完整状态NPZ、原始Observation/三头/解码Decision、输入编码和DN特征归档；数据、图、旧checkpoint和新checkpoint绑定SHA256。不存在pickle加载。
- `learning/x2_behavior_training.py`及`experiments/x2_behavior_training.py`：默认preview不导入Torch/NumPy/SciPy，不创建输出。显式execute使用新data目录、父进程单完整脑锁、子进程一次claim、仿真关闭预检、至少1GiB系统预留、6GiB封顶有效RSS/MPS预算、0.1秒监督、总墙钟上限及最大更新数；失败保留。
- `probe`完全按[m3-probe-contract.md](m3-probe-contract.md)实现：train首episode前两帧、第二窗唯一损失、第一窗末v保留实际梯度、最多一次Adam更新、600秒。缺失/非有限/全零跨窗梯度，或theta/行为/阶段头未实际改变均失败。首episode挥手标签令参数头按mask零梯度及不更新，属于预期，原样记录。
- `readout`冻结theta，逐完整episode计算神经特征后训练新头；验证集损失选择最优checkpoint，平局保留较早者。`joint`两决策窗口联合theta/头，持久episode状态；仅有界末次候选做完整验证。两者最多200更新，具体正式运行预算仍须协调者根据探针结果另登记。
- `evaluate`只接受事先选定的新checkpoint，读取全部100条locked-test；同一读出配对“每episode首帧常量特征”和“episode内时间反转”对照。probe/readout/joint只打开train/validation，不能调用locked-test选择模型。

## 实际检查

唯一执行的定向检查：

```sh
PYTHONPATH="$PWD/src" .venv-train/bin/python -m pytest -q tests/test_x2_behavior_training.py
```

结果：**8 passed in 0.77s**，进程exit0。覆盖preview无重栈/文件副作用、固定探针和有限预算、episode整段与任务平衡选择、传递源码身份、参数mask梯度、仅神经特征的读出接口、非pickle新checkpoint往返/篡改/覆盖拒绝、标量优化器声明。

测试仅使用小CPU头和明确的archive fixture，没有构造任何脑图或调用MPS；不把这些检查称为完整图梯度、运行资源或学习能力证据。因当前没有发生实际完整脑运行，前向/反向耗时、内存上限、跨窗信用和实际参数更新仍需首次探针回答。没有全项目测试、build或重复已通过检查。

## 运行入口与限制

审查通过后由协调者独占运行：

```sh
PYTHONPATH="$PWD/src" .venv-train/bin/python -u -m malecns_lab.experiments.x2_behavior_training --execute --mode probe --output data/x2_behavior_probe_20261023_r1
```

默认正式学习/评估预算只是可执行CLI边界，不表示已获此次探针合同的运行批准；实施者未运行它们。原始逐步数值校验的实际成本需要测量，超限不自动改内核、缩图、截短或重试。

当前M1数据显式为模拟执行反馈，评估只报告离线教师序列一致率、宏F1和特征干预差异；代码明确不把它作为P3真实反馈任务成功或P5实际动作通过。后续必须接入M2真实仿真反馈并登记新的数据/评价身份，不能在此版本隐藏替换。

新策略提供实时推理的Observation→BehaviorOutput→Decision接口，可注入真实spike观察器；本轮训练档案主要保留状态/输入/神经特征/原始输出，没有接入旧FrozenInference专用遥测manifest或桌面。M4需使用正确的新策略来源与request/decision/execution关联，不得沿用旧manifest的FrozenInference描述。没有将探针checkpoint选作部署模型。

业务改动仅新增本报告列出的3个Python模块与1个定向测试文件；其他代理的M1/M2文件和协调者规范不改，未commit/push。
