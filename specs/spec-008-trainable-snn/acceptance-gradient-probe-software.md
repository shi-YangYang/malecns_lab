# Acceptance — 固定窗口MPS递归梯度诊断软件

## Result

**PASS**。独立验收范围仅为 [GPU修复Spec](gpu-recovery.md) A 的诊断软件与 [Plan](gpu-recovery-plan.md) A1–A5；允许协调者进入一次已登记的受监督完整图诊断。尚未执行完整图四分支，因此不将本结论解释为根因已定位、修复有效、A1科学标准完成或全GPU训练验收通过。

## Spec Coverage

- 固定来源：`gradient_probe_evidence.load_anchor` 锁定原r3 run ID，使用既有严格reader核对CPU父链、完整160窗封存链及159→160父档关系；生产入口不能改选图、数据窗口或horizon。选择两帧必须与旧160证据相等，原图角色必须为`full_reference`。
- 完整恢复与不更新：每个分支复用原`restore`，恢复全部参数、Adam、神经/感觉数组和时钟；实际设备往返与旧档比较。CPUAdamBridge在此仅用于恢复和不变性核对，无`step`调用，分支后核对初态/参数/Adam全部数组保持。该诊断忠实重放旧感觉编码，不伪称B阶段全GPU训练。
- 前向与损失：四支串行调用同一个原`_block_function`，50步checkpoint及逐步DN相加次序保持。每支均执行原两帧1000步、原动作MSE和gain正则；500/250/100仅在登记边界detach完整递归状态，DN运行和及早期输出仍参与损失，不截短前向或只保留末段输出。
- 比较门槛：1000支核对原160动作、全部DN特征、theta/读出原始梯度、全部末态数组及状态摘要；短支核对同次1000支前向、末态及读出梯度。差异或非有限值先保留分支产物，再失败退出，无运行后放宽门槛路径。
- 伴随证据：每50步记录v/g/全部pending的pre-detach输出hook；refractory明确不可微。hook不返回替代梯度，不保留大梯度张量；未触发标记`observed=false`。稳定范数先缩放再平方，避免1e30量级float32直接平方溢出。
- 固定feature候选损失：MPS计算`softplus(2z)-(1+y)z`及其解析导数比较；0.1步仅修改局部logit变量，模型参数不更新，原部署动作仍与基线比较。该局部纠错证据不代替真实神经网络训练结果。
- 监督与来源：轻量父入口持有共享锁，worker在Torch导入前核对监督父PID、锁和单次claim。复用6GiB申请、available减1GiB实际预算、600秒/.1秒监督、MPS分配限额与保存样本边界核查。原42份源与新增4份源共同绑定；产物清单、SHA、原图计数前后摘要、恢复及固定输入在父进程复核。无Noria/Docker操作或旧资产覆盖路径。

## Validation

- Risk Level：**S3**，范围限定于反向边界、状态恢复及诊断证据/监督接口。
- 静态审查四个新增`gradient_probe*.py`、`tests/test_gradient_probe.py`、实施报告，以及直接调用的原`mps_recompute`、`mps_window`、`mps_state`、`mps_optimizer`、`mps_policy`、`mps_contract`、`mps_evidence`、`mps_memory`、`resource_guard`和`LIFState.detach`。
- 已阅读并采用实施者报告的唯一相关数值检查：session76019，exit0，`tests/test_gradient_probe.py` **5 passed in63.55s**。检查覆盖小电路四范围前向/状态与读出梯度相等、原完整范围theta梯度相等及短范围变化、重算纯净、大值稳定范数、饱和logit纠错、非法直接worker和单ULP拒绝。未把小电路作为完整图实测。
- 本验收仅额外用stdlib读取原run来源表并计算源码SHA：**42份全部匹配，changed=[]**。没有导入Torch、执行模型或重算训练。
- 未重复已通过测试，未运行完整图、旧98窗检查、全项目suite、build或lint。其他Agent新增的GPU感觉/优化器模块不属本验收范围。

## Issues

无阻断本批软件范围的问题。完整图baseline的逐项精确重放、实际伴随量级与资源峰值仍须由后续一次受监督实测证明；软件PASS不提前保证实际运行通过。

## Regression Risks

观察hook会增加同步和额外计算，探针时间不能当作正常训练性能。递归horizon是状态传播的分段范围，保留全部DN损失，并非只训练最后若干神经步。若完整图baseline或分支前向比较失败，应保留失败证据并分析，不能用本软件PASS或小图结果覆盖失败。

## Required Rework

无。完成实际诊断后，由新的独立验收者复核保存证据，再冻结B/C修复与执行方案。
