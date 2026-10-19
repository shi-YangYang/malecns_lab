# Acceptance — 全 GPU 修复训练集成软件

## Result

**PASS**，仅覆盖 [集成规格](gpu-recovery-integration.md)与[实施报告](implementation-gpu-recovery-integration.md)中的新增软件。本验收者未参与实现。此结果允许进入已登记的完整图 initialize、可丢弃 gate、正式单窗与新进程单窗实际门槛，**不代表这些完整图门槛已经完成，不代表已学会控制或允许实飞**。

## Spec Coverage

- **GPU 数值路径与完整图准入：满足。** `GPUPolicy` 复用原完整 MPS 脑与神经元映射，实际 GPUDrive 身份及脑身份与轻量预登记值相等后才执行。生产登记严格沿原 CPU/MPS 父链验证至指定第 160 窗、checkpoint 与 prefix，要求 `full_reference`；新闭包包含实际新模块及原严格读者依赖。fixture 注入仅存在测试脚本，生产入口没有小图或 CPU fallback 开关。
- **损失、递归信用与前向：满足。** logit 目标、原 gain 正则和原动作 MSE 分别计算和记录，动作仍是原尺度乘 tanh，不改变轴符号。生产 horizon=100、theta 原始稳定 L2 上限 1000 被登记与恢复入口双重冻结。50 步重算将 phase/counts 和神经状态作为显式输入输出；每个 100 步边界只 detach 递归神经状态，DN 累加保持梯度。两帧全部 1000 步前向及原两半帧累加顺序保留。已有定向检查证实重算不改变输入或末态，改变信用范围只影响 theta 梯度而保持前向及读出梯度。
- **优化与错误饱和 gate：满足软件范围。** 网络、loss、raw gradient、稳定分组裁剪、唯一主参数和 fused Adam 的 m/v/step 留在 MPS；保存证据的 CPU 算术不回写优化器。超界、非有限或零内部梯度拒绝，保存失败现场；更新后独立审查实际参数变化、裁剪、矩与 Adam 参考算术。gate 固定读取旧第 160 窗负 vx 教师，从新 genesis 实际前向/反向并联合更新，要求两帧 vx logit、vx loss 与总 logit loss 下降；明确不把 logit 下降解释为动作已反向。
- **迁移、genesis 与丢弃 gate：满足。** 父 theta/读出逐位继承，优化器全部零矩和零步数，新场景按 600 步 blank 初始化。父进程审查 actual policy/optimizer、继承数组、零矩、phase 初始化、全状态类型/时钟及 SHA，不只接受 worker 的布尔声明。gate 不写训练 checkpoint；正式首窗从 untouched genesis 恢复。原第 160 窗不会被称为同协议第 161 窗。
- **完整恢复和有界续训：满足。** 新存档明确 float32 phase/int64 counts，保存全部 pending、神经时钟、感觉状态、参数和所有 Adam 数组，设备往返恢复逐项比对。首两段强制各一窗，后续每段最多四窗，累计最多新 160 窗；场景内保留实际前向末态，场景转换才 reset。每窗/段的选择输入、状态、数值、checkpoint、prefix 与资源证据绑定，失败或不完整先前段阻断续接。前两窗后能否推进长训练仍须协调者完成规格要求的独立实际验收，不能用本软件 PASS 替代。
- **资源及旧资产：满足。** 轻量父进程、PID/共享锁/一次 claim 后重型导入、申请 6 GiB 与 available 减 1 GiB 有效上限、600 秒/0.1 秒监督和 MPS allocator 取样均沿已验收设施接入。没有操作 Noria/Docker 的代码。独立 stdlib SHA 核对旧 r3 **42 份绑定源码保持，changed=[]**；GPUDrive 与 GPUAdam 两份已验收源码也与当前提交逐位一致。

## Validation

- Risk Level：**S3**，限新增训练数学、状态持久化、进程与证据契约。
- 阅读 AGENTS、使命、技术栈、全部现有项目规则、独立验收流程、当前交接、Decision 015、修复 Plan/集成规格、实施报告、GPU 基础 R1 PASS、梯度诊断 r2 实际 PASS 与原因报告。
- 静态审查新增 10 个 learning 模块及两份完整测试；对照原 MPSPolicy/MPSLIF、GPUDrive/GPUAdam、父来源/游标/资源监督和状态接口，核对实际调用路径与测试断言。
- 采信实施检查 session **59148**、exit **0**，`3 passed in 15.56s`：纯重算/递归信用、实际负目标纠错、全状态恢复后再更新、伪造 dtype/读出梯度与门槛拒绝等。没有重复运行。
- 采信后续子进程集成检查 session **94962**、exit **0**，`1 passed in 10.85s`：显式五节点 fixture 仅替换图/旧父链准入，实际 initialize→gate→两个独立训练子进程及资源/证据链运行；首窗保持未被 gate 更新的 genesis，篡改 gate terminal 被拒绝。该检查使用最终窗口实现；不冒充完整图证据。
- 唯一新增执行检查为 stdlib 读取旧 run JSON 并比较源码 SHA、读取 Git 中两份基础源码。未导入 Torch、未运行数值作业、模型、训练、全量测试、build 或 lint；没有重复已通过检查。
- 本验收只新增本报告，无业务代码、旧数据或提交修改；既有未跟踪 SciPy 审计文件保留。

## Issues

本软件范围内无阻断项。完整图的实际 GPU 资源、genesis/gate 数值、两次真实窗口与进程恢复仍待执行和独立实际验收。

## Regression Risks

- 新 float32 感觉、目标函数、10 ms 递归信用及清零 Adam 均是新数值协议；原完整脑持续前向保留，但不能称为原轨迹逐位延续。
- Python/设备标量验证存在同步开销；全部训练数值留在 MPS 不等同于吞吐必然提高。实际速度应由完整图运行测量。
- gate 的固定特征一步纠错和内部参数变化仅是工程门槛，尚不证明闭环跟随、泛化或真实连接组的性能贡献。

## Required Rework

无。本软件范围必要证据充分，停止追加验证；按既定实际阶段继续。
