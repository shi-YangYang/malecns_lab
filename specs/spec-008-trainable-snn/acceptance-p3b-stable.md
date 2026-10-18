# Acceptance — P3b r2 稳定分组裁剪软件

## Result

**PASS**（软件范围）。2026-10-19，由未参与本轮实施的新独立验收 Agent 审查。实现满足 [r2 稳定分组裁剪协议](p3b-stable-group-clip.md)及其[独立协议验收](acceptance-p3b-stable-protocol.md)。协调者处理本报告并提交后，可在新目录执行一次受监督 r2 实验。

本报告没有运行生产完整图，不证明 r2 联合更新或内部动作作用已经成立。r1 的实际 FAIL 保持；AC3–AC6 仍未完成。

## Spec Coverage

- **完整路径与范围：通过。** 本轮只修改 joint_probe 的登记、worker、优化证据及父校验，新增历史锚点和优化复核模块，更新相关测试/使用说明。原脑核心、策略、感觉驱动、教师数据、部署参数 checkpoint 和资源监督源码保持不变；两帧之间仍直接传递完整状态，无 detach/reset。原 50 ms/帧、60 ms 无梯度预处理、100 ms 反传窗、固定背景、全部输入/DN、目标和三轴损失均保持，未添加视觉到动作旁路或生产假图回退。
- **稳定分组裁剪：通过。** 所有原始梯度先保存并检查 float32/有限性，再修改任一梯度。theta 与 weight+bias 分别以 float64 平方归约得到 L2，计算登记系数，显式转换为 float32 后乘入原梯度；两个系数、完整前后梯度、组范数及非零数均保留。worker 和父复核均验证每组上限 1 加预登记 1e−6 误差，父复核另检查合并 sqrt(2) 上限。新规则明确是分组裁剪，不冒称旧全局裁剪的等价修复。
- **一次 Adam 与独立复算：通过。** 保持两个参数组、原学习率/betas/eps、foreach=False/fused=False 和唯一一次 step。三个参数对象按稳定名称保存 float32 scalar step=1 与同形 float32 一阶/二阶矩；严格字段、shape、dtype、有限性和步数校验落实。独立 NumPy 复算从零矩开始，按 float32 运算记录二阶矩下溢，未使用理想化首步公式代替实际矩；矩容差为 rtol=1e−6、atol=float32 最小正次正规数，参数更新保留原容差。对照已安装 Torch 的非 capturable/非 foreach Adam 路径，lerp、addcmul、偏差修正、sqrt/div/add 和 addcdiv 的语义与复算及定向算术检查相符。
- **原始梯度与历史来源：通过。** 新登记绑定 r1 registration/protocol ID、旧 registration/result 和七个数值/身份比较文件摘要，同时核对旧终态的六文件清单。独立只读 SHA 检查确认九个冻结锚点、六个原终态文件及十七个不允许改变的源文件均与实际资产一致。数值 NPZ 的锚点被明确解释为 r1 已验收后为 r2 冻结，不声称原失败终态已封存它们。旧记录没有重写，当前登记允许更新优化/证据源码，仍要求脑/策略/数据等来源保持。
- **优化前对照门槛：通过。** 当前同帧输入、DN 顺序、完整初始参数、策略身份、baseline 全部动作/全部 DN 特征和原始梯度与 r1 比较，保存 exact/allclose/maxdiff。初始参数和输入要求精确一致；动作/特征/梯度使用预登记 rtol=1e−6、atol=1e−7。数值超差时先保存比较报告并在裁剪/Adam 前停止；父复核独立重做比较，不能仅依据 worker 的通过标记放行。
- **五分支、证据与结果分层：通过。** 更新后存档、急性恢复、完整恢复、释放旧模型再生产重载的顺序保持；独立 NumPy 参数/读出/损失/恢复/重载和状态摘要核对继续存在。新增命名 Adam 状态及 r1 比较进入必需产物清单；v2 registration/protocol/worker/final schema 与旧 v1 明确区分。protocol_complete、joint_numeric_update、internal_action_effect 分开，零梯度或 loss 增加等阴性结果仍可构成完整协议证据，不被自动提升为学习成功。
- **父进程与资源边界：通过。** 父进程及新历史/优化复核模块不构造模型或导入 Torch；完整图加载、反向及 checkpoint 重载继续只在受监督 worker。1 GiB 额外预留、6 GiB 申请上限、available−reserve 有效预算、600 秒/0.1 秒采样和新目录限制保持。没有修改 Noria/Docker 或加入自动重试、缩窗、裁图、改选输入的路径。

## Validation

- Risk Level: **S3**，范围限受影响的训练优化与证据链。
- Evidence Reviewed / Checks Performed：AGENTS、使命/技术栈、项目规则及独立验收流程；Spec 008/Plan/当前交接，Decision 009/011/012；原 P3b 与 r2 协议、r2 协议验收、r1 实测及失败证据验收；本轮实施报告、全部改动 diff、新 history/optimizer 模块、worker/contract/evidence 的完整相关路径和定向测试。
- 已读取实施者收取终态的检查记录：`PYTHONPATH=src .venv-train/bin/python -m pytest tests/test_joint_probe.py -q`，**62 passed，1 warning，1.64 s，exit 0**。测试内容包含真实数值小电路两帧连续反传/一次 Adam/五分支、巨大但有限梯度的组间隔离与二阶矩下溢、NaN/±Inf 在任一梯度变更前拒绝、命名矩字段篡改、历史身份拒绝及阴性分类。没有无理由重跑这些已通过检查。
- 另执行一次只读标准库 SHA 检查，针对测试 fixture 无法证明的真实历史锚点：**9 个冻结文件、6 个原终态文件、17 个保持源码及 r1 ID 全部相符**。仅读取小型历史文件和源码，不导入 Torch、不读取图矩阵、不调用模型或生产 validator。
- 阅读本机已安装 Torch Adam 源码确认实际运算路径；未再执行优化器、模型、硬件、全量测试、build、服务操作或 commit。仅新增本验收报告，协调者的交接更新不计为实施代码。

## Issues

没有阻断软件提交或一次新登记运行的问题。实施报告所述实际图、学习与生理解释限制与代码一致，没有将数值 fixture 当作实际 r2 证据。

## Regression Risks

分组裁剪重新分配内部与读出的更新尺度，不能解释巨大原始 theta 梯度或保证长轨迹稳定。裁剪后部分分量/二阶矩可能下溢，实际有效增益变化和动作效应须以 r2 产物判断。采样监督不等于硬隔离；后四分支及完整图父复核还没有实际 r2 证据。一次离线两帧成功也不代替闭环、泛化、拓扑对照、部署时延或实飞。

## Required Rework

无。协调者处理本报告后提交软件，再按冻结协议在 `data/joint_update_20261019_r2/` 运行一次并由新的实际验收者复核。失败或阴性原样保留，不放宽容差、不覆盖 r1、不自动改变超参数重试；整体目标保持未完成。
