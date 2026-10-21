# Acceptance — 完整连接组 MPS 探针 A 软件

2026-10-19；新的独立验收者。依据 [Ready A 协议](mps-fullgraph-probe.md)、[Decision 013](../../.ai/decisions/013-mps-validation-before-migration.md)及实施报告。

## Result

PASS

仅确认 A 阶段隔离探针的软件实现与登记协议相符，允许协调者在新目录执行一次受监督的完整图测量。尚无本批完整图内存、速度或数值结论；不授予 B/C 迁移通过。

## Spec Coverage

- **完整资产与格式**：固定文件 SHA、逻辑图 SHA、原始有序计数 SHA 和 166700/25582938/124177617 数量；调用既有 `load_training_graph` 与 `TrainableLIF`，保留原 float32/0.275 系数。两个原始 CSR 各自转为完整 canonical COO；逐块回读 MPS 坐标和值进行精确检查，前后缓存与原图摘要核对。未发现完整 NxN 稠密化、裁边或模型替换。
- **CPU/GPU 路径**：CPU 前向使用现行共享 SciPy CSR，伴随使用现行 Torch CSR；GPU 使用驻留 MPS COO。登记与 worker 校验 fallback 必须为 `0`，核对实际设备、float32、CPU 4/interop 1；每次 GPU 运算前后 synchronize。
- **固定实验**：沿用 `make_inputs(N,1901)` 的 zero/sparse/dense；两方向、一次预热及五次正式调用共 72 次，顺序符合协议。全部调用含预热输出独立保存；输入传输、结果回读、存档和缓存构造成本与核计时分开。
- **数值与自定义反向**：全部输出按登记容差、形状、dtype、有限性和保存摘要检查；float64 统计 exact、绝对/相对误差、超容差数、零支持和符号差。固定矩阵自定义 autograd 保存全部 N 维输入梯度，以现行 CPU W/WT 计算对照，不申请逐边梯度；没有将其称为 SNN BPTT 或 Adam 验证。
- **资源与完整性**：父进程轻量登记，独占工作区锁、新目录拒绝覆盖、worker 父进程/单次 claim 校验；源 AST 闭包包含惰性本地依赖，封存快照并核对实际导入。使用既有 6GiB/1GiB/600s/0.1s 监督；MPS 分配上限向下舍入且不高于有效预算/推荐值，独立周期及阶段采样 current/driver，采样失败使 worker 失败。RSS 与 MPS driver 未相加。
- **结论门槛**：完整性、数值通过和三种输入各自前向+伴随中位耗时均严格优于 CPU 才产生 `mps_kernel_candidate=true`。单方向与总和均保存；父 `completed` / worker `protocol_complete` 与科学通过分离，没有自动切换训练或覆写旧 run。
- **范围保护**：本批实现仅新增四个 examples 模块和一个定向测试文件；审查时 `git diff --name-only` 没有既有源码改动。未修改 G29/R13、旧性能脚本或生产训练后端；另一个未提交 SciPy Draft 不属于本次实现。

## Validation

- Risk Level: S2，隔离探针的登记、监督、数值核与资源采样交互。
- Evidence Reviewed / Checks Performed：完整阅读新增实现、定向测试、Ready 协议、实施报告及所复用的资源监督/矩阵构造/输入生成/来源接口，检查当前 Git 差异。
- 实施者报告的四个不同定向检查已通过：轻量导入/源闭包/CLI，失败证据/独占/拒绝覆盖和 fallback，真实资产字节身份/快照与协议篡改拒绝，以及 200×200 signed/non-symmetric MPS 双向调度/自定义反向/内存采样。首次 tiny 项最后的 allocator cap 断言因浮点不足一字节的超量失败；向下舍入后仅重跑该失败项并通过。先前三项未重复。
- 独立验收未重复已通过测试，未运行完整图、模型数值或新增性能实验；未操作 Noria/Docker，未连接硬件。

## Issues

无阻止 A 阶段实际运行的问题。

## Regression Risks

完整图可能在缓存构造或实际 MPS 算子阶段触及资源、算子或数值门槛；tiny 通过不能排除此风险。采样峰值不是瞬时最大值或 OS 硬限额保证；核级耗时不能作为训练端到端加速。实际运行后须独立复核保存证据，再决定是否满足 B 的前置条件。

## Required Rework

无。
