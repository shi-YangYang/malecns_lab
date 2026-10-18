# Acceptance — P3e0 首次完整 R 生成成本实测

## Result

**PASS**。由未参与协议编写、前次协议验收或本次执行的新 Agent 独立核对 [Ready 协议](p3e-generation-probe.md)、[实际报告](p3e-generation-run-20261019-r1.md)及 `data/topology_generation_20261019_probe_r1/` 的原始证据。结论仅为：固定 100000 提案的完整规模 R 成本样本已实际生成、保存、重新加载验证，执行与报告符合协议。该图仍是 `generation_probe_only`，不是正式 R 训练图；本批不关闭 AC6 或整体目标。

## Spec Coverage

| 要求 | 独立核对与结论 |
| --- | --- |
| 完整母本、只读身份与研究角色 | 流式重新计算母本文件 SHA，等于登记值及 worker 的运行后值 `98f83158ed12f0c836fb085c2e286e54f9458630c2a158bee3164af52648f6f3`。读取母本和产物的小型 ZIP 元数据，确认母本为 `full_reference`、产物为 `research_control`，source、license、roles、节点元数据摘要及父 source/逻辑图身份保持。没有将派生图伪装成完整参考母本。 |
| 预先登记的参数和来源 | 登记固定 seed=1901、max_attempts=100000、target_swaps=100000、with_paths=false；产物 provenance 中参数、协议名和 `self_edges=fixed` 一致。登记 canonical hash 复算为 `f140cfb27220784590b181808ae9bce6b8415af72a19c807d5e7136bab09e363`。101 项登记文件摘要全部匹配，包含全部 98 个当前包内 Python 文件、两个完整实验脚本及已提交的 Ready 协议。 |
| 真实生成与真实重载 | `worker.py` 先调用 `create_research_control`，保存返回值，随后实际调用 `load_research_control(control, parent)`，没有以存在 NPZ 或只读 JSON 冒充重载。已核对这两个既有 API：构造执行 `validate_rw`，重载重新核对母本、派生内容/来源、注释角色及离散不变量；所有断言完成后才写 completed。 |
| 完整规模及机械校验的依据 | worker 的生成后重载统计为 166700 节点、25582938 边、124177617 绝对整数权重和、101 条自环，与父 assessment 一致。独立读取 NPY 头，母本/产物 shape 均为 `(166700,166700)`，body_ids 为 166700、indptr 为 166701、indices/data 为 25582938，整数格式一致。逐节点度、符号入度、自环权重、逐源强度、全局有符号权重多重集及全部元数据的校验依据为实际执行的已验收 `validate_rw`；本次没有重新加载完整矩阵或声称新增逐源权重多重集验证。 |
| 交换预算与完整记账 | `51847 + 47509 + 640 + 3 + 1 = 100000`；同目标、原自环拒绝数均为 0。成功数 51847、`target_reached=false`、`stop_reason=proposal_budget_exhausted` 在 creation、产物 provenance 和 assessment 中一致。按预登记固定提案停止符合成本探针要求，不是假装达到成功交换目标。 |
| 诊断及产物封存 | 保留边 25479474、新增/移除各 103464、共有边权改变 1、保留比例 0.995955742065278；类型配对改变 145877、边计数 L1 变化 192204，均与原始 creation 和图内 diagnostics 相同。派生 provenance canonical hash 为 `be0b65aac40e58b8fc0b9dc6c7a5f76e60529e83782036035c6fd7ec912d8dfe`；文件 SHA 独立复算为 `0e7e337aea87b7658fb0daa97628f1331bdda4895ac72e063f497d7f09d127c9`，与 worker/assessment 一致。 |
| 资源、串行及服务边界 | `launch.py` 在共享 `workspace_lock()` 内登记和执行一次 worker，父进程不加载图。启动 available 7267909632，预留 1073741824，申请 6442450944，有效上限 `min(6 GiB, available−1 GiB)=6194167808` bytes；600 秒与 0.1 秒监督覆盖生成和重载。脚本无 Noria/Docker 操作、重试循环或神经运行。 |
| 原始终态和独立父判断 | `result.json` 与 `supervisor_result.json` 内容及文件 SHA 相同，均为 completed / worker_exited / returncode=0 / worker_status=completed；末 phase 为 finished，worker 自身亦 completed。assessment 单独保存 passed，没有覆盖原始监督结果。其 9 项文件清单摘要全部重新验证，worker 的 3 项产物摘要也一致；目录恰有两个脚本及十个登记/运行/图文件，无未封存临时尾部。 |
| 不覆盖及不扩大运行范围 | 登记已存在时 launcher 立即拒绝，生成器及保存使用不可覆盖路径；无追加提案、换 seed、重试或神经评估分支。报告所述执行一次，与当前原始证据和脚本路径相符；验收没有重复执行实验。 |
| 科学边界 | 图内 paths/neural_activity 为 not_executed、mixing 为 not_established。约 0.4044% 的边位置变化不能支持充分随机化、输入到 DN 功能通路或拓扑收益；报告明确保留这些限制，正式 G/R 图、同机会训练及未见场景比较仍未完成。 |

## Validation

- Risk Level：**S0**，本次只新增验收报告；首次完整 R 实验本身按既有 Ready 协议审查实际证据，不修改或重新执行业务代码。
- 阅读 AGENTS、项目使命/技术栈、项目规则、交接、Spec 008/Plan、独立验收流程、Decision 009/011/012、P3e0 协议/协议 PASS、图工具既有验收和本次运行报告。
- 静态阅读完整 `launch.py`、`worker.py`、图生成/重载及其不变量、来源封存、资源监督和共享锁实现。
- 一次标准库只读证据检查退出 0：复算 101 项源码/脚本/协议摘要、9 项 assessment 文件摘要、3 项 worker 文件摘要及母本 SHA；检查登记与母/派生 provenance canonical hash、NPY 头与小元数据、全部 209 行资源 CSV、终态及完整文件清单。没有导入 NumPy/Torch、加载完整图或模型，也没有扫描原始完整数据集。
- 资源 CSV 时间严格递增、全部有限；最大采样墙钟 23.068135542 s，监督终态 23.072730625 s；采样峰值 RSS 2075377664 bytes，最低 available 4592156672 bytes，均未触发相应预算。
- 生成外层耗时 16.017872917 s、重载 3.310659917 s、worker 总墙钟 19.456357375 s；创建函数另记录其自身计时 16.009766875 s 及进程生命周期 getrusage 峰值 2358689792 bytes。报告准确区分该峰值与 0.1 秒采样峰值，未用较小者冒充瞬时最高占用。
- 采纳已验收图工具的 11 项定向测试记录，没有重跑测试、生成、重载、大图计算或任何设备/服务操作。仅新增本报告，不创建 commit。

## Issues

本批范围内没有阻断问题，也未发现运行报告需修正的数值事实。Ready 协议中的“尚未执行”属于已被登记 SHA 绑定的运行前状态，不能为更新状态而改写；当前事实由实际报告及本验收承接。

## Regression Risks / Evidence Boundaries

- 本次轻量独立核对没有重新计算完整矩阵的离散不变量、逻辑图 SHA 或类型对直方图；其依据是已验收且来源摘要相符的 API 在本次 worker 中实际执行，配合未变化的完整产物 SHA、原始成功终态及小元数据交叉核对。不能把本报告描述为第二次完整图计算。
- 资源监督是采样保护，getrusage 又是 worker 进程截至生成函数返回时的生命周期峰值；二者均不能提供全过程精确瞬时进程树峰值或硬内存隔离保证。
- 没有新增逐源权重多重集独立检查、当前 3377 个视觉输入到 1314 个 DN 的路径检查、充分混合标准或正式研究策略入口。它们属于后续正式拓扑实验，不因本次成本样本通过而消失。
- 总耗时包含加载、验证、诊断和写盘，未隔离纯交换循环；正式预算不能简单按提案数线性倍乘，也不能把本图投入训练后按行为挑选后续图。

## Required Rework

无。本批实际证据及报告可交付，停止本批验证。后续按新协议冻结正式 R 预算和研究边界；不重跑此成本样本，不关闭整体研究目标。
