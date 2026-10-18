# Acceptance — 完整图伴随 CSR 缓存（软件）

## Result

**PASS** — 仅覆盖 [子规格](p1-adjoint-cache.md) A–D 软件部分。本验收者未参与实现，独立阅读了规格、实施报告、相关源码、测试及工作区 diff；没有修改业务代码。

E 的同协议完整图 r4 对照尚未运行，不属于本次软件 PASS。不得据此把整批标记完成，也不得声称实际提速、降低内存、视觉到 DN 可观测或目标跟随已完成。

## Spec Coverage

| 条目 | 结论与证据 |
|---|---|
| A：双方向来源、全边及临时内存 | PASS。`TrainingGraph.counts` 为 canonical、唯一非零、有符号整数的 `(post, pre)` CSC。前向维持 `counts.tocsr()`；伴随直接将原 CSC 的 `indptr/indices/data` 解释为反向 `(pre, post)` CSR，shape 反转。两边采用相同 dtype、`weight_mv` 缩放并复制到 Torch，没有筛选或稠密化。注册前向矩阵后先 `del csr`，再分配伴随缓冲；原始只读图保持。 |
| B：独立代数及全细胞反向 | PASS。自定义 backward 仅使用保存的伴随 CSR 与完整上游向量相乘，返回全部密集输入梯度，固定双矩阵梯度为 `None`；不存在逐步 transpose、转换或按发放索引剪枝。数值检查以独立 dense 代数对照 float32/float64、三种尺度、正负非对称边、自环、孤立行列及部分/全部静默输入；既有延迟递归静默梯度检查保留。 |
| C：方程、持久化、身份及 P2c | PASS。diff 未修改增益参数化、代理梯度、积分、门控、输入限制或延迟。新 buffer 为 `persistent=False`；核心 `state_dict` 仍仅 theta，checkpoint NPZ 内容不变。`trainable.py` 文件摘要继续参与 model identity，恢复时严格比较整个 identity，无旧身份绕过。生产入口、数值工厂及 `LaminaTonicLIF` 均调用同一父类构造/step；P2c 原有独立 schema 和 gain-only checkpoint 拒绝规则保持。 |
| D：检查、范围与文档 | PASS。实施报告记录的 44 项定向检查覆盖此次核心反向风险和相关 checkpoint/P2c 路径，检查范围相称。业务 diff 限于 `trainable.py`、核心数值测试和使用说明；已替换旧逐步转置路径，未增加兼容包装。文档明确额外静态缓存约 0.287 GiB，未把小图检查说成完整图性能证据。 |
| E：完整图实际对照 | 未运行、未验收。须由协调者软件提交后使用新目录、新登记执行，并独立报告与 r3 的数值、时间和资源对照。 |

## Validation

- Risk Level: **S3**，核心共享反向算子改变；采用与其直接相关的定向软件证据。
- 已阅读 `implementation-adjoint-cache.md` 的完整命令及 **44 passed, 1 warning in 1.37s** 记录。唯一警告为 Torch sparse CSR beta 提示；此处没有把实施者执行记录写成验收者重新运行结果。
- 独立核对 `trainable.py`、`trainable_graph.py`、`tonic.py`、`learning/checkpoint.py`、相关数值及持久化检查的实际断言，确认其与上表需求对应。
- 全仓调用检索仅发现 `TrainableLIF.step` 和数值测试调用 `fixed_sparse_matvec`；二者均采用新签名，无遗漏的旧调用路径。
- 阅读相关 diff、新规格/报告及治理交接；没有发现需要扩大或重复检查的具体疑点。因此未重跑已通过的 44 项检查，也未运行全套测试、build、lint、完整图、资源监督或硬件。
- 未停止或调整 Noria/Docker，没有修改资产、历史证据或 Git 历史，未提交。

## Issues

无软件范围内阻断问题。

## Regression Risks

伴随缓存增加常驻稀疏数组；完整图构造峰值、反向时间及总成本仍以 E 实测为准。新的源码身份使旧 checkpoint 在当前实现中不再匹配，这是既有严格身份契约的预期行为；没有迁移或覆盖旧产物。

## Required Rework

无。A–D 可按既定逐步提交流程关闭；E 及整体 AC3–AC6 保持待完成。
