# Acceptance — P1a 完整图探针工具

## Result

FAIL

2026-10-17，由未参与本批实现的独立验收 Agent 审查。结论仅针对 [P1a 工具](p1-tools.md)，不对未执行的完整图参数更新、目标跟随或实飞作结论。

## Spec Coverage

| 条件 | 结论与证据 |
|---|---|
| checkpoint 来源与参数身份 | PASS（软件路径范围）。保存新目录、JSON 与 `allow_pickle=False` NPZ；数组摘要、完整模型身份、图角色、类型映射、输入集合、精度、参数形状和有限性均在参数写入前验证。生产重载经完整参考图加载器，重新构造核心身份并比较当前核心代码摘要；数值夹具不能进入生产重载。 |
| 固定完整图探针协议 | PASS（代码审查范围）。生产加载保留完整参考图，输入与观察使用预登记原始类型；200 步、固定电刺激、单次 Adam、仅内部 theta、相同输入更新前后/归零恢复/checkpoint 重载均与规格一致。原始图摘要复核与损失方向分别记录，没有可学习输入或动作旁路。实际执行尚无证据。 |
| 资源预检及预算覆盖 | PASS（已有工具证据范围）。登记先于子进程；重依赖导入、载图、构造、数值计算、保存及重载均在父监督时间内。RSS、系统预留、超时及异常可产生失败终态。监控明确是采样监督，不宣称硬隔离。 |
| 不完整证据不得误报成功 | FAIL。父监督只验证退出码和 `status` 字段，错误 schema、失败的数值判据及缺少所有模型/响应产物仍可得到 `passed`，见下方复现。 |
| 首轮实际文件与报告边界 | PASS。`data/trainable_probe_20261017_p1/registration.json` 为 available=3,674,931,200 bytes、reserve=4,294,967,296 bytes、effective budget=0；`result.json` 为 `resource_preflight_failed` / `not_launched`。报告和文档未将其宣称为完整图训练，原始文件未修改。 |

## Validation

- Risk Level：S3（训练持久化、资源监督与完整图集成），仅审查 P1a 相关路径。
- 阅读 AGENTS、项目使命/技术栈/规则、验收工作流、任务交接、Spec 008 的 Plan、P0/P1/P1a 规格、P0 验收和 P1 实施报告。
- 逐段审查 `checkpoint.py`、`resource_guard.py`、`probe.py`、完整 `test_training_probe.py`、TRAINABLE_SNN 文档差异，以及相关既有核心/图身份接口。
- 接受实施者已有 **16 passed** 的定向检查记录，未重复该测试文件或 P0 测试。注意其中 `test_supervisor_requires_clean_exit_and_complete_worker_report` 的 `complete` 案例仅写入 `{"status":"passed"}`，因此没有覆盖真正完整的 P1 证据。
- 针对上述具体缺口执行一次临时目录中的轻量监督复现：子进程退出 0，写入 `schema="wrong-schema"`、`status="passed"`、`zero_restoration_matches=false`、`raw_graph_unchanged=false` 的报告，不生成 checkpoint 或 responses。实际输出为 `reported_status="passed"`、`worker_status="passed"`、`checkpoint_exists=false`、`responses_exists=false`、`reason="worker_exited"`。复现只导入资源监督模块并启动普通 Python 写 JSON，没有载图或导入训练核心；临时目录随后清除。
- 没有重试完整图、修改业务代码、提交、运行全项目检查或操作硬件。用户释放内存后的通知仍是下一次实际完整图运行的前提。

## Issues

1. **阻塞：完成状态没有绑定完整有效的实验结果。** `src/cnsbridge/learning/resource_guard.py:102–108` 仅读取任意字典的 `status`，`probe.run_registered` 直接返回这一判断。它不验证 P1 报告 schema、登记/协议身份、必要数值判据或必要产物；因此能把缺失或自相矛盾的结果写为成功。此问题直接违反 P1a 的“不完整证据不会误报成功”。当前真实首轮是启动前失败，不受这一误报路径影响，但不能据此跳过修复。

## Regression Risks

- checkpoint 本身的严格校验不能弥补探针完成状态未验证 checkpoint 的问题；读取 `result.json` 的后续工具或研究人员会被错误的通过状态误导。
- 16 项小型检查不能验证完整图加载、反向内存、实际增益变化或恢复响应；这些仍属 P1 后续实测。
- 资源监督为采样控制，保留其已有能力边界；本报告不要求扩展到操作系统硬隔离。

## Required Rework

- 在完整图探针的终态确认路径增加与登记绑定的完整性验证；通用监督层可以保留独立职责，但 P1 的最终 `passed` 必须以正确 schema/协议身份、所需有限数值结果和全部成功判据、身份一致且实际存在的 checkpoint/响应证据为条件。失败时保留具体原因，不能仅信任一个 `status` 字段。
- 将现有“complete”测试改为真正完整的有效结果，并补充本次已复现的错误 schema/失败判据/缺失产物拒绝路径。只执行与本次返工相关的检查，不重跑旧核心、全图或硬件。
- 由新的实施 Agent 返工、新的独立验收 Agent 复核；保留本 FAIL 和首轮资源失败文件。P1a 通过后也不能勾选 P1 实际更新或总 Spec AC2。
