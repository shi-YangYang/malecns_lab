# Implementation Report — P3b 两帧三轴联合更新工具

## Summary

已实现冻结 P3b 的独立入口、受监督 worker 与不导入 Torch 的父证据校验器。只增加本子批模块、定向测试和运行说明；既有脑核心、策略、数据、参数存档和历史工具未修改。尚未运行完整图，没有实际动作学习结论；由新的独立验收 Agent 判断本软件是否满足协议，协调者提交后才运行一次实际实验。

## Files Changed

- `src/cnsbridge/learning/joint_probe.py`：新目录事前登记、轻量父进程、原始监督终态保留和 CLI。
- `joint_probe_contract.py`：冻结协议、源码/文件绑定、数据精确选帧和登记校验。
- `joint_probe_worker.py`：连续两帧梯度、一次联合 Adam、全参数裁剪、五分支和先释放旧图再生产重载。
- `joint_probe_identity.py`：无模型的生产图/策略/感觉/固定背景/时钟/读出身份校验。
- `joint_probe_evidence.py`：NumPy 独立参数、梯度、loss、动作、存档、状态摘要与资源验据，三层结论独立。
- `tests/test_joint_probe.py`：显式数值 fixture 和受损/阴性/进程边界检查。
- `docs/JOINT_PROBE.md`：实际入口、固定计算、退出码、证据与失败保留位置。

## Important Decisions

- 精确使用已封存数据的 combined 0/1 两帧同帧限幅教师目标；不选下一帧、执行动作或无效帧替代。
- 所有实际脑构造和 Torch 操作只在 worker；父进程仅流式文件摘要、小型 metadata、数据和 NumPy 证据操作。生产入口不接受数值 fixture。
- 正则初值/梯度独立检查为零；保留裁剪前后全部梯度、范数/系数和前后参数，防止把正则梯度说成动作任务更新。
- 五分支保存独立 NumPy 副本及预处理/末状态全数组摘要；重载前不保留旧模型、优化器或计算图引用。存档只称为完整策略参数，不作续训承诺。
- 先保存 `supervisor_result.json` 再独立验据；协议完整和数值/功能阴性分别报告，loss 上升不使有效负结果被丢弃。资源失败、注册失败、缺失或无效证据保持失败。

## Validation

- Risk Level: **S3**（新训练/持久化证据/资源交互工具），范围限本协议新检查。
- Checks Performed: `PYTHONPATH=src .venv-train/bin/python -m pytest tests/test_joint_probe.py -q`
- Result: **36 passed, 1 warning in 1.14s**；warning 为现有 PyTorch sparse CSR beta 提示。首个工具调用输出在 1 秒 yield 时只返回部分点号、未取得终态，因此仅为取得明确退出结果重跑一次；明确通过后未追加旧套件或全图检查。
- 检查覆盖真实小电路两帧联合更新、跨帧同一状态与梯度、一次 Adam、五分支恢复/重载、零梯度阴性、loss 增加、全参数裁剪及非有限拒绝、缺失/篡改/形状dtype证据拒绝、注册改变拒绝、同帧教师选取、父进程不导入 Torch、严格受监督启动和资源 CSV 复核。
- 检查了本次 diff/新增文件及改动范围。未运行 full graph、真实数据采集、硬件、旧通过套件、构建或全量 lint；没有服务/Docker/Noria 操作。

## Known Limitations

100 ms 完整图计算图的实际内存、总耗时、内部梯度和动作效应仍未知。本次检查只证明显式数值 fixture 与工具行为，不能作为完整图 P3b 结果。固定两帧都是正标签，不验证负方向、长轨迹闭环、泛化或真实果蝇学习；整体 AC3–AC6 未完成。

采样资源监督并非硬内存隔离。状态证据是各分支预处理/末状态摘要，不是逐步全节点轨迹。父复核验证文件和数值一致性，不通过再执行模型证明神经轨迹；生产 full-reference loader 与绑定源码负责实际图加载。

## Remaining Questions

无新的规格选择。待独立软件验收；之后由协调者使用 `docs/JOINT_PROBE.md` 的固定入口、新目录并按预算运行一次，保留阴性或失败结果。
