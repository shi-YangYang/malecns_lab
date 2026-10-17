# 清理子批 — 移除被 V2 替代的 V1 离线流程

状态：Ready。用户授权删除无用内容；本子批仅删经只读调用审计确认已被 V2 替代、且不被设备链使用的离线流程。风险 S2。

## 范围

- 删除 `diagnostics/activity.py`、`collector.py`、`reporting.py`、`stimuli.py`。
- 删除 `learning/prepare.py`、`dataset.py`、`train.py`、`decoder.py`，同步调整 `learning/__init__.py`，保留 V2 冻结脑/读出及其数据协议作为对照。
- 删除已不代表项目主线的 `examples/04_shared_motion_decoder.py`。共享 BodyVelocity/适配器的有效示例可另在现行架构文档中简洁保留，不建立 V1 兼容包装。
- 删除 `docs/NEURAL_ACTIVITY.md`、`docs/DATASET.md`；将仍有用的 `experiments.tello.record` 采集说明及会话格式迁入 `docs/TELLO.md`，现行诊断/学习链接转向 V2 文档。
- 删除纯 V1 测试 `test_prepare_motion_dataset.py`、`test_motion_dataset.py`、`test_neural_activity.py`；从 `test_universal_decoder.py` 只删除 V1读出测试，保留动作/适配器有效验证；调整 `test_business_layout.py` 的活跃入口声明。
- 更新 README、ARCHITECTURE、RESEARCH、TELLO 的上述活跃引用；不把尚在实现的 SNN 写成已训练成功。实际必要时更新其他活动文档中的同类引用。

## 保留和边界

不动新 P0/P2a 代码、pyproject、旧实飞链、旧 SimDrone、CLI/数据下载/过滤构建（后续替换子批）、底层 LIF、原始数据/日志/环境、研究对照工具、许可、历史 Spec/Decision/验收。历史文档的旧源码引用是对应旧提交事实，不以删除旧结论掩盖历史；活动入口不能继续推荐已删命令。

本子批在旧离线模块与当前 V2 工具之间已有独立边界，因此可与新核心/新环境并行。协调者负责 constitution 中旧文档/命令入口同步，避免多人编辑治理文件。

## 验收

- [ ] 活跃源码和入口没有被删除模块的导入，V2 控制/诊断保留。
- [ ] 采集命令和有用会话格式仍可查；不遗留空壳兼容文件或误导命令。
- [ ] 保留动作契约和适配器测试；运行 business layout 与对应保留测试的最小相关检查，不重复全脑诊断或全量测试。
- [ ] 新独立验收确认删除范围有依据，无数据/证据丢失。

实施报告 `implementation-cleanup-v1.md`，验收后单独 commit。整个清理目标还包含后续旧主入口与设备链替换，不能以此子批代替全部清理完成。
