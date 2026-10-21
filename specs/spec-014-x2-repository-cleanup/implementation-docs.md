# 文档清理实施记录

范围：README、docs 与历史 specs。未修改业务数值源、原始数据、权重、日志、环境或 Docker；未执行训练或完整脑计算。

- 删除 213 份重复或过期 Markdown：阶段 implementation/acceptance/rework/plan 流水、已结束清理计划、移除功能的操作说明。保留独有科学协议和真实运行/失败报告；未提交删除文件不声称有 Git 副本。
- 新增 `docs/EXPERIMENT_RESULTS.md`，汇总完整图构建、GPU 训练边界及单轴失败；保留原始 run/checkpoint 路径和固定模型验证报告。
- 合并两份 checkpoint 手册为 `docs/CHECKPOINTS.md`；迁移说明并入 `docs/ARCHITECTURE.md`。docs 数量从 28 减至 25，历史 specs 大幅精简。
- README、图谱文档和索引仅列当前刺激响应能力。官方 X2 + Mac linux/amd64 Docker 明确为下一阶段待验证方案；旧单轴不再是前置。
- 历史审计中的旧 `src/cnsbridge/` 链接改为标有原始路径/行号的历史定位；不把旧行号冒充当前实现。
- 保留 `mps_contract.NUMERIC_PRECEDENT` 引用的 `specs/spec-008-trainable-snn/mps-fullgraph-run-20261019-r1.md` 原路径原文，以及它直接链接的核级验收报告；该报告与 HEAD 逐字节相同。没有修改绑定训练源码。

验证为文档静态检查：本次负责范围（README/docs/历史specs）的本地 Markdown 文件链接检查为 0 个悬空目标；没有残留已删 yaw 部署、导出、信号探针的当前操作命令。检查后仅修正发现的 ARCHITECTURE/CPU runner 过期状态措辞和历史描述，新增链接均指向已存在的实验结果页。未运行测试、构建或训练。最终仓库链接与代码检查由协调者汇总，不重复相同验证。
