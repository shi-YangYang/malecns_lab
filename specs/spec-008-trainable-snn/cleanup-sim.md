# 清理子批 — 移除已被目标跟随环境替代的旧房间仿真器

状态：Done，独立[验收 PASS](acceptance-cleanup-sim.md)。用户授权清理无用实现；P2a 环境已独立验收并提交 `365fab0`。

## 依据与范围

只读调用检索表明 `drones/base.py` 的 Drone 抽象和 `drones/sim.py` 的 SimDrone/Room/Box/RayCamera 仅通过包导出供 `tests/test_senses.py` 的一个视网膜回归测试使用；没有当前运行命令或硬件链依赖。目标跟随已有实际采用 BodyVelocity 的独立环境，继续保留旧 FlightCommand 房间仿真器会留下第二套未使用的运行入口。

- 删除 `src/cnsbridge/drones/base.py`、`sim.py`，更新 `drones/__init__.py` 仅描述现行仿真职责，不建立兼容别名或新的包装。
- 调整 `tests/test_senses.py`：保留方向通道检查；旋转与逼近的视网膜回归意图保留，优先用已验收 FollowEnvironment 生成图像。如果该标记物场景不能覆盖原测试的纹理特性，应解释限制并采用明确的局部数值刺激，不编造仍验证了原房间仿真。
- 定向核对活跃文档／代码是否仍推荐旧类，只处理本子批相关引用。历史审计/Spec 不批量改写。

## 边界

不改 Retina 算法、完整脑、训练核心、P1 工具、已验收 FollowEnvironment、CLI/数据构建、设备链或已有资产。保留 V2 冻结脑的科学对照。不得为让检查通过而调整算法或无依据降低原测试语义。

## 验证与交付

风险 S2（删除包内旧公共类并迁移感知测试的图像来源）。定向检查现行 `tests/test_senses.py` 和 `tests/test_follow_environment.py` 的导入／运动路径，不运行训练或硬件。P2a 的业务实现未变，不机械重复其全部测试；只有实际导入或包初始化变化涉及的最小检查。记录 implementation-cleanup-sim.md，由新独立验收者输出 acceptance-cleanup-sim.md 后单独 commit。

- [x] 旧仿真器、抽象与导出均删除，无空壳兼容。
- [x] 无活跃调用断裂；视网膜回归意图与替代刺激的证据明确。
- [x] 新目标环境、V2 和设备链未被改动；检查与范围相称。

这是第二个清理子批，不代表旧主入口／硬件链已完成替换，也不代表整体学习目标完成。
