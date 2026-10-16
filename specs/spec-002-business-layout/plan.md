# Plan — 业务职责迁移

状态：Done。实施及独立验收均已完成，无返工项；见 [验收报告](acceptance.md)。风险等级 S3：跨控制、数据/模型导入和实验监督多个子系统，涉及 spawn 与旧入口兼容。

## 实施顺序

1. 一名实施 Agent 负责所有耦合业务模块、工具、测试和使用文档，避免并行搬迁导入导致不一致。
2. 将 control/learning 核心移入真实包；旧路径仅转发；新包内部遵循单向依赖。
3. 将 Tello 工具迁入 experiments，抽取 policy、worker、session，修正源码根路径定位。
4. 修正 CLI、示例、测试的导入和 mock 目标，保留旧脚本参数及兼容入口；同步 README、ARCHITECTURE、tech-stack。
5. 运行一次完整离线 pytest（所有相关子系统均受导入影响），以及一个有界的兼容/入口检查；若失败仅修复直接问题并重跑相关失败范围。
6. 实施报告完成后，新的独立验收 Agent 审查移动差异与测试证据；不重复已通过检查。PASS 后协调者维护状态并提交。

## 影响文件与数据流

src/cnsbridge 中动作/学习/控制保护/旧校准模块，七个 tools 入口，引用上述模块的 tests、examples、docs。brain/senses/config/defaults 保持实现，根 data.py 下载入口无需重命名，现有 drones 离线仿真仅修正导入。

视觉 → Brain 的算法与哈希源保持不变；其后 control/legacy 或 learning/decoder → control 动作契约 → 适配器的数值流不变。experiment 负责会话与监督，工具入口只转发。

## 验证计划

- 审查移动/提取差异，确认没有改变常量、门控、清理顺序、持久化契约。
- `.venv/bin/python -m pytest -q` 一次完整离线回归。现有测试使用假 SDK/时钟及纯计算 spawn；不得连接实体设备。
- 补充必要的迁移回归：新旧类/函数身份、从其他 cwd 执行原七个 --help 和两个无起飞入口、CLI help；可以放在一个新测试文件并并入同一次 pytest，禁止执行会连接设备的默认地面命令。
- 用 Git 基线内容核对四个哈希源文件字节不变；测试新旧实现特征身份需有原实现基准，不以“同一新代码重复运行相同”替代迁移兼容证据。
- 独立验收仅在证据缺口或新具体问题时加定向检查，可执行一次 git diff --check；不机械运行 build/lint 或重复整套测试。

## 风险、兼容与回滚

主要风险是层级变化引起路径错误、旧入口函数全局 patch 错位、兼容层循环依赖、spawn 目标不可导入或飞行清理被改变。测试转向真实实现模块；旧导入仅显式导出；源码项目根从稳定来源确定，仓库配置不搬进 site-packages。未迁移实验数据，失败时按 Git 差异修复本次代码，不 reset 用户修改，不重新建立备份目录。

## 分工

协调 Agent 维护 Spec/Decision/交接、组织审查和提交；一名实施 Agent 串行处理业务改造和直接文档；独立验收 Agent 不修改业务代码，只写 acceptance.md。多 Agent 只读调研已提供依赖事实。
