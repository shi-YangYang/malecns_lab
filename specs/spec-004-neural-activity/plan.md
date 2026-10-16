# Plan — 离线神经活动记录与刺激分析

状态：Done。基线 5854ba1，工作区初始干净。定向测试 18 passed、一次三图目检及独立验收 PASS；无返工项，不追加验证。

## 涉及模块

新增 diagnostics，复用 brain/senses 和 learning.prepare.read_recording；只追加 analysis 依赖与当前使用文档。

## 修改顺序

1. 协调者建立 Ready Spec、Decision、交接；只读审计已确认计数索引、时基和空群组问题。
2. 一名实施者负责强耦合的记录、刺激验证/统计、图表/CLI、依赖及定向测试；另一名实施者仅负责使用文档。
3. 实施者运行一次相关诊断测试（其中包含 MiniFly 视频端到端）；保存可供目检的合成结果到忽略目录。
4. 协调者目检图像；全新验收者核对 AC 与 diff，不重复通过的测试。必要返工严格限于本轮问题。
5. PASS 后更新治理状态并 commit，不 push。

## 数据流与接口变化

session/video/frames → 50 ms Retina/Encoder/Brain → activity.npz + CSV/JSON → PNG + index.html。stimuli.csv 仅标记刺激，不影响网络推进。新格式版本独立，不迁移旧数据。

## 预计涉及文件

- 实施 A：src/cnsbridge/diagnostics/、tests/test_neural_activity.py、pyproject.toml、implementation.md。
- 实施 B：docs/NEURAL_ACTIVITY.md、README.md、docs/DATASET.md、docs/ARCHITECTURE.md（仅新增职责与交叉引用）。
- 协调者：本 Spec/Plan、Decision 004、任务交接、specs 索引、constitution 技术栈/路线与项目规则的直接相关说明。
- 独立验收者：acceptance.md。

## 验证计划

- Risk Level：S3，仅因新增持久化格式和可选依赖；验证限制在新诊断子系统，不默认全量测试/构建/lint。
- 已知计数 fixture 验证 Hz、NaN 空组、采样映射；刺激区间/暖机/基线/时基；损坏会话与帧数不匹配；拒绝覆盖；依赖阻断下帮助；MiniFly 短视频端到端保存/加载无 pickle 数据与出图。
- 一次静态图目检，审查四个来源哈希文件未变，独立验收运行 git diff --check。无需真实设备或全量 MaleCNS 仿真。
- 相关测试通过且无具体问题后停止，不重复测试；修复后仅重跑受影响检查。

## 风险与迁移策略

输入 drive 不等于放电，空组不等于静默，视频帧时间不等于采集主机时间；分别记录。人工偏置、视觉近似与单次记录只能形成描述性观察。新命令和输出目录独立，不影响旧链路。

## 分工与交付

协调者维护约束与合并报告；实施 A 汇总实际验证，实施 B 回报文档；独立验收者未参与实现。用户已授权 commit，提交由协调者完成。
