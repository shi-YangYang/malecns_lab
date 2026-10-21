# Spec — 按职责重构业务目录

状态：Done。业务目录重构已完成，独立验收 AC1–AC6 全部 PASS；见 实施报告和验收报告。用户已明确授权工程结构调整，沿用此前完成后本地 commit 的交付要求。

后续状态：本任务当时保留的兼容入口由 [Spec 003](../spec-003-remove-compat/spec.md) 移除；以下需求与验收记录描述本任务完成时的行为。

## 背景

基线为 `8546f6c`。`motor/` 同时承担动作契约、设备数值转换、人工读出、新神经读出和数据评估。`tools/` 中包含实际业务实现，试飞模块通过兄弟脚本导入共享逻辑；736 行试飞脚本混合策略、子进程、输入监督、控制、结果报告和 CLI。

## 目标

把可复用业务实现纳入可导入的 Python 包，建立控制、学习、Tello 实验的明确边界，拆分试飞中已有的独立职责。旧使用命令继续可用。

## 非目标

不改变飞行算法、限幅、门控、时序、清理顺序、模型结构、学习方法或实验结论；不增加新硬件能力或 CLI 产品功能；不重命名已清晰的 brain/senses；不改变依赖、数据格式或实验资产；不操作无人机、不下载或重建连接组。

## 当前行为

真实飞行、地面录像和离线数据准备直接从 `tools/*.py` 运行。测试部分以脚本路径或 sys.path 注入导入代码。Brain 的四个实现文件字节参与特征来源哈希；默认配置与数据路径按源码项目根解析。

## 目标行为

建议组织如下，允许实施者在同一职责内确定文件细节，但不得改变包边界或行为：

```text
src/cnsbridge/
├── brain/                  # 保留原模型实现
├── senses/                 # 保留原感知实现
├── control/                # 动作契约、设备换算、旧读出和控制保护
│   ├── body.py
│   ├── command.py
│   ├── adapters.py
│   ├── legacy.py
│   ├── calibration.py
│   └── safety.py
├── learning/               # 新读出与离线数据/评估流程
│   ├── decoder.py
│   ├── dataset.py
│   ├── prepare.py
│   └── train.py
├── experiments/tello/      # 地面、录像、观察和受限试飞
│   ├── check.py
│   ├── hover.py
│   ├── shadow.py
│   ├── record.py
│   ├── trial.py
│   ├── policy.py           # RC转换、时序门控、持续时间规则
│   ├── worker.py           # Brain子进程
│   └── session.py          # 共享遥测/起飞检查/降落清理
├── motor/                  # 仅旧导入兼容层
├── drones/                 # 保留现有离线仿真辅助
├── config.py / defaults.yaml
├── cli.py / data.py        # 保留CLI和下载入口
└── ...                     # 必要的项目路径解析与旧导入兼容入口
tools/*.py                  # 保留原七条命令，转调真实业务入口
```

## 功能需求

1. control 拥有 BodyVelocity、FlightCommand、设备数值适配、MotorDecoder、旧校准与 SafetyGovernor；learning 依赖 control 的动作契约，control 不反向依赖 learning。
2. learning 接收原 motor/neural.py、dataset.py 及两个离线工具实现。不拆分尚无独立维护价值的微小包装。
3. Tello 工具实现在 experiments/tello，消除业务代码对 `tools` 或无包兄弟脚本的导入；试飞至少拆出纯策略、计算子进程和共享会话清理。必要时将汇总报告拆到 reporting.py，但保持先降落、再停止进程与写日志的顺序。
4. tools 原文件名和命令参数保留为简短启动入口，无重复业务实现。旧 `cnsbridge.motor.*`、`cnsbridge.safety`、`cnsbridge.calibrate` 的已存在符号保留显式兼容导出；不通过运行 exec、修改 sys.modules 或复制实现伪造兼容。
5. 当前内部调用、示例与测试使用新业务路径；包不反向依赖兼容层。更新测试导入及必要 patch 目标，不删掉原断言或放宽时序/安全验证。测试目录保持原位，避免无收益重排。
6. 保留源码工程根下的 configs、data、logs、readout 路径语义；从任意 cwd 用绝对工具路径运行仍有效。共享项目路径定位有明确依据，不能因搬迁误认 src 或 site-packages 为仓库。
7. README、ARCHITECTURE 和 constitution 技术栈同步真实目录；当前治理规则允许这次已授权调整，完成后新目录成为后续约束；保留上一任务的历史事实。

## 边界条件

- 不连接或构造真实 SDK；无 --takeoff 的飞行入口和所有 --help 必须维持不连接语义。地面程序执行只用于人工任务，本次仅检查帮助和离线逻辑。
- multiprocessing spawn 子进程可导入新真实模块；无需把 tools 加进 sys.path。
- 包的低层导入不急加载可选 OpenCV、DJITelloPy。保持其原本的延迟导入能力。
- 新逻辑是职责提取和导入路径修正；如发现无关历史缺陷，只记录不修复。

## 技术约束与兼容性

仍用 Python >=3.11、setuptools src layout，版本和依赖不变。保留所有 `flydrones.*` 格式、JSON/NPZ/CSV字段、标签时间基准和数据拒绝覆盖行为。`brain/brain.py`、`brain/lif.py`、`senses/retina.py`、`senses/encoder.py` 保持字节不变，RecordingBrain 的哈希计算和特征顺序不变。BodyVelocity/FlightCommand 坐标、单位及符号不变。

原 Python 导入只通过兼容层转发到同一个类/函数对象；新业务代码使用新路径。兼容入口是有外部调用依据的迁移策略，不是新的业务抽象层。

## 验收标准

- [x] AC1：control / learning / experiments/tello 实现归位，tools 为兼容启动入口，原主要职责混杂得到实际拆分。
- [x] AC2：低层依赖方向正确，不再导入兄弟 tools；旧公开导入返回相同实现对象，无副本、循环导入或可选SDK急加载。
- [x] AC3：现有离线测试全部保持实质断言并通过，覆盖控制门控、清理、超时、spawn、数据和学习兼容。
- [x] AC4：旧七个工具的 --help 与无 --takeoff 入口、既有 CLI 可离线使用；从非项目 cwd 运行正常，参数和默认路径语义不变。
- [x] AC5：数据格式/特征身份和飞行数值语义不变，四个哈希文件字节不变，实验资产未改动。
- [x] AC6：文档、决策、实施及独立验收记录与实际布局一致，无伪造实飞结论。

## 待确认问题

无阻塞问题。用户已授权业务目录重构；职责与兼容边界由现有代码确定，无新增产品/模型/设备决策。调研所得实际依赖约束足以实施，不需重复请求目录重构许可。

## 交付

独立验收 PASS 后更新完成状态并创建中文 refactor 本地提交；不推送。必要的后续实飞验证不属于本次离线目录重构。
