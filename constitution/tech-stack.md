# 技术栈与工程约束

项目模式为 `EXISTING`，业务代码目录为 `src/cnsbridge/`。沿用 Python src layout；按用户授权的 [Spec 002](../specs/spec-002-business-layout/spec.md) 将包内业务按控制、学习和 Tello 实验重组。

## 已有技术

| 项目 | 当前事实 |
|---|---|
| 语言与运行时 | Python >= 3.11；项目声明以 `pyproject.toml` 为准 |
| 构建与安装 | setuptools >= 68、wheel，构建后端 `setuptools.build_meta`；pip 与 venv |
| 核心依赖 | NumPy >= 1.24、SciPy >= 1.10、PyYAML >= 6.0 |
| 可选依赖 | `vision`：OpenCV；`data`：pandas、pyarrow；`tello`：DJITelloPy、OpenCV；`dev`：pytest、Ruff |
| CLI | `cnsbridge` → `cnsbridge.cli:main`；download、build-brain、inspect、calibrate、bench |
| 测试与检查 | pytest，测试目录 `tests/`；Ruff 配置在 `pyproject.toml`，目标 Python 3.11，行长 130 |
| 数据与持久化 | 本地连接组、NPZ、JSON、CSV 和视频；没有数据库服务 |
| 服务与部署 | 本地工具与设备实验；没有前后端服务、外部 AI/LLM 服务或既定服务部署流程 |
| 工程设施 | 当前没有 lockfile、CI、Docker 配置或专用构建脚本；没有明确既定的格式化命令流程 |

依据：[pyproject.toml](../pyproject.toml)、[安装说明](../README.md)。不要把当前本机 Python 版本变成比项目声明更严格的最低要求，也不要为补齐表格创建新的工程设施。

## 路径与职责

- [src/cnsbridge/brain/](../src/cnsbridge/brain/) 与 [senses/](../src/cnsbridge/senses/)：连接组仿真与视觉编码，来源哈希文件保持原位。
- [control/](../src/cnsbridge/control/)：动作契约、设备数值适配、旧读出/校准、控制限制。
- [learning/](../src/cnsbridge/learning/)：新解码器、数据准备、离线拟合与评估；依赖 control，反向依赖不允许。
- [experiments/tello/](../src/cnsbridge/experiments/tello/)：设备实验；policy、worker、session 分别负责纯策略、脑计算子进程、共享遥测与清理。
- [drones/](../src/cnsbridge/drones/)：离线仿真；[motor/](../src/cnsbridge/motor/)、`safety.py`、`calibrate.py`：旧导入兼容。详细边界见[架构](../docs/ARCHITECTURE.md)。
- [tools/](../tools/)：原七条命令的兼容启动入口，不放新业务实现。包内模块也可用 `python -m cnsbridge.experiments.tello.trial --help` 等方式调用。
- [configs/](../configs/) 与包内 [defaults.yaml](../src/cnsbridge/defaults.yaml)：实验参数。
- [tests/](../tests/)、[docs/](../docs/)、[examples/](../examples/)：测试、使用与研究文档、离线示例。
- `data/`、`logs/`、`readout*.json`、`.venv/`：本地实验资产及环境，按 [.gitignore](../.gitignore) 留在源码提交之外。
- `AGENTS.md`、`constitution/`、`specs/`、`.ai/`：Agent 治理；不替代业务目录。

## 命令参考与使用条件

以下是供对应任务选用的命令，不是每次修改的必跑清单：

| 用途 | 在项目根目录执行 | 条件 |
|---|---|---|
| 建立环境 | `python3 -m venv .venv` | 仅在尚无所需环境时 |
| 安装开发与实验依赖 | `.venv/bin/python -m pip install -e '.[tello,data,dev]'` | 安装或依赖确需更新时 |
| 查看 CLI | `.venv/bin/cnsbridge --help` | 需要核对命令入口时 |
| 离线示例 | `.venv/bin/python examples/04_shared_motion_decoder.py` | 合成输入，仅演示接口，不连接设备 |
| 定向测试 | `.venv/bin/python -m pytest tests/test_universal_decoder.py` | 共享动作或新读出发生相关改动时，按风险调整范围 |
| 全部测试 | `.venv/bin/python -m pytest` | 只有明确的跨模块风险或任务要求才使用 |
| 定向静态检查 | `.venv/bin/python -m ruff check <本次相关路径>` | 存在相关需要时；尖括号内容须替换 |
| 构建 wheel | `.venv/bin/python -m pip wheel --no-deps . --wheel-dir dist` | 仅打包变更需要时；不是已有独立脚本，也不是本次执行记录 |

文档修改按 S0 阅读差异即可，不默认执行上述命令。源码工程根由 `paths.source_project_root()` 按实际 src layout 定位；配置、模型和日志位置不依赖启动 cwd。实验工具要求源码 checkout 和 editable 安装，不假定普通 wheel 内包含工作区数据。

实体设备步骤见 [Tello 文档](../docs/TELLO.md)，不能将地面采集或实飞当成自动化验证命令。
