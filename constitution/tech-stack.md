# 技术栈与工程约束

项目模式 EXISTING。业务源码位于 src/malecns_lab，沿用 Python src layout；包、分发与 CLI 均为 malecns_lab，不保留旧包入口。已有数据 schema 与研究来源保持原身份。

## 当前技术

| 部分 | 实际实现 |
|---|---|
| Python | >=3.11，setuptools/pip/venv；依赖以 [pyproject.toml](../pyproject.toml) 为准 |
| 神经计算 | NumPy、SciPy；完整 MaleCNS 参考图、LIF 及输入编码 |
| 学习 | .venv-train、PyTorch 2.14.1/MPS、psutil；CPU 与全 GPU 研究分支，不承诺实时性 |
| 图谱 | apps/atlas，Vite + TypeScript + Three.js；npm/package-lock.json |
| 桌面 | apps/desktop，Electron 45.0.0 + TypeScript + Three.js；Spec022 同一冻结MPS控制脑遥测、离线回放与GPU仿真生命周期已实现，两轮原生桌面联调及独立验收通过；本机开发运行，尚无安装包 |
| 本地服务 | 网页 127.0.0.1:8765，Python 刺激任务服务 127.0.0.1:8766；单计算子进程 |
| 设备 | 已有 Tello 工具；官方X2 Sim + MC + AimDK已完成两轮纵向神经反馈MVP实测，使用里程计驱动虚拟标记，真机待适配 |
| 数据 | 本地 NPZ/JSON/CSV/视频，没有数据库；data/logs 不随源码分发 |
| 检查 | tests 下 pytest；前端 npm run build；不默认全量测试或格式化 |

## 模块边界

- brain/、senses/：连接组、数值仿真与感觉编码。
- diagnostics/：离线活动记录与信号报告。
- control/：动作契约、数值适配、旧读出及限制。
- learning/：读出与内部学习、协议、存档与评估；其中 yaw/ 为停止推进的历史研究，数值源码保留。
- drones/：历史目标跟随与单轴简化环境，不是 X2 物理仿真。
- visualization/ 与 apps/atlas/：胞体图谱、已有记录和可调刺激实验；单轴网页与未验证部署接口已清理。
- experiments/tello/：现有 Tello 地面与受限飞行工具。
- configs/、examples/、tests/：配置、数值性能研究脚本与相关测试。

## 后续官方 X2 环境

Apple Silicon Mac 使用 `deploy/x2/` 的独立 linux/amd64 Docker：ROS Humble、官方 MC/Sim v1.0.0-20260522、源码构建 AimDK 1.1.0 的 aimdk_msgs/py_examples，当前采用私有Lighter VM与VirtualGL/Mesa/Venus/MoltenVK接入Mac GPU渲染、noVNC显示；旧llvmpipe CPU仿真保持停止。macOS MPS完整脑与官方运动链路已完成两轮纵向反馈实测，模型与官方组件未改；传感器为里程计虚拟标记，连续实时控制与现场版本匹配待验证，见 [AGIBOT](../docs/AGIBOT.md)。保留脑模型在 macOS 运行的能力，不把环境可用写成闭环成果。

## 本地约束

本地根目录为 /Users/yang/project/malecns_lab。数据和环境保持原样；运行前核实实际资源。Noria 正在运行，不可为本项目停止其服务或重启 Docker。后续完整脑任务沿用用户要求的 1 GiB 系统预留并重新测量资源，不并行启动多个完整脑。

旧 run 的源码身份可能与当前 checkout 不同，不能跳过 hash 校验恢复；已有 sources 快照与模型保留。数值与研究说明见 [MODEL_V2](../docs/MODEL_V2.md)、[MPS_TRAINING](../docs/MPS_TRAINING.md) 和[历史结果](../docs/EXPERIMENT_RESULTS.md)。
