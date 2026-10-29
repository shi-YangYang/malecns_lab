# 架构与接口

MaleCNS Lab 将神经计算、行为读出、执行保护和设备控制分开。当前产品入口是 [X2 桌面](../../apps/desktop/README.md)，独立网页图谱与历史离线研究另有入口。以下描述当前源码职责；各版本的实际训练和动作结果统一见 [history.md](../../history.md)，不能将接口接通视为行为验收通过。

## 当前 X2 桌面链路

```text
Electron 开始/停止与任务按钮
  → runtime/x2/behavior：会话、候选、GPU 与进程管理
  → learning/x2_behavior_session：请求、真实反馈、神经决策与收尾
  → senses/x2_behavior：44 通道任务/身体/反馈工程编码
  → learning/x2_frozen_history_policy：完整冻结脑与历史读出
  → control/x2_* + experiments/x2_behavior_bridge：行为准入、官方技能及停稳
  → AimDK / MC → MuJoCo X2 → 真实仿真状态反馈

同一脑实例 → x2_behavior_telemetry / x2_neural_state
  → 有界 JSON 与二进制旁文件 → Electron 身份/哈希校验
  → src/neural-state.ts / brain.ts / workbench.ts → 全细胞状态与群组曲线
```

当前 B 候选读取 L1/L2 电流特征及历史，不能套用早期“1,314 DN / 2,628维输出”的设计描述。模型、编码、历史算法与参数身份由候选文件及来源快照约束；当前桌面配置在 `data/x2_behavior_desktop.json`，不随 Git 分发。其旧复合任务开发评分仍未达门槛，不把后续独立动作要求回写成旧成绩。

桌面使用 continuous 会话与 automatic-start；开始后自动准备仿真并确认真实画面。主动作完成后执行层关闭同请求进一步技能准入，后台消费真实反馈并确认停稳；收尾可预选一项，准备完成后才发下一请求。“结束互动”入脑，顶部停止/关闭窗口是独立保护。官方 MC 承担平衡、步态与预设轨迹，执行层负责有界运动和目标/停稳判断。

实时显示已扩到全部166,700个细胞的窗口平均V、ΔV、Hz；实际几何为674,411个原始来源点与21,952条抽样连接示意。节点亮度按ΔV，连接亮度按突触前Hz；零基线、柔光及平滑不是新神经测量。独立 `npm run brain` 仍是两个历史电流窗口的展示，没有机器人控制。


## 技术栈

| 部分 | 约定 |
|---|---|
| Python | >=3.11，setuptools/pip/venv，依赖以 [pyproject.toml](../../pyproject.toml) 为准 |
| 神经计算与学习 | NumPy/SciPy；独立 `.venv-train` 中 PyTorch 2.14.1/MPS、psutil；CPU 参考与 GPU 研究分支分别记录数值身份 |
| 网页脑图 | `apps/atlas`，Vite + TypeScript + Three.js，npm/package-lock.json |
| 桌面 | `apps/desktop`，Electron 45.0.0 + TypeScript + Three.js；本机开发运行，尚无签名安装包 |
| 本地服务 | 脑图 `127.0.0.1:8765`，刺激任务服务 `127.0.0.1:8766`；单计算子进程 |
| X2 | 官方 MC/Sim v1.0.0-20260522，AimDK 1.1.0；私有 Lighter + VirtualGL/Mesa/Venus/MoltenVK，容器 linux/amd64，noVNC 主机端口 6081 |
| 数据 | NPZ/JSON/CSV/媒体，本地 `data/`、`logs/`；无数据库，不随源码分发 |
| 检查 | Python pytest、前端自身 npm 脚本；按改动范围执行，不默认全量测试或构建 |

本机私有 GPU 安装在 `data/x2_sim/gpu_probe/runtime-rebuild`。部署源码放 `deploy/x2`；新 Dockerfile 未重新构建验证，已有 GPU 安装的实测不能替代新部署验收。不得为本项目停止 Noria、重启 Docker 或切换默认 context。完整脑任务按当前授权与预算运行，至少保留 1 GiB 系统余量，不并行启动多个完整脑。

## 独立诊断与历史设备链路

```text
地面录像或标准刺激 → 工程视觉编码 → 完整 MaleCNS 参考图 → 神经活动与因果历史
                                                              ↓
                                                    信号报告 / 掩码读出基线
                                                              ↓
                                                        BodyVelocity
                                                       ↙            ↘
                                          FlightCommandAdapter   DifferentialDriveAdapter
                                              FlightCommand             WheelSpeeds
```

`python -m malecns_lab.diagnostics.replay_v2` 记录输入、模拟神经活动、时间身份、覆盖和失败，不读取动作标签、不实例化旧读出、不连接硬件。`report_v2` 只读取已保存记录生成报告，不重复模拟脑。输入驱动 Hz 与实际放电 Hz 分开，缺失群组不作为静默群组统计。详见 [V2 完整脑诊断](../research/offline.md#model-v2-usage)。

`python -m malecns_lab.learning.train_v2` 根据独立期望动作和掩码拟合读出；基线神经活动缓存不能用于假装网络内部已经学习。适配器只转换数值，不发送硬件指令；这些学习产物尚未接入 Tello 实飞。V1 重复离线诊断和读出入口已移除，历史产物与来源标识仍保留。

Tello 地面观察和既有飞行试验仍使用旧链路：视觉输入 → `Brain` → `MotorDecoder` → 限幅与试验监督 → Tello。`MotorDecoder` 包含人工设计的权重、基线、巡航与逃逸行为，是实验基线。`experiments.tello.shadow` 也观察这条旧链路，不能加载新候选模型。

## 模块职责

工程采用 Python `src` 布局，界面、运行管理、领域计算、部署和科研资产分别存放：

```text
apps/
  atlas/                 网页脑图与刺激响应
  desktop/               Electron 桌面与界面
src/malecns_lab/
  runtime/x2/            GPU 仿真生命周期与桌面会话管理
  brain/ senses/         神经计算与感觉编码
  learning/ control/     学习、行为契约与运动适配
  experiments/           设备实验与容器内协议入口
  diagnostics/           离线诊断
  visualization/         图谱导出与本地任务服务
  drones/                历史简化环境
deploy/x2/
  scripts/ keys/         GPU 容器脚本与公共签名材料
tests/x2/                X2 子系统离线检查
docs/
  architecture/          工程结构、数据契约与神经模型
  guides/                独立图谱和历史 Tello 使用手册
  research/              训练、离线实验与研究边界
  tasks/（仅任务需要时创建） 每任务最多一份，阶段结果完成后精简归档
configs/ examples/       配置与离线示例
constitution/ .ai/prompts/  项目使命与精简交接
history.md               已完成阶段与重要失败记录
data/ logs/              本地科研与运行资产，不入库
```

`runtime/x2/desktop.py` 管理私有 GPU 运行资源，`skills.py` 管理有界官方技能会话，`behavior.py` 管理候选脑与桌面通信。Electron 通过 `python -m malecns_lab.runtime.x2.behavior` 启动后端；容器中的 ROS/AimDK 协议仍属于 `experiments`，模型训练仍属于 `learning`。运行管理不改变官方 SDK 或神经算法。

目录迁移只更新当前源码与入口；旧运行记录的源码路径和 hash 保持原值。新运行登记新路径，复现旧运行使用其封存源码，不通过改写历史身份来续跑。

| 旧源码或工具位置 | 当前职责位置 |
|---|---|
| `experiments/x2_desktop.py` | `runtime/x2/desktop.py` |
| `experiments/x2_behavior_desktop.py` | `runtime/x2/behavior.py` |
| `experiments/x2_skill_host.py` | `runtime/x2/skills.py` |
| `tests/test_x2*.py` | `tests/x2/test_x2*.py` |
| `deploy/x2/*.sh`（容器脚本）、`preset-compat.py` | `deploy/x2/scripts/` |

旧阶段的实测命令和 hash 描述当时源码，已汇总到 [history.md](../../history.md)；从当前 checkout 操作时使用上表新入口。预设兼容工具的构建、动作与取消验证分阶段记录在 history.md；目录迁移本身不改变其资格。

| 模块 | 职责 |
|---|---|
| `brain/reference.py` | 完整神经元参考图构建、原始来源与边界账本 |
| `brain/connectome.py` | 旧设备链连接矩阵、群组选择与过滤图构建 |
| `brain/lif.py`、`brain/brain.py` | 冻结基线的 LIF 动力学、外部刺激、脉冲计数与发放率 |
| `senses/vision_v2.py`、`encoding_v2.py` | V2 工程视觉特征、输入群组映射与因果历史 |
| `senses/retina.py`、`encoder.py` | 旧设备链的光流、亮度、膨胀特征及神经输入编码 |
| `diagnostics/sources_v2.py` | 地面会话、独立视频、标准刺激及预登记片段读取 |
| `diagnostics/runtime_v2.py`、`replay_v2.py`、`report_v2.py` | 完整图重放、活动记录、覆盖统计与报告 |
| `learning/contracts_v2.py`、`dataset_v2.py` | 版本化特征/标签契约、来源、掩码与场景族隔离 |
| `learning/decoder_v2.py`、`train_v2.py` | 冻结脑的掩码岭回归、独立拟合与验证集选参 |
| `learning/experiment_v2.py` | 研究协议登记、图条件比较、失败和预算记录 |
| `control/body.py`、`adapters.py` | 统一运动目标、无人机与差速底盘转换 |
| `control/legacy.py`、`command.py` | 既有人工读出和归一化飞行指令 |
| `control/safety.py`、`calibration.py` | 独立控制限制与旧读出校准 |
| `experiments/tello/check.py`、`hover.py`、`shadow.py`、`record.py` | SDK 检查、悬停、旧神经输出观察、地面录像 |
| `experiments/tello/trial.py`、`policy.py`、`worker.py`、`session.py` | 既有试飞监督、纯 RC 策略、脑子进程、遥测及清理 |
| `drones/follow_env.py` / `follow_scenes.py` | 三轴目标跟随环境与隔离场景；简化速度响应，不是 Tello 标定动力学 |

依赖方向为 `learning → control`；控制契约不依赖学习流程。诊断复用自身 `sources_v2` 与 `brain`、`senses`，不执行动作解码。包内实现直接导入所属模块，实验、诊断与学习命令通过 `python -m` 运行。

这些模块位于 `src/malecns_lab/`。路径解析验证源码 `pyproject.toml` 和包布局；从其他目录以 editable 安装环境启动不会改变工作区资产位置。完整参考工具默认读取 `data/malecns_v2/full_reference.npz`；旧设备链按 `configs/malecns.yaml` 读取旧图及校准文件。缺少所需资产时停止，不生成替代网络；不能把旧过滤图当作完整图实验。

## 统一运动表示

`BodyVelocity` 的轴顺序固定为 `(vx, vy, vz, wx, wy, wz)`，采用右手机身坐标系 FLU：

| 分量 | 正方向 | 单位 |
|---|---|---|
| `vx`、`vy`、`vz` | 前、左、上 | m/s |
| `wx`、`wy`、`wz` | 绕对应轴的右手旋转；正 `wz` 为左转 | rad/s |

这些是期望速度，不是实测速度。V2 冻结脑基线只监督 `vx/wz`，其他轴是未训练占位，必须保留支持掩码；Spec 008 的三轴目标跟随另行验证 `vx/vz/wz`。

无人机适配器支持三个平移轴和偏航；差速底盘支持前进和偏航。不支持的非零分量会报错。适配结果包含限幅后的目标与饱和轴。差速底盘轮速超限时等比例缩放双轮，保持目标曲率。

下面仅演示同一运动目标的数值转换，不训练网络、不连接设备，比例和底盘几何也不是实测标定：

```python
from malecns_lab.control.body import BodyVelocity
from malecns_lab.control.adapters import DifferentialDriveAdapter, FlightCommandAdapter

motion = BodyVelocity(vx=0.1, wz=0.2)
aircraft = FlightCommandAdapter(
    forward_scale_mps=1.0, lateral_scale_mps=1.0,
    vertical_scale_mps=0.5, yaw_scale_rad_s=1.0,
)
ground = DifferentialDriveAdapter(
    wheel_radius_m=0.05, track_width_m=0.25, max_wheel_rad_s=12.0,
)
print(aircraft.adapt(motion).command)
print(ground.adapt(motion).command)
```

既有 `FlightCommand` 横移和偏航正方向为右，适配器负责转换符号。Tello RC 并非精确速度接口，速度比例需要实测。共同数学接口不证明跨设备迁移，实体底盘还需轮速闭环与设备监督。

## 冻结脑学习基线与来源

`FeatureSchemaV2` 固定模型来源、特征顺序、表示和时间契约。`MaskedRidgeV2` 只拟合有监督的轴，返回 `supported_axes` 和 `prediction_valid`；无效结果和未训练轴不能当成有效动作。标准化只使用训练数据，验证集选参，测试集独立。完整用法见 [V2 学习工具](../research/offline.md#learning-v2)。

图、参数、编码、特征顺序或时间身份变化后，必须重新生成相应活动；旧缓存不能冒充新网络的响应。模型与数据写入要求新路径，保留原图、录像、标签和运行记录。地面采集使用既有 `flydrones.ground_recording` 标识；V2 活动和读出有独立格式，不通过改名伪装兼容。

## 仿真时间与运行时间

冻结基线的连接矩阵为 CSC 格式 `W[post, pre]`，以脉冲、延迟缓冲、膜电位和不应期推进。V2 视频每帧推进 50 ms 仿真；实际耗时单独记录。控制线程发包频率不代表模型更新频率，也不能把固定地面录像的诊断解释成飞行闭环。

增加设备需分别实现数值适配与硬件控制；增加学习方法需保留来源、时间、动作单位及支持掩码。软件检查、离线任务效果和实体能力分别验证，测试不启动实机运动。

## 环境迁移与历史来源

项目、发行包、Python 包和 CLI 统一为 `malecns_lab`，不保留 `cnsbridge` 兼容入口。本机根目录为 `/Users/yang/project/malecns_lab`。移动目录后，既有虚拟环境的启动脚本、激活脚本和 editable 安装可能仍指向旧路径；应修复路径，并用各环境自己的 Python 更新安装：

```bash
.venv/bin/python -m pip install --no-deps --no-build-isolation --no-index --disable-pip-version-check -e .
.venv-train/bin/python -m pip install --no-deps --no-build-isolation --no-index --disable-pip-version-check -e .
```

上述命令假定依赖和构建后端已安装。磁盘中的 `cnsbridge.*`、`flydrones.*` schema 和历史来源标识仍保留，不能机械替换存档字段。路径或源码变化会改变实验身份，格式可读不代表可以在新源码下精确续跑；复现旧实验需匹配的源码、环境与原始资产，不改写原 run 摘要绕过校验。Git、许可、模型、数据和日志独立保留。


文档整理仍须遵守源码身份约束。`brain/trainable.py` 和 `senses/encoder.py` 的旧注释分别提到 `docs/TRAINABLE_SNN.md`、`docs/RESEARCH.md`，现在对应 `docs/architecture/neural-model.md`、`docs/research/boundaries.md`。前者的整文件哈希参与当前冻结 B 候选校验，因此保留源码原字节，不为更新注释而破坏既有模型准入，也不改写封存哈希。
