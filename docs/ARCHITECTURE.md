# 架构与接口

MaleCNS Lab 将神经活动产生、动作学习和设备转换分开。V2 离线基线冻结连接组和神经元动力学，只训练输出读出；完整图内部增益与读出联合训练也已有实际运行，但训练执行完成不等于控制成功。当前主线为[官方 X2 接入](AGIBOT.md)，研究结果见[实验记录](EXPERIMENT_RESULTS.md)。

## 信号诊断与控制链路

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

`python -m malecns_lab.diagnostics.replay_v2` 记录输入、模拟神经活动、时间身份、覆盖和失败，不读取动作标签、不实例化旧读出、不连接硬件。`report_v2` 只读取已保存记录生成报告，不重复模拟脑。输入驱动 Hz 与实际放电 Hz 分开，缺失群组不作为静默群组统计。详见 [V2 完整脑诊断](MODEL_V2_USAGE.md)。

`python -m malecns_lab.learning.train_v2` 根据独立期望动作和掩码拟合读出；基线神经活动缓存不能用于假装网络内部已经学习。适配器只转换数值，不发送硬件指令；这些学习产物尚未接入 Tello 实飞。V1 重复离线诊断和读出入口已移除，历史产物与来源标识仍保留。

Tello 地面观察和既有飞行试验仍使用旧链路：视觉输入 → `Brain` → `MotorDecoder` → 限幅与试验监督 → Tello。`MotorDecoder` 包含人工设计的权重、基线、巡航与逃逸行为，是实验基线。`experiments.tello.shadow` 也观察这条旧链路，不能加载新候选模型。

## 模块职责

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

`FeatureSchemaV2` 固定模型来源、特征顺序、表示和时间契约。`MaskedRidgeV2` 只拟合有监督的轴，返回 `supported_axes` 和 `prediction_valid`；无效结果和未训练轴不能当成有效动作。标准化只使用训练数据，验证集选参，测试集独立。完整用法见 [V2 学习工具](LEARNING_V2.md)。

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
