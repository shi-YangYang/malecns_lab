# CNSBridge

连接组神经活动到机器人运动目标的可训练解码框架。

项目以果蝇 MaleCNS 连接组的脉冲网络为神经活动来源，学习从活动到运动目标的映射，再由设备适配器转换为控制指令。当前以 Tello 为实验平台，为差速移动底盘保留统一接口。

```text
摄像头 → 视觉特征 → 冻结的连接组网络 → 可训练解码器 → 机身运动目标
                                                       ├─ 无人机适配器
                                                       └─ 差速底盘适配器
```

## 当前状态

- 已有 Tello 地面连接、视频、神经输出观察与分阶段短时实飞工具。既有实飞使用人工设计的读出与监督逻辑。
- 已有共享运动表示、岭回归解码器、设备适配器，以及地面录像、人工标注、离线特征提取和按场景评估的流程。
- 新解码器尚未完成真实数据上的有效性验证，也尚未接入实飞。差速底盘适配器只有离线数学转换，尚未连接实体底盘。
- 尚未证明可靠避障、连接组优于直接视觉控制，或跨设备迁移。MiniFly 是合成测试网络，不能作为真实果蝇连接组的实验结果。

## 安装

需要 Python 3.11 或更新版本。在项目目录运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[tello,data,dev]'
```

项目名称、Python 包和命令行入口为 `CNSBridge` / `cnsbridge`。本地文件夹仍可叫 `FlyDrones`；已有脚本文件名与数据路径保持可用。更新后需重新执行上面的安装命令。

先运行不连接设备的合成示例：

```bash
python examples/04_shared_motion_decoder.py
```

它训练一个读出，并将同一个运动目标转换为无人机指令与差速轮速。示例数据、速度标定和底盘尺寸都是演示值，结果只验证接口。

## 开始第一轮数据实验

将 Tello 固定在地面，关闭手机 App 和其他 Tello 程序，让电脑连接 Tello Wi-Fi：

```bash
python tools/tello_record_dataset.py --scene-id room_a --seconds 30
```

该工具读取遥测、保存视频与时间戳，不起飞、不发送运动指令。先检查一段录像，再按实际画面标注期望动作。完整步骤见 [数据集流程](docs/DATASET.md)。

下载和构建真实连接组：

```bash
cnsbridge download malecns
cnsbridge build-brain --out data/malecns_brain.npz
```

已有 `data/malecns_brain.npz` 时可以直接使用。过滤条件和来源见 [研究与数据](docs/RESEARCH.md)。

## 项目结构

| 路径 | 用途 |
|---|---|
| `src/cnsbridge/brain/` | 连接组、LIF 仿真、MiniFly 测试网络 |
| `src/cnsbridge/senses/` | 图像运动特征与神经输入编码 |
| `src/cnsbridge/motor/` | 旧读出基线、共享动作、可训练解码器与数据评估 |
| `src/cnsbridge/drones/`、`safety.py` | 离线仿真辅助与控制限制；Tello SDK 调用在 `tools/tello_*.py` |
| `tools/` | 地面采集、离线训练、Tello 分阶段实验 |
| `configs/` | Tello 和 MaleCNS 配置 |
| `tests/` | 对应模块的检查 |
| `data/`、`logs/` | 本地数据、模型与实验记录，不作为源码提交 |

保留的命令行工具为 `download`、`build-brain`、`inspect`、`calibrate`、`bench`。`calibrate` 服务于旧读出基线；新解码器使用 `tools/train_motion_decoder.py`。

## 文档

- [架构与接口](docs/ARCHITECTURE.md)：模块、动作坐标和扩展方式。
- [数据集流程](docs/DATASET.md)：地面采集、标注、离线训练与评估。
- [Tello 实验](docs/TELLO.md)：地面检查、已有实飞工具及结果边界。
- [研究与数据](docs/RESEARCH.md)：建模假设、连接组构建、对照实验与引用。

## 来源与许可

本项目基于 [SpikeCalls/FlyDrones](https://github.com/SpikeCalls/FlyDrones) 的代码发展，当前整理聚焦可训练解码器和可复现的实验流程。原代码采用 [MIT 许可](LICENSE)，原版权声明保留。MaleCNS 数据采用独立许可；来源和引用说明见 [THIRD_PARTY.md](THIRD_PARTY.md)。
