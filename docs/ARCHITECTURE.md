# 架构与接口

CNSBridge 将神经活动产生、动作学习和设备转换分开。现阶段冻结连接组与神经元动力学，只训练输出读出。

## 两条实验链路

新训练链路：

```text
地面录像 → Retina → InputEncoder → Brain → 神经活动数据
                                   人工期望动作 ─┘
                                          ↓
                                RidgeMotionDecoder
                                          ↓
                                   BodyVelocity
                                  ↙            ↘
                     FlightCommandAdapter   DifferentialDriveAdapter
                         FlightCommand             WheelSpeeds
```

适配器只转换数值，不发送硬件指令。新训练结果尚未接入 Tello 实飞。

保留的 Tello 地面观察和飞行试验使用旧链路：视觉输入 → `Brain` → `MotorDecoder` → 限幅与试验监督 → Tello。`MotorDecoder` 包含人工设计的权重、基线、巡航与逃逸行为，是实验基线。`experiments/tello/shadow.py` 也观察这条旧链路（原命令 `tools/tello_shadow.py` 保留），不能用于加载新候选模型。

## 模块职责

| 模块 | 职责 |
|---|---|
| `brain/connectome.py` | 连接矩阵、细胞类型与侧别、群组选择、MaleCNS 构建与保存 |
| `brain/lif.py`、`brain/brain.py` | LIF 动力学、外部刺激、脉冲计数与发放率 |
| `brain/synthetic.py` | MiniFly 合成网络，供离线检查 |
| `senses/retina.py`、`senses/encoder.py` | 光流、亮度、膨胀特征及神经输入编码 |
| `learning/decoder.py` | 特征模式、观测、解码协议和岭回归基线 |
| `control/body.py`、`control/adapters.py` | 统一运动目标、无人机与差速底盘转换 |
| `learning/dataset.py` | 数据校验、合并与按场景评估 |
| `control/legacy.py`、`control/command.py` | 既有人工读出和归一化飞行指令 |
| `control/safety.py`、`control/calibration.py` | 独立控制限制与旧读出校准 |
| `learning/prepare.py`、`learning/train.py` | 标注录像的神经特征提取、离线拟合与评估入口 |
| `experiments/tello/check.py`、`hover.py`、`shadow.py`、`record.py` | SDK 检查、悬停、神经输出观察、地面录像 |
| `experiments/tello/trial.py` | 试飞入口、实时监督和记录；只在主进程发送 RC |
| `experiments/tello/policy.py` | 纯 RC 限幅、持续时间约束与巡航门控 |
| `experiments/tello/worker.py` | 可由 spawn 导入的独立脑计算进程，不访问 SDK |
| `experiments/tello/session.py` | 新鲜遥测、输入监督、起飞检查和降落清理 |
| `drones/sim.py` | 离线仿真辅助，不用于宣称实飞效果 |

依赖方向为 `learning → control`；控制契约不依赖学习流程。实验模块组合脑、感知与控制，七个 `tools/*.py` 仅调用真实模块的 `main`。旧 `motor.*`、`safety`、`calibrate` 显式导出相同对象，用于迁移兼容；包内实现和测试使用新路径。

试飞依旧先尝试降落，再关闭计算进程，最后写入缓冲日志。路径解析验证源码 `pyproject.toml` 和 `src/cnsbridge` 布局；从 `server/` 等目录用绝对工具路径启动不会改变资产位置。脑与视觉的四个来源哈希文件及数据格式均保持不变。

这些模块位于 `src/cnsbridge/`。配置由 [defaults.yaml](../src/cnsbridge/defaults.yaml) 和 [configs](../configs/) 中的覆盖项组成；嵌套字典深度合并，`terms` 块整体替换。

## 统一运动表示

`BodyVelocity` 的轴顺序固定为 `(vx, vy, vz, wx, wy, wz)`，采用右手机身坐标系 FLU：

| 分量 | 正方向 | 单位 |
|---|---|---|
| `vx`、`vy`、`vz` | 前、左、上 | m/s |
| `wx`、`wy`、`wz` | 绕对应轴的右手旋转；正 `wz` 为左转 | rad/s |

这些是期望速度，不是实测速度。第一轮数据实验只学习 `vx` 和 `wz`，其余轴标为零。

无人机适配器支持三个平移轴和偏航；差速底盘支持前进和偏航。不支持的非零分量会报错。适配结果包含限幅后的目标与饱和轴。差速底盘在轮速超限时等比例缩放双轮，保持目标曲率。

既有 `FlightCommand` 的横移、偏航正方向是右，适配器负责转换符号。Tello 的归一化 RC 并非精确速度接口：速度比例需实测标定，并与实际发送 RC 的缩放比例一致。实体底盘仍需轮速闭环和设备监督。共同的数学接口本身不证明跨设备迁移。

## 可训练解码器

`NeuralSchema` 固定特征名称与顺序、来源标识、模拟发放率窗口。特征可以是群组均值或单神经元发放率。来源标识应包含真实图版本和特征定义；名字一致不能代替来源核验。

`NeuralObservation` 校验缺失特征、非有限值和负发放率。`RidgeMotionDecoder` 拒绝不匹配的模式；模型文件还记录输出轴、坐标系和单位。

```python
model = RidgeMotionDecoder.fit(schema, rates_hz, targets)
target = model.decode(observation)
model.save("candidate.json")
```

`rates_hz` 为 N×D，`targets` 为 N×6。标准化只拟合训练数据。岭回归是可替换的线性基线，不改变连接组突触，也不保证解决现有转向响应问题；其他学习方法可实现 `MotionDecoder` 协议。

先按场景或独立运行划分数据，再拟合。期望动作来自独立任务规则或教师；旧控制器 RC 不是正确性标签，实际运动速度也不自动等于期望动作。详见 [数据集流程](DATASET.md)。

模型和数据保存拒绝覆盖。为保持既有文件兼容，序列化格式中的 `flydrones.*` 标识保留；Python 导入和命令行入口使用 `cnsbridge`。

## 仿真时间与运行时间

连接矩阵为 CSC 格式 `W[post, pre]`。仿真累加实际放电神经元的突触输入，以环形缓冲区处理延迟，并更新膜电位和不应期。高密度放电时可切换为稀疏矩阵乘法。

模型推进时间与实际计算耗时不同。离线视频按每帧 50 ms 推进模型；全量网络的实际处理速度需要单独测量。不能从控制线程的发包频率推断模型更新频率。

## 扩展与验证

增加设备时实现运动目标转换及实际设备控制；增加学习方法时保留特征模式和动作单位契约。实体能力验证需在接口检查之后单独完成。

检查范围与改动匹配：文档修改检查差异，局部逻辑运行对应测试，模块交互运行相关子系统检查。不要仅为覆盖实现而重复扩大验证，也不要让测试启动实机运动。
