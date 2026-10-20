# Tello 实验

实现位于 `malecns_lab.experiments.tello`，下列命令通过 `python -m` 直接运行包内模块。试飞策略、脑计算子进程、会话清理分别在 `policy.py`、`worker.py`、`session.py`，不需要从其他脚本导入业务逻辑。

当前实体平台是 Tello。电脑通过 Tello Wi-Fi 读取相机与遥测、发送 SDK 指令；飞机自身飞控负责姿态稳定。

这里保留的是既有人工读出链路的地面检查与短时飞行工具。V2 学习产物（如 `model_neural.json`）尚未接入这些工具；可训练 SNN 的部署按 Spec 008 单独验证。

## 地面连接与视频

关闭手机 App 和其他 Tello 程序，将飞机放在地面，电脑连接 Tello Wi-Fi。在项目目录激活 Python 环境后运行：

```bash
python -m malecns_lab.experiments.tello.check --seconds 20
```

该步骤检查 SDK、遥测与视频，不起飞、不发送 RC。只有 `SDK OK` 和可见的实时画面，才说明连接与解码正常；不能据此判断控制算法有效。

如果出现 `response decode error`、收不到响应或画面中断，先处理连接和其他程序占用问题，再进行后续实验。

## 地面观察旧读出

```bash
python -m malecns_lab.experiments.tello.shadow --seconds 30
```

默认使用 `configs/malecns.yaml` 中的真实 MaleCNS 模型和校准读出；相对模型及校准路径按源码工程根解析，缺少所需资产时直接停止。等待 `SETTLING` 完成后再改变画面。终端的 `NOT SENT` 表示只记录算法输出，没有发给飞机。

已有 `readout_malecns.json` 时无需重新校准。缺少它时，在模型准备完成后运行 `malecns_lab calibrate --brain data/malecns_brain.npz --out readout_malecns.json`，为旧读出生成校准资产；该校准不训练新的通用解码器，也不证明实机效果。

该工具使用 `MotorDecoder`，不加载新岭回归候选模型。左右移动小物体未引起明显偏航，可能涉及视觉特征、神经活动或读出映射，不能仅通过加大 RC 推断已解决。

## 地面录像与会话格式

已有录像时可以直接离线分析；新录像在 Tello 固定于地面、手机 App 和其他 Tello 程序关闭、电脑已连接 Tello Wi-Fi 后录制：

```bash
python -m malecns_lab.experiments.tello.record --scene-id room_a --seconds 30
```

Q 或 Escape 结束预览。也可使用 `--no-window` 保存而不显示画面；`--out-dir` 必须指定不存在的新目录。录制仅连接 SDK、读取视频和遥测，不加载脑模型、不发送运动指令。`--seconds` 允许 5–60 秒；终端打印 `data/ground_recordings/` 下的实际会话位置。

| 文件 | 保留内容 |
|---|---|
| `video.avi` | 名义 20 fps 的 MJPG 视频 |
| `frames.csv` | `frame_index`、主机取帧时间 `t_s`、播放时间 `video_t_s` |
| `session.json` | `flydrones.ground_recording` version 1、`scene_id/session_id`、文件名、名义帧率和地面采集标识 |
| `labels.csv` | 仅有 `start_s,end_s,vx_mps,wz_rad_s,reason` 表头，不包含自动生成的正确动作 |

`video_t_s = frame_index / 20` 是 AVI 播放秒数；`t_s` 是主机从解码队列取帧的相对时间，两者都不是相机曝光时间。采集有间隙时，30 秒墙钟录制可能不到 30 秒 AVI。标注必须以实际 AVI 播放时间为准，不能用墙钟或直接把 `t_s` 当标签时间。中断的部分文件保留，不自动覆盖。

同一物理布置的重复录像复用 `scene_id`；离线重放还需为同素材及其裁剪、增强指定共同的 `family_id`，不能改名字制造独立训练/测试样本。

### 30 秒信号观察拍摄参考

固定 Tello、背景和照明，只移动边缘清楚的卡片或手机背面。左右按预览画面定义；避免物体一直挡满镜头。下面时间只用于操作安排，实际标记需看片确认。

| 参考时间 | 操作 |
|---|---|
| 0–6 秒 | 物体在画面外，保持静止；前 2 秒供模型预热 |
| 6–9 秒 | 保持大致距离，从画面左侧向右侧匀速移动一次 |
| 9–12 秒 | 移出画面并准备下一次，记录为过渡 |
| 12–15 秒 | 相近距离和速度，从右侧向左侧移动一次 |
| 15–18 秒 | 移出、暂停，再准备正面靠近 |
| 18–21 秒 | 物体对准画面中央平稳靠近，使图像逐渐扩大 |
| 21–24 秒 | 移出后复位，记录为过渡 |
| 24–27 秒 | 从相近位置重复正面靠近 |
| 27–30 秒 | 移出物体，保持静止 |

这份安排只用于检查信号响应，不规定应左转、右转或上升。各阶段可记录在会话旁的实验笔记中；V2 不读取旧 `stimuli.csv` 区间表。需要独立片段时，按 [V2 分段协议](MODEL_V2_USAGE.md#4-显式分段与失败) 在分析前登记，并注意每段都重新预热 2 秒。

采集后可以断开 Tello，使用完整参考图反复重放：

```bash
python -m malecns_lab.diagnostics.replay_v2 \
  --recording data/ground_recordings/SESSION --family-id room_a_session_01 \
  --stimulus-kind object_motion --out data/neural_v2/recording_01
```

将 `SESSION` 换成实际会话目录，输出目录必须是新的。完整 [诊断与报告](MODEL_V2_USAGE.md) 不需要动作标签；若要运行冻结脑读出对照，按 [V2 标签与训练协议](LEARNING_V2.md) 独立标注。现有空白 `labels.csv` 只可经显式 `legacy_v1_intervals` 选项导入两轴标签；首选 V2 JSON 和支持掩码，不把未知轴写成已训练的零动作。不要将 shadow 的旧 RC、神经输出或实测速度自动当成期望动作。

地面信号录像不能验证动作改变相机观察的闭环。当前三轴 SNN 目标跟随训练按 [Spec 008](../specs/spec-008-trainable-snn/spec.md) 独立推进，不以这些静态录像替代仿真和实机闭环。

## 既有短时飞行工具

先查看试飞入口的参数和限制：

```bash
python -m malecns_lab.experiments.tello.trial --help
```

实飞需要显式 `--takeoff`。试飞入口包含多种模式：

| 模式 | 作用 | 实验边界 |
|---|---|---|
| `yaw` | MaleCNS 偏航 | 其他运动轴关闭 |
| `reflex` | MaleCNS 偏航与升降 | 不前进、不横移 |
| `forward-check` | 短时固定前进 RC | 检查设备响应，不验证神经控制或避障 |
| `cruise` | MaleCNS 受限巡航 | 有前进时间窗、静止段及监督限制 |
| `full` | 全量 MaleCNS 旧读出 | “全量”指连接组来源，仍受限幅、门控和结束条件限制 |

所有脑驱动模式加载 `configs/malecns.yaml` 的真实连接组与校准读出；模型或校准文件缺失时会报错，不切换到替代网络。`forward-check` 的固定指令仅用于设备探针。统一模型来源不扩大各模式的动作权限，也不代表这些模式已经用当前配置完成实机复测。

`full` 当前总时长 12 秒：前 2 秒零 RC、中间最多 8 秒允许控制、末尾 2 秒零 RC；模型准备状态、数据时效或停止条件可能缩短实际动作时间。它不意味着每个轴都会有非零输出，也不意味着自主导航。

现有试验的前进上限为 RC 15，偏航、升降和横移各为 RC 10；具体开放轴依模式变化。起飞前要求至少 40% 电量，空中低于 25% 结束试验；高度超出 30–150 cm、航向变化达到 25°、模型结果或输入超时等也会结束试验。`full` 中触发逃逸后会锁止本次前进。完整条件以脚本实现为准，不能把这些限制当作位置控制或避障保证。

飞行前应有桨叶保护、明确的空旷试验区域和现场观察者；不要在镜头前近距离挥动物体接近旋转桨叶。Ctrl+C 请求降落；收到降落确认后，仍需目视确认触地停桨。连接失效时软件无法保证收到或完成降落指令。

## 如何判断结果

`nonzero_forward`、`nonzero_yaw` 等统计的是发出的命令，不是实际位移。发送 RC 时长、模型计算次数和肉眼可见运动时长也可能不同。SDK 的速度字段和高度字段不能替代独立的位置测量。

既有现场观察已记录过短距离前进、升降和轻微转向。这说明部分感知、模型计算和指令链路能够运行；它不证明可靠避障、正确的刺激响应或新解码器有效。

保留 CSV、配置、模型来源、场景描述以及外部视频，分别说明：

- 哪些神经活动或模型输出发生变化；
- 哪些指令实际发送；
- 飞机实际做了什么；
- 哪些结论仍无法从此次记录判断。

日志位于 `logs/`。地面视频供重复信号诊断与冻结脑对照；可训练 SNN 的目标跟随、在线观察与闭环实飞需独立接入和验证。
