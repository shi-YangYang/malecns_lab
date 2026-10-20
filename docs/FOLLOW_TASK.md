# 三轴标记目标跟随：环境与教师

本工具是后续完整 MaleCNS 训练的离线任务环境。当前包含外部像素检测、常规视觉基线、特权教师和评分，不加载脑模型，不连接 Tello。教师的表现不能作为 SNN 学习成功或实飞证据。

协议 `cnsbridge.marked_follow.pilot.v1` 对应 [P2a Spec](../specs/spec-008-trainable-snn/p2-environment.md)。依赖现有 NumPy、SciPy，无需 PyTorch、OpenCV、录像或硬件。

## 运行

在仓库根目录使用已安装的项目环境：

```bash
.venv/bin/python -m malecns_lab.learning.follow_task --list-scenes

.venv/bin/python -m malecns_lab.learning.follow_task \
  --scene train-sine-combined --policy visual \
  --out data/follow_task_example/visual

.venv/bin/python -m malecns_lab.learning.follow_task \
  --scene train-sine-combined --policy zero \
  --out data/follow_task_example/zero
```

如果源码未做 editable 安装，在命令前设置 `PYTHONPATH=src`。输出目录必须不存在，工具不会覆盖旧实验。`--policy visual` 是工程视觉控制器；`zero` 是全零速度基线。

## 相机与坐标

- 20 cm 红色方形标记；默认期望距离 1.5 m。相机 160×120、BGR uint8、水平视场 72°、正方形像素。
- 世界 x/y/z 为初始前/左/上；无人机 FLU 正 `vx` 前进、正 `vz` 上升、正 `wz` 左转，单位 m/s、m/s、rad/s。
- 图像观察 `x_error` 正值在画面右侧，`y_error` 正值在画面上方；各自以半幅宽/高归一化。因此右侧目标要求负 `wz`，上方目标要求正 `vz`。
- 标记平面固定朝向世界 x，中心按独立轨迹移动；渲染实际四角投影，不把标记朝向相机。接近相机平面的角点不渲染。
- 动作只允许 `vx/vz/wz`，分别限幅 ±0.3、±0.2、±0.5；其他轴非零报错。默认周期 0.1 s，速度响应 tau=0.25 s，一阶滞后后积分位姿。零速度指令不代表瞬间停住。

这是运动目标层面的简化仿真，没有经实测标定的 Tello 电机、气动或刚体动力学。地面 z≤0.15 m、空间 |x/y|≥5 m 或 z≥3 m、距离标记中心≤0.2 m 分别构成终止条件。目标碰撞使用距离代理，没有房间障碍物几何，不能检验避障。

## 接口与真值隔离

| 接口 | 输入和用途 |
|---|---|
| `TargetCamera` | 不可变相机/已知标记尺寸标定 |
| `detect_target(frame, camera)` | 仅 BGR 像素；返回不可变 `TargetObservation` |
| `target_mask(frame, camera, shape=(12,16))` | 图像空间保序的红色占用率网格，行向下、列向右；不代表解剖视网膜映射 |
| `FollowEnvironment(scene, camera, dt_s=0.1)` | `frame()` 取得相机图像；`step(BodyVelocity)` 改变下一观察；`truth()` 单独访问特权状态 |
| `policy(FollowPolicyInput)` | 只收到 frame、observation、camera、time_s，返回 `BodyVelocity` |
| `truth_teacher(FollowTruth)` | 仅教师标签，不能作为以视觉为输入的部署策略 |
| `rollout(scene, policy, policy_name=...)` | 封闭环境生命周期，记录策略、执行、教师与评分；可传 `record_frames` 回调收集对应像素样本 |

`FollowPolicyInput` 不包含 scene、环境或真值；API 隔离用于避免意外泄漏，不是防止任意恶意 Python 闭包的沙箱。部署策略不得闭包捕获环境或调用真值教师。

检测器选择最大的红色连通区域，对极小、边缘裁切、多个同级候选、明显缺损或形状不符的标记返回 `valid=False` 和原因。无效观察的误差/距离为 `None`，不会伪造有效的零误差。表观高度通过已知尺寸估计距离，存在像素量化和斜视误差；coverage 是包围盒内红色占用率，confidence 是工程分数，不是校准概率。不能区分所有保持方形轮廓的遮挡，也不能识别任意物体。

基线与模型共用外部失去目标保护：检测无效时执行零速度，并记录 `lost_target:<reason>`。此时仍记录原始策略动作，不把保护动作计作模型学会搜索。保护不会向策略提供真值。后续模型可用 `target_mask` 构造保留空间信息的感觉输入，不能把观察误差从读出旁路到主动作输出。

## 场景划分与评分

`scene_registry()` 登记了 14 个固定场景，每个场景默认 8 s；完整场景归属一个 split，不可随机切相邻帧。

| Split | 轨迹族 | 背景/光照 | 种子 | 内容 |
|---|---|---|---|---|
| train | sine | flat / 1.0 | 101–104 | 前后、左右、上下、组合 |
| val | triangle | stripes / 0.8 | 201–204 | 前后、左右、上下、组合 |
| test | chirp、excursion | checker / 0.65 | 301–306 | 三轴与组合、遮挡、出画再返回 |

目标初始偏移、振幅、周期、遮挡区间及随机种子在场景中显式保存。轨迹完全不依赖策略动作。测试场景只能用于冻结策略后的报告，不能用于拟合教师或选参；开发测试只在 train 场景比较视觉基线和零动作，另用独立 pilot 静态夹具检查软件。

评分在每个控制周期动作之前进行：有效观察、水平/垂直归一化真值误差绝对值各≤0.15、真值距离误差≤0.25 m 时才合格。全轨迹合格比例≥80%、完整执行且无碰撞/越界才成功。提前失败时比例分母仍是完整计划步数，不能通过提前终止提高成功率。遮挡/出画帧不算合格；丢失与首次重新有效观察的时间单独报告。

报告包含横/纵/距离 MAE、可投影评分样本数、观察覆盖、合格比例、实际动作每轴非零样本数、动作平均变化、介入与失败。中心在相机后方时投影误差为 `None`，对应覆盖单独显示；距离误差仍参与报告。MAE 使用实际已执行轨迹样本，提前失败需同时读取覆盖和失败，不能只比较 MAE。

## 输出数据

`summary.json` 包含 `metadata` 和 `metrics`。metadata 保存 schema、协议、完整场景/相机参数、策略身份、动作坐标单位、周期/响应、冻结容差和证据边界。

`trajectory.jsonl` 每步一行，字段为：

- `time_s`、`observation`：策略当时看到的像素检测结果；
- `raw_policy_action`、`executed_action`、`intervention`：原始模型动作与经过保护/限幅的执行值；
- `teacher_action`、`teacher_source=simulator_truth`：独立特权教师标签，没有经过动作限幅；
- `truth`、`next_truth`：动作前后真值，仅供训练标签/评分；
- `errors`、`qualified`：按冻结规则计算的结果。

CLI 默认不存图像。训练集成可使用 `record_frames` 回调以 time_s 对齐图像/mask 与日志，但训练输入只能来自回调的视觉字段；真值仅供教师/评分。

## 首次环境自检记录

2026-10-17，本机各执行一次 `train-sine-combined`、8 s / 80 步的视觉基线与零动作；证据在本地 `data/follow_task_smoke_20261017/{visual,zero}/`。

| 指标 | 视觉基线 | 零动作 |
|---|---:|---:|
| 横向 MAE | 0.06855 | 0.14232 |
| 垂直 MAE | 0.09631 | 0.15100 |
| 距离 MAE / m | 0.17853 | 0.51529 |
| 有效观察比例 | 1.00 | 1.00 |
| 合格比例 | 0.75 | 0.00 |
| 每轴非零步数 vx/vz/wz | 80/80/77 | 0/0/0 |

两者均无碰撞/越界。视觉基线的合格比例仍低于预定 80%，所以该次 **任务成功为 false**；没有为改变结果而调整阈值。这是闭环环境和评分的自检，不是模型训练结果。
