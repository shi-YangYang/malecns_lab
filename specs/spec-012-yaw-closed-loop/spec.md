# Spec 012 — 完整 MaleCNS 单轴视觉稳向闭环

> 当前定位：本阶段作为历史设计保留；2026-10-21 清理后网页和运行入口已移除，原始结果仍保留。不是当前 X2 物理仿真或接入前置。

状态：Done，2026-10-21 独立验收 PASS。用户在确认下一步方案后要求“做”，现已交付慢速锁步闭环、成对关闭控制基线和网页回放。两次正式完整图任务已留档，见[实测报告](actual-run-20261021.md)；固定 LIF 与工程读出没有一致稳向优势，未开始网络训练或实体设备控制。

## 目的与边界

实际因果链：世界固定纹理 → 当前偏航对应的相机图像 → 完整 MaleCNS 连续神经状态 → 转向读出 → 单轴机身运动 → 下一帧图像。网页三维果蝇仅表示该简化机身姿态，不声称果蝇肌肉、飞行动力学或智元仿真已实现。

每个任务包含 neural / zero 两段：同一纹理 seed 7、神经 seed 0、初始状态、外部偏航扰动与时长；各段独立重置一次脑与机身，段内保持状态。zero 也运行完整脑并记录读出，只把实际发送给仿真机身的动作设为零。不得以图像、扰动标签、偏航或速度真值直接生成 neural 的动作。

固定图、LIF 和感觉核心不改，完整图必须经现有 load_reference 校验。初版读出是未训练工程基线，无控制性能保证。后续网络内部参数与读出联合训练另立实验协议；不把本轮称为已经学会稳向。

## 数值契约

- 控制周期 0.05 s（20 Hz），每帧 100 个 0.5 ms 神经步；各段预热 40 帧（2 s），预热动作及外部扰动为零。实际墙钟耗时独立记录。
- 偏航 FLU 正值左转，单位 rad，动作使用 BodyVelocity(wz=...)，其余轴为零。执行器为一阶角速度响应，tau=0.2 s；外部等效角速度驱动也经独立 tau=0.2 s 的一阶响应，机身实际速度是两响应之和。在每区间用解析平均角速度积分偏航。扰动不输入读出；其衰减状态使关闭控制基线在扰动结束后仍有短暂残余旋转，而非瞬时归零。
- 扰动从观察期起连续 0.5 s，幅值 direction×disturbance_rad_s；其余时刻为零。观察时长 1.5–4 s，0.05 s 整帧，默认 2 s；幅值 0.15–0.4 rad/s，默认 0.25，方向 left/right。预热后的第一张图仍由扰动施加前姿态生成，禁止窥视未来图像。
- 纹理沿用标准源的确定性二维有纹理图样；水平采样 x_world=x_pixel−60×yaw_rad，固定世界纹理，近轴圆柱角度投影，96×72 BGR。正偏航使固定特征向图像右侧滑动。无自运动以外的独立画面平移；参数及源码 hash 存档。
- 工程读出：DNg02_L/R 群组平均 Hz，以预热最后20帧平均差为基线；signal=(L−R)−baseline；EMA alpha=0.2，wz=clip(0.005×EMA(signal),−0.35,+0.35) rad/s。仅使用完整脑输出的群组率，缺组或非有限立即失败；符号和增益为显式工程假设，未校准为生物/机器人映射。记录左右率、原始差、基线和最终指令。zero 仍计算 candidate_wz，但执行 command_wz=0。
- 每帧顺序：渲染 yaw_before → SNN 推进 → 读出 → 零控制开关 → 机身积分 → 保存 yaw_after 和区间实际角速度。下一帧 yaw_before 必须等于本帧 yaw_after。
- 指标只覆盖观察期：实际角速度 RMS、峰值、绝对转角积分；恢复时间为从扰动结束到首个满足条件的区间起点的延迟，此后直到观察结束角速度均≤0.03 rad/s，且至少有0.5 s记录支持，否则 null。改善率=1−neural_RMS/zero_RMS，负值照实显示；软件 completed 与稳向有效分开。目标为抑制旋转，不要求恢复初始朝向。

## API 与文件契约

保留 Spec011 所有旧请求/记录。仍用 `/api/experiments`，新增严格请求：

```json
{"experiment":"yaw_stabilization","direction":"left","disturbance_rad_s":0.25,"duration_s":2.0}
```

旧四字段刺激请求保持原样。新 Job parameters 为上述对象；total_frames=2×(40+duration_s×20)，frames_completed 累积两段，simulation_s=frames_completed/20。单 worker、内存/预算/取消/权限与原任务共享；两段都完整才发布。失败不降级缩图，不操作其他应用。

新任务资产：catalog.json、atlas.json、run-neural.json、run-zero.json、stimulus-neural.png、stimulus-zero.png。catalog/runs 沿用 atlas.v1，id 为 neural/zero，stimulus 为 yaw_neural/yaw_zero，label 为“神经控制”/“关闭控制”。两段 run 都含现有神经/帧数组及 generated 实际输入预览。

新增 `run.closed_loop`：

```text
schema: malecns_lab.yaw_episode.v1
mode: neural | zero
dt_s: 0.05
disturbance_end_s: 2.5
readout: {kind: fixed_engineering, baseline_hz, gain_rad_s_per_hz:0.005, smoothing:0.2, limit_rad_s:0.35, groups:[DNg02_L,DNg02_R]}
environment: {kind: single_axis_yaw, response_tau_s:0.2, disturbance_response_tau_s:0.2, pixels_per_rad:60}
每帧等长有限数组：yaw_rad, yaw_after_rad, candidate_wz_rad_s, command_wz_rad_s,
  actuator_wz_rad_s, disturbance_wz_rad_s, actual_wz_rad_s,
  left_rate_hz, right_rate_hz, readout_signal_hz
可附加raw_rate_difference_hz记录原始L−R；disturbance_wz_rad_s是外部驱动输入，actual_wz_rad_s才是两响应合成的机身实际速度。
metrics: {rms_rotation_rad_s, peak_rotation_rad_s, integral_abs_rotation_rad, settling_time_s: number|null}
```

每段 records-<mode>.npz、inputs-<mode>.npz、neuron_totals-<mode>.npz、trajectory-<mode>.npz 原始留档。manifest 保存全图身份、完整参数、种子、模型/环境/读出/worker源码摘要、每段基线及时间契约；result 累计帧/神经步/时间和两段 coverage/metrics。DISPLAY/EVIDENCE 文件清单必须按任务类型严格选取，不把新白名单文件当成旧任务必需项。不得污染旧结果或其身份。

## 网页

实验类型提供“视觉稳向”与“刺激响应”，默认前者；稳向参数为扰动方向、幅度、观察时长。成对执行，显示累计真实进度与耗时，支持现有取消和历史恢复。

结果同屏呈现实际视野、脑活动、动作及机身角速度、偏航和两段指标对照；曲线/姿态按同一个记录帧同步，三维身体只使用记录姿态，不能依据刺激按钮播放预设动画。切换对照时保留相同帧便于比较；旧刺激保持静态果蝇并清除稳向指标。页面明确“工程读出 · 未训练”“简化机身 · 计算后回放”，用短标签呈现必要限制，无技术说明堆积。

## 验收与停止

S2：覆盖新增环境/读出/成对任务与旧路径直接回归，生命周期共享处只做定向检查；不跑全仓库训练/测试。

1. 定向检查动作符号、投影、因果先后、持续状态、神经信号断开与 zero 开关、指标区间/未恢复判定、参数和新旧文件契约。测试替身仅验证编排，明确不作为真图证据。
2. 前端一次构建与浏览器检查：两种模式、进度、成对比较、同帧切换、实际姿态和指令数值、旧历史回放、窄屏/错误边界。
3. 默认方向及反向各执行一次真实正式任务（共四段完整脑），核对全图/帧/步/状态连续、实际保存输入与轨迹匹配、zero 动作全零、模型指令与真实率公式一致；报告效果包括失败。相同初态与参数段间可从预热数组比较验证，无需第三次重复任务。
4. 独立验收复用证据，不重复通过项；服务和 worker 交付前停止。没有请求时不 commit/push。
