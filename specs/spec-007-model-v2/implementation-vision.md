# Implementation Report — P1 纯视觉子批

## Summary

新增不依赖脑、学习或设备的 V2 视觉编码器。它实现本期固定 20 Hz 播放时间下的 14 维群组特征，以及每眼 48 格光流与质量诊断。此子批不是整个 P1，也没有运行神经模型或产生动作证据。

## Files Changed

- `src/cnsbridge/senses/vision_v2.py`：独立新接口与视觉算法；未修改旧 Retina。
- `tests/test_vision_v2.py`：视觉算法及失效合同的定向验证。
- 本报告。

## Important Decisions / Interface

`VisionV2Encoder(config=None).encode(image_bgr, video_time_s)` 返回 `VisionV2Observation`；`reset()` 开始新连续片段。图像必须是 H×W×3 的 BGR uint8，尺寸至少 96×72。缩小采用纯 NumPy 的可分离精确面积平均，包含非整数缩放边界；不猜测颜色、不接受低分辨率自动放大。

时间必须有限、非负、严格递增；相邻差固定 0.05 s，误差容许量默认 `1e-9 s`。此版本不接受任意帧率，帧缺口返回 `time_step_mismatch`，不以更长 dt 吞掉缺帧。

观察的关键字段：

- `input_cutoff_s`、`valid`、`initialization`、`segment_stopped`、`invalid_reason`、`invalid_eye`、`segment_stop_reason`。
- `features`：顺序为 `L.brightness/ftb/btf/up/down/loom/loom_speed`，然后相同顺序的 R，共 14 个；无效观察为 `None`，不能拿到伪造零特征。
- `eyes["L"/"R"]`：6×8 亮度、真实像素中心、6×8×2 原始 px/s 光流、梯度结构张量 trace 纹理量、最小特征值、条件数、有效掩码/数量/比例；各轴扩张、最小扩张、EMA、loom 与原始正向 loom 变化率。
- 对外数组只读；未估计或无效格的光流为 NaN，初始化特征的零运动不是测量为零。

`valid=True` 仅表示该步视觉特征可交给后续处理。第一帧 `initialization=True`，亮度来自实际图像，运动与 EMA 初始化为零，只能用于预热；本模块不声称神经历史或 2 s 预热已经完成，后续回放层统一判断可预测状态。

局部 LK 使用前后帧平均图像的空间梯度、当前减前帧的时间差与未加 ridge 的 2×2 结构张量。左右眼梯度独立，避免跨分界计算。质量候选默认：最小特征值 `1e-6`（归一化亮度²/像素²）、最大条件数 `1000`；两者及时间误差均在 `VisionV2Config.identity()` 中记录并标记 `engineering_candidate_not_optimized`。这些不是文献生物值或本项目实测最优值。纹理仅作诊断，没有旧版纹理乘子或原始流裁剪。

方向特征逐有效格整流、除以 24 px/s、截到 [0,1] 后均值，L 的 ftb 向左、R 向右，up 向上。扩张分别以 `[1,x]`、`[1,y]` 的真实像素中心作无权最小二乘拟合，要求设计矩阵满秩，率单位 s⁻¹。`e=min(ex,ey)`，EMA 系数 `1-exp(-dt/0.1)`；loom 与 loom_speed 按设计公式处理。

两眼光流与扩张全部验证通过后才提交前图、时间、EMA 与前 loom。`no_valid_flow`、`unidentifiable_expansion`、`nonfinite_expansion` 等失败不推进上述状态，也不返回特征；仅设置片段停止标记，后续调用返回 `segment_stopped` 并保留原始停止原因，直至显式 reset。坏图像、时间错误也采用相同停止策略。此模块不生成神经输入率，后续集成必须拒绝无效观察。

## Validation

- Risk Level: **S2**，独立新增状态化视觉接口和质量/时间合同；不改生产入口、旧视觉或脑模块。
- Executed: `.venv/bin/python -m pytest tests/test_vision_v2.py -q`
- Result: **23 passed in 0.05 s**。
- 范围：BGR 与初始化、缩放、四方向单位和镜像、静态纹理零运动、真实图像膨胀与 EMA、实际像素坐标下扩张单位、秩亏/非有限拟合、平面及一维纹理、第二眼失败的原子性、锁停/reset、部分有效格、坏图像、时间错误、工程参数身份。
- 人工图像与构造光流只验证视觉算法合同，不替代真实神经网络、训练数据、实飞或行为证据。未运行全项目测试、硬件、全脑或独立验收。

## Known Limitations

单尺度局部 LK 仍依赖小位移与近似亮度恒定；本期质量门限检查纹理矩阵及数值可辨识性，没有宣称解决遮挡、强烈照明变化、大位移或所有光流错误。工程质量门限需要后续训练/验证 pilot 检查。固定 20 Hz 是播放协议，不等于相机真实曝光间隔。

当前模块尚未被 V2 神经回放、群组输入率或视觉对照历史调用；集成在后续 P1 子批中完成。输出对应工程图像分区，不是解剖视网膜映射。

## Remaining Questions

本子批无需要用户决策的问题。后续集成须把本接口配置和源码纳入来源身份，保持失效不推进神经/视觉历史、第一帧及 2 s 统一预热的约束。
