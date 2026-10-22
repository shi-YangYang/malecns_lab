# 实施 — 冻结推理 wrapper

2026-10-23。范围 S2。仅新增 `learning/x2_frozen.py` 与定向检查；未修改 Spec019 的加载器、监督器、CLI、原来源源码、训练路径或参数。

## 接口与身份

```python
engine = FrozenInference(verified_historical_policy, checkpoint_id=authenticated_checkpoint_id)
execution_identity = engine.execution_identity
state = engine.initial_state()
output = engine.forward_tensor(frame, state)
```

- 仅接受已加载的 `cnsbridge.learning.gpu_policy.GPUPolicy`、原 MPSLIF/GPUDrive 类型和 `full_reference` 图，无 fixture/CPU/缺图回退。调用者仍负责来源与 checkpoint 鉴权、完整图构造和资源监督。
- `initial_state()` 返回原来源 `PolicyState`；`forward_tensor()` 返回原 `GPUOutput(logits, action, features_mv, state)`，保留全部三轴、完整状态、原时钟。可选 `frame_offset` 只允许 `0` 或 `policy.frame_steps`；没有 `direct_numerical_fixture` 开关。
- 独立 `execution_identity` 包括 wrapper 文件 SHA、原四个执行模块文件 SHA、原 policy/model/drive ID、图身份、配置、参数摘要、Torch 版本和可选 checkpoint ID。其 `execution_id` 独立于旧 `policy_id`。返回状态保留旧 `policy_id` 仅用于原序列化类型兼容；归档必须同时保存新的执行身份。
- checkpoint ID 是经调用者鉴权的引用，wrapper 不宣称自行读取/校验 checkpoint。构造不加载图、不建立 optimizer。

## 执行与错误监测

- 所有公开执行方法均 `no_grad`，不更改参数 `requires_grad`。每次调用前后核对参数摘要、参数/缓冲区绑定与版本、对象/配置/ID；复用原参数与状态边界验证。
- 每帧预计算原括号顺序下的感觉增量、衰减系数、背景项与分组增益。保留原 `fixed_mps_matvec`、`surrogate_spike`、感觉事件顺序、神经表达式、输出累加顺序、归一化、readout 和 tanh。
- 保留所有节点、边、延迟队列与构造出的反向矩阵。构造时间和反向矩阵不是本次优化范围。
- 每神经步在设备累计 int32 sticky 标志：非有限值=1、phase 范围=2、event counts 范围=4、refractory 范围=8。检查包含 reset 前的电压/电导/阈值距离与新传递量，有限 reset 不能抹掉异常。每 50 步及末尾不足 50 步检查一次；异常抛出且不返回动作。完整输入状态仅在调用边界检查；不可变延迟队列内每个新元素在产生时检查。
- 输出 features/normalized/logits/action 另行检查有限值，整数时钟在执行前检查溢出界限。失败不会修改传入的状态。

## 实际检查

```text
.venv-train/bin/python -m pytest -q tests/test_x2_frozen.py
27 passed in 0.67s
```

仅执行以上一个定向 pytest 命令。CPU 与实际 MPS 小张量均执行，无跳过。检查范围包括 NaN/正负 Inf 瞬态、范围错误累计、无提前主机读回、reset 后有限值仍保留溢出错误、50 步检查与最终余块停止、拒绝外来 policy/fixture 参数及身份副本隔离。

其中单标量故障注入仅验证监测器接入，不构造合成脑，不提供连接组数值一致性证据。没有启动完整脑、仿真 VM、ROS、训练或硬件。

## 尚待验收

完整图预热与三帧所有状态逐元素精确一致、冻结参数实测一致、受监督资源/退出与至少 20% 提速仍待协调者按 Spec020 执行。当前代码检查通过不代表 AC3/AC4 或优化验收通过。
