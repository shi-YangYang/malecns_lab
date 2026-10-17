# 可训练完整连接图核心

本模块是 Spec 008 的 P0 数值核心，身份为 `cnsbridge.trainable_lif.v1`。它提供完整参考图加载、内部类型共享增益和脉冲代理梯度。P0 的小电路检查不代表完整图训练通过，也不代表已经学会跟随或可实时控制设备。旧冻结 LIF/V2 未被修改。

## 环境与入口

训练使用独立 `.venv-train`，不修改既有 Tello `.venv`：

```sh
python3 -m venv .venv-train
.venv-train/bin/python -m pip install -e '.[train,dev]'
.venv-train/bin/python -m pytest tests/test_trainable_core.py -q
```

`train` 固定 PyTorch 2.14.1；核心只允许 CPU float32/float64。具体安装版本和定向检查记录见 `specs/spec-008-trainable-snn/implementation-p0.md`。尚未验证 MPS 稀疏计算，没有设备回退逻辑。

```python
import torch
from cnsbridge.brain.trainable import TrainableLIF

# 必须事先明确感觉编码对应的原始 body ID；此处空集合仅演示零输入接口。
model = TrainableLIF.from_reference(
    "data/malecns_v2/full_reference.npz", input_body_ids=[], dtype=torch.float32,
)
state = model.reset()
# 以下仅展示 API；全图运行须先按 P1 登记时间窗和资源预算。
increment_mv = torch.zeros(model.graph.n)
spikes, state = model.step(increment_mv, state)
state = state.detach()  # 截断反传，保持电压、突触状态、队列和时钟。
```

`from_reference` 必经既有 `load_reference` 的图、注释和来源校验。不会裁边、裁节点、忽略自环或用小图替代。`for_numerical_test` 是明确标记为 `numerical_test_only` 的独立测试入口，不是设备或任务模型。模型身份记录图来源、原始整数计数摘要、完整节点/边/突触计数、类型分组、外部驱动 body ID、计算精度、参数、语义和实现文件摘要。

原始有符号 int64 计数保留在只读 `model.graph.counts`；计算权重另存 CPU CSR 浮点矩阵，不会覆盖原始资产。float32 可能舍入较大的整数或计算结果；身份记录精度。`state_dict()` 只包含训练参数 `theta`，不重复保存原始大图；后续 checkpoint 必须同时保存并校验 `model.identity`，不能把裸参数文件当作独立可复现模型。

## 数值契约与来源

参考 [Shiu 等的官方 model.py](https://github.com/philshiu/Drosophila_brain_model/blob/main/model.py)，2026-10-17 核对的内容 SHA256 为 `fc45837d7122c6ce2a7f3f2f23c515992e4b232aadb919efabb72337fac88e4e`。使用其方程、默认参数、严格阈值、输入细胞无不应期、复位清除 g 这些设计依据。这是方程启发的工程模型，不宣称逐时间步复现 Brian2，亦不宣称 MaleCNS 真实生理参数已知。

单位 mV/ms，默认 `v_rest=v_reset=-52`、`v_threshold=-45`、`tau_m=20`、`tau_syn=5`、`refractory=2.2`、`delay=1.8`、`weight=0.275`、`dt=0.1`。时间步必须整除不应期与延迟；不自动取整。

```text
dv/dt = (v_rest - v + g) / tau_m
dg/dt = -g / tau_syn
```

每步 k 的固定顺序：

1. 队首到期增量加入 g。
2. 非不应期细胞精确积分耦合线性方程一个 dt；不应期细胞 v/g 冻结，但到达的增量保存在 g。
3. 指定外部输入细胞的 v 加上显式输入增量。核心不生成 Poisson 噪声、默认驱动或视觉特征；非输入细胞传入非零增量会报错。
4. 满足严格 `v > threshold` 且允许发放时产生二值脉冲；发放后 v=reset、g=0。
5. 非输入细胞在 `k + refractory_steps` 再次允许积分和发放；输入细胞没有不应期。发送的连接增量在 `k + delay_steps` 开始到达。

设 `a=exp(-dt/tau_m)`、`b=exp(-dt/tau_syn)`，活动细胞的解析积分为：

```text
v_next = v_rest + a*(v-v_rest) + g*tau_syn/(tau_m-tau_syn)*(a-b)
g_next = b*g
```

时间常数相等时，g 对 v 的系数采用极限 `dt/tau_m*a`。测试以独立的 2×2 矩阵指数和手写事件时间表核对，未用核心自己的输出作参考答案。

与官方实现的边界：官方依赖 Brian2 调度、PoissonInput 和不应期时间判定，本模块将其明确为上述整数步协议；外部输入由调用者提供，不采样 Poisson；官方 reset 字符串包含 `w=0`，但其神经元方程未声明 w，本模块不引入这个未定义神经元状态；本模块增加工程代理梯度和共享增益。以上不是生物学习规则。

## 学习参数与反向规则

按突触前细胞原始 `type` 注释共享内部增益：

```text
gain = 1 + 0.5*tanh(theta)
delayed_increment = W_base @ (gain[group(pre)] * spikes)
```

theta 初始为 0。缺失、空串与全空白注释进入组 0，其余原始字符串不去空格、不改名，按 Unicode 排序。IEEE 浮点饱和可能把端点舍入为 0.5/1.5，因此以 `nextafter` 保护严格内部端点，身份明确记录该舍入保护。正增益保持源符号；未知递质原始符号假设仍通过参考图来源公开。

前向脉冲严格二值，反向使用 `1 / (scale * (1 + abs((v-threshold)/scale))**2)`，默认 scale=1 mV。复位门、不应期门停止梯度，递归传播脉冲保留代理梯度。稀疏线性反向对所有细胞计算 `W_base.T @ gradient`，不只选取实际发放索引，因此静默细胞的潜在代理梯度不会被丢弃。固定矩阵没有梯度，不构造 N×N 密集图、不按步重建边权。

`run(inputs, state)` 的输入是每步一个 N 维 mV 向量，可传迭代器。返回脉冲序列和完整状态；连续两段传递状态与一次运行一致。`reset()` 重置动态状态而不重置训练参数；`state.detach()` 截断全部状态（含延迟队列）的梯度。跨模型状态、非法精度/形状、非有限参数/输入/状态会明确报错。
