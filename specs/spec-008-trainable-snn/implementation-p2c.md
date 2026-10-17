# Implementation Report — P2c 早期视觉背景变体

## Summary

已实现 Ready P2c 的可选 LaminaTonicLIF 数值变体和 gain-only checkpoint 防遗漏保护。原 TrainableLIF 默认仍为无背景、只有 theta；P1 协议未修改。没有加载完整 MaleCNS 图、运行资源探针、执行任务训练或接入设备。本报告由实施者编写，不代替独立验收。

## Files Changed

- `src/cnsbridge/brain/trainable.py`：默认返回 None 的平衡点偏移 hook；可选项只加入 active 积分前的电位解析式，默认分支没有新增数值运算。
- `src/cnsbridge/brain/tonic.py`：独立模型 schema、L1–L5 精确选择、learnable/fixed rho、严格有界偏移、成员/模式/来源身份及显式测试工厂。
- `src/cnsbridge/learning/checkpoint.py`：在目录创建或参数恢复前核对原动力学 schema 与精确 theta 参数集合；包含参数别名的额外注册也拒绝。读取时拒绝非 theta 参数声明和不支持的模型 schema。
- `tests/test_tonic_dynamics.py`：独立解析参考、发放/复位/不应期、固定对照、rho 梯度更新、形状/精度/设备/有限值、分组/身份/原图保持检查。
- `tests/test_training_probe.py`：补充 tonic 两模式、额外注册参数/别名和 metadata 缺失/额外参数/新 schema 的拒绝且无副作用检查。
- `docs/TRAINABLE_SNN.md`：说明工程假设、比较条件、时间窗和持久化限制。
- 本实施报告。

协调者同时维护的 Decision 010、P2c Spec、plan 与 signal-observability 文档未由本实施者修改。

## Important Decisions

- 依照 Decision 010，生产入口固定全部原始 L1–L5 类型，不能由调用方传子集。测试工厂允许按 L1–L5 顺序选择非空子集，身份保留 `numerical_test_only`。没有建立裁剪图或替代脑入口。
- rho 初始0；`16*sigmoid(rho)` mV 以 nextafter 防浮点饱和到边界。非成员严格0，L1–L5 保留正常不应期，不改变 input_mask 或 g 的背景项。
- fixed 模式 rho 为 buffer，learnable 模式为 Parameter；初始前向相同而状态身份不同。严格零背景对照继续使用原 TrainableLIF。
- 共享核心只新增 hook；变体复用现有 step。图对象、全部原始计数与其身份不变。新身份记录模式、类型顺序、各组 body ID 摘要与数量、总成员数、公式、范围、初值及 tonic.py 源码摘要。
- 本批选择明确拒绝变体 checkpoint，包括 fixed 模式，不在旧格式中静默遗漏 rho。变体完整持久化仍待后续策略批次。

## Validation

- Risk Level：S3，因共享数值积分入口与 checkpoint 持久化保护；只验证直接相关子系统。
- 已阅读改动 diff 与新增文件，`git diff --check` 通过。
- 唯一定向测试命令：

```sh
PYTHONPATH=src .venv-train/bin/python -m pytest \
  tests/test_trainable_core.py tests/test_tonic_dynamics.py tests/test_training_probe.py \
  -k 'not resource and not preflight and not supervisor and not timeout and not rss_stop' -q
```

- Result：**53 passed，8 deselected，1.42 s**。唯一 warning 为既有 PyTorch sparse CSR beta 提示。
- 原 P0 独立矩阵指数/时序检查覆盖默认 hook 回归。tonic 独立指数解计算首次严格阈值跨越为第416步，即41.6ms；核对跨越之前每步电位、复位、blocked 时电位冻结及到达 g 保留、边界恢复积分。
- rho 的单步电位梯度与独立导数 `4*(1-exp(-0.1/20))` 一致；优化改变 rho 和后续电位，reset 保留学习参数；固定 buffer 不被参数优化器更新。
- 原 checkpoint 精确往返检查通过；拒绝测试确认不创建输出父目录、不部分恢复参数，生产 load 在不支持 metadata 时不会尝试读取参考图。
- 未运行完整测试集、完整图、资源监督进程、任务训练、硬件或新性能测试；未重复已通过检查。

## Known Limitations

8mV 是候选工程工作点，数值检查只确认按指定方程运行。未确认完整图视觉信号到达下降神经元，未证明学习或控制收益。背景自发活动不等于视觉传播；未来需要登记预热、空白/位置/尺度对照及相同背景下的冻结/可训练增益比较。所有细胞统一 LIF 仍是建模近似。

变体不能使用现有 gain-only checkpoint 保存或加载。共享核心源码改变意味着代码身份改变，旧 checkpoint 不会自动跨源码身份兼容。用户尚未通知内存已释放，本批没有重试完整图计算。

## Remaining Questions

无本批实施阻塞。等待独立验收；后续完整图传播、背景选型、持久化和策略学习按各自登记推进。未提交 Git commit，由协调者在独立验收后分步提交。
