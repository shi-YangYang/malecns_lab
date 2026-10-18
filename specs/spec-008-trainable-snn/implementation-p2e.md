# Implementation Report — P2e 有状态神经策略接口

## Summary

新增完整图生产工厂与显式数值测试工厂，将现有目标占用编码、R1–R6 事件驱动、有状态 LIF、全部 DN 两半窗电位特征及三轴线性读出相连。提供保留 autograd 的训练张量接口，以及显式场景 reset、no_grad 转换 BodyVelocity 的独立环境适配器。本报告不承担最终独立验收。

## Files Changed

- `src/cnsbridge/learning/neural_policy.py`：时间配置、完整状态/结果契约、来源和角色绑定、可微策略与部署边界。
- `tests/test_neural_policy.py`：仅五节点数值 fixture 的接口检查。
- `docs/NEURAL_POLICY.md`：参数、状态、时间、动作、环境调用和未完成的训练/持久化边界。
- 本报告。

未修改已有脑核心、空间驱动、元数据、环境或诊断源码；未覆盖协调者同时运行的完整图实验及治理文件。本批建立新的训练接口，没有已被其完整替代的旧运行入口，因此未删除仍被使用的基线或资产。

## Important Decisions

- 生产必须通过 `from_reference` 校验完整图与完整元数据映射；数值入口只能接受明确 `numerical_test_only`。无任意默认时间窗，无隐式裁图或缺失回退。
- 每次 forward 从显式状态构造局部事件驱动，不留跨调用可变相位；失败不改变传入状态。完整脑状态、驱动状态及 frame index 三者时钟严格绑定。
- 全部 DN 的 post-reset 电位分别在前后半窗求均值，不筛选活跃神经元。固定物理归一化，无验证/测试统计拟合，无视觉/真值直连读出。
- 构造线性层时恢复全局 CPU RNG，再用局部 Generator 赋非零随机权重，避免构造器默认初始化污染全局 RNG。
- 固定背景严格保持非训练 `rho=0`；参数集合只有 theta 和读出 weight/bias。theta/readout 的学习值不改变结构 policy ID；旧 gain-only checkpoint 未扩展、未当成完整策略保存格式。
- controller 必须显式 reset 才能运行，按登记 camera/control clock 检查输入；只在部署边界转换 Python float，外部丢失目标保护仍归原 rollout。

## Validation

- Risk Level: **S3**，跨模块状态/来源/梯度接口；检查范围仅新接口，不重复旧核心/诊断套件。
- Checks Performed:
  - `PYTHONPATH=src .venv-train/bin/python -m pytest -q tests/test_neural_policy.py`
  - 结果：**28 passed, 1 warning in 3.90s**，退出码 0。唯一 warning 是既有 Torch sparse CSR beta 提示。
  - 手工两半窗均值和独立线性代数（float32/64）；连续神经步与两帧状态/相位一致；detach 保持全部值且移除梯度；动作损失产生非零内部 theta 梯度，实际联合更新 theta/weight/bias 后可继续 TBPTT。
  - 其余检查覆盖：完整 DN 顺序/未知类型、输入只驱动注册细胞、构造不预处理/不改全局 RNG、固定背景、图/映射角色与绑定、相机/精度/时间/状态拒绝、异常不前进、controller 显式 reset 和忽略观察误差/有效性、无像素跳接。
- 未启动完整图模型或硬件，未运行旧套件、构建或全量检查；未改变 Noria/Docker、资源预留和其他服务。

## Known Limitations

这是软件接口通过，不是完整图策略学习证据。持续输入实测及训练窗口由协调者另行处理；本批未训练任务、未拟合归一化、未保存新策略 checkpoint、未做拓扑/冻结脑等任务对照、未检验实时性或实飞。

## Remaining Questions

无阻碍本批数值接口的未决问题。后续实际训练协议、完整策略 checkpoint 和评估仍按总 Spec 推进；由新的独立验收 Agent 判断本批是否符合 P2e。
