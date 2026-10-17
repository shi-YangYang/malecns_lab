# Implementation Report — P0 可训练核心

## Summary

实现独立的 `cnsbridge.trainable_lif.v1` CPU 数值核心，生产入口通过 `load_reference` 校验完整图。保留精确原始有符号整数计数，固定稀疏传播矩阵按突触前原始类型乘有界可学习增益；动态状态显式包含电压、突触量、不应期、延迟队列与步号。小电路前向、反向与一次内部更新检查通过；没有执行完整图训练、任务训练或设备动作。本报告不声明整个 Spec 008 完成。

## Files Changed

- `src/cnsbridge/brain/trainable_graph.py`：完整参考适配、只读原始计数、类型分组、来源与分组身份；明确隔离数值测试图。
- `src/cnsbridge/brain/trainable.py`：精确线性积分、显式事件顺序、严格阈值、代理梯度、固定 CSR 全细胞伴随、内部类型增益、reset/detach/step/run 与模型身份。
- `tests/test_trainable_core.py`：独立矩阵指数参考、手工事件边界、稀疏/密集代数、静默细胞梯度、内部更新/恢复、分组、状态连续性与拒绝路径。
- `pyproject.toml`：可选 `train` 依赖，固定 torch 2.14.1，并声明 psutil。
- `docs/TRAINABLE_SNN.md`：环境、用法、数值/梯度契约、来源摘要、工程差异和证据边界。

其他同时出现的治理、清理与目标环境改动不属于本实施批；没有触碰旧 `lif.py`、V2 或设备环境。

## Important Decisions

- 默认 dt=0.1 ms，发放步 k 的增量在 k+delay_steps 到达，不应期细胞于 k+ref_steps 再次允许发放；外部输入集合固定，不设置不应期。
- 前向二值，反向为登记的 fast-sigmoid；复位和不应期门停止梯度，递归脉冲与延迟队列保留梯度。固定 CSR 乘法的反向对全部 presynaptic 分量计算，不根据实际发放索引裁剪。
- 缺失、空或全空白类型进入组 0；其他原始字符串按 Unicode 顺序排列。theta 初值 0；数学增益为 1+0.5*tanh(theta)。以 nextafter 处理浮点端点饱和，保持严格 0.5<gain<1.5，已写入身份。
- 官方 model.py 只作为方程/参数来源。2026-10-17 获取 SHA256 `fc45837d7122c6ce2a7f3f2f23c515992e4b232aadb919efabb72337fac88e4e`；未声称 Brian2 调度逐步复现。
- 稀疏图、类型映射和输入掩码为非持久化固定 buffer；`state_dict` 仅含 theta。后续 checkpoint 必须保存并验证完整模型身份与参考图，不能只依赖裸 state_dict。

## Validation

- Risk Level: S3（新数值和梯度核心），仅执行对应新增模块检查。
- `PYTHONPATH=src .venv-train/bin/python -m pytest tests/test_trainable_core.py -q`：**19 passed**，1.17 s，1 条 PyTorch CSR beta 功能状态提示。测试没有加载或运行完整脑，也没有硬件操作。
- 初跑 3 个夹具错误已修正：独立 NumPy 参考的电压数组因整型默认值被推断为整数；手工时间表输入未显式指定 float64。修正后同一新增文件全部通过。没有更改实现来迎合错误参考。
- 独立参考使用 `scipy.linalg.expm` 的 2×2 线性系统，与核心解析式独立；包含真实递归发放、抑制到达、相等时间常数。手工时间表覆盖严格阈值、不应期冻结且保留到达、精确开放步及输入细胞豁免。
- 代数参考直接比较 CSR 前向和 dense matrix product、伴随和转置乘积；另外从静默细胞的显式输入反传经过延迟传播，防止仅测试稀疏函数而漏掉完整链路。
- 内部 gain 更新检查实际执行 SGD，确认递归响应改变，并在恢复 theta 后恢复原响应。
- 生产拒绝检查覆盖过滤资产、研究资产和缺失路径；真实完整图加载/资源/更新由 P1 单独实测，未以数值夹具替代。
- `git diff --check` 通过。未运行全项目测试、旧全脑诊断、build 或 lint。

## Known Limitations

完整图在新后端的时间、峰值内存和一次内部学习尚待 P1；小图检查只支持数值/梯度结论。当前无任务编码器、动作读出、checkpoint 管理或训练循环。仅 CPU 单序列，float32/64；未评估 MPS。PyTorch 自身将 CSR 支持标记为 beta。

训练环境版本由本机查询记录：Python 3.14.6、torch 2.14.1、NumPy 2.5.4、SciPy 1.18.1、PyYAML 6.0.3、pytest 9.1.1、psutil 7.2.2。由 root 创建和安装 `.venv-train`；本实施未运行 pip，既有 Tello `.venv` 未修改。

## Remaining Questions

无 P0 关键歧义。待新的独立验收 Agent 判断 P0 覆盖，再由 root 阶段提交；P1 需登记全图探针预算并保存实测证据，不能将本批测试当成 AC2 完成。
