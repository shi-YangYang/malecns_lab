# Implementation Report — P2b 完整神经集合与空间事件接口

## Summary

按 [P2b](p2b-mapping.md) 实现独立的完整输入／输出集合、工程空间映射和可连续恢复的事件编码。没有修改 P0 核心、P1 探针、现有参考图群组、任务环境或硬件链；未加载完整图或运行神经动力学。

## Files Changed

- `src/cnsbridge/brain/policy_groups.py`：生产入口需要已验证完整 `TrainingGraph`；懒读 NPZ 元数据并沿用现有 `annotation_hash` 契约，核对来源、注释和 ID 顺序，选择全部精确 `R1-R6` 输入和 `descending_neuron` 输出。
- `src/cnsbridge/senses/spatial_drive.py`：rootSide 分组、PCG64(1701) 均衡空间分配、PCG64(1702) 初始相位、连续事件计数、原子恢复、稀疏增量及按需完整节点向量。
- `tests/test_policy_mapping.py`：显式 numerical_test_only 接口夹具，元数据身份绑定、完整成员规则、均衡与空间区分、相位／计数、错误状态和输入的检查。
- `docs/NEURAL_INTERFACE.md`：接口、精确算法、来源／工程边界、状态契约及静默问题。
- 本实施报告。

## Important Decisions

- 不修改原参考资产的旧 33 细胞群组。新 PolicyNeuronMap 身份独立，保存实际成员、图内索引及 rootSide/somaSide 缺失信息；缺失输出类型不影响 superclass 选择。
- 注释检查明确依赖先前完整图校验，不读取 NPZ 的 data/indices/indptr，也不把元数据校验称为新的完整图校验。接口夹具的元数据 NPZ 故意不含权重数组，以验证该读取边界。
- 空缺 rootSide 和原始空字符串分开分组；缺失组先处理，其他字符串按 Unicode 排序，各组共用单个 PCG64(1701) 流。成员按 body ID 升序后置换，置换位置 j 对应 j % 192 格，覆盖缺口显式记录。
- 默认刺激为 dt 0.1 ms、0–150 Hz、事件 68.75 mV；显式配置包含于驱动身份，rate × dt > 1 的配置拒绝。没有背景活动或输出偏置。
- 映射、选择、导出状态和事件数组采用不可改写的 bytes backing；身份属性返回独立对象。非法 mask／恢复在状态更改前拒绝。完整节点数组／CPU 张量只在调用者请求时分配。

## Validation

- Risk Level: S2（元数据身份与跨模块输入契约）。
- Checks Performed: `PYTHONPATH=src .venv-train/bin/python -m pytest tests/test_policy_mapping.py -q`。
- Result: **29 passed in 0.75s**。包括精确成员／缺失侧别、生产拒绝数值图、空集合、元数据源／摘要／ID 错配、缺权重元数据读入、均衡映射、四角空间区分、相位连续／计数／重置／恢复、坏输入和坏恢复不推进，以及按需 CPU 张量。
- 只检查本批新增文件的差异格式；未重复 P0、P1、环境或全套检查。

## Known Limitations

- 本批没有实际完整图加载、完整脑响应、动作训练、部署延迟或硬件证据；生产完整资产的成功构造仍需在资源门槛满足后验证。
- 空间映射是明确的工程编码，不是解剖 retinotopy。默认刺激常数也不是测得的神经生理参数。
- [可观测性记录](signal-observability.md) 所述抑制性 R1–R6 与无背景初态问题仍存在；本批没有添加偏置或其他外部驱动来制造输出。
- 当前只实现事件编码和输出集合，不接动作读出、教师真值或目标误差直连。

## Remaining Questions

无本批实现阻塞项。完整图实测、背景工作状态／分级电位模型的独立决策和训练集成由协调者按后续登记推进；需要新的独立验收，本报告不代表最终验收。
