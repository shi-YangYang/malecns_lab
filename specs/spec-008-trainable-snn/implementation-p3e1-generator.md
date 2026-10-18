# Implementation Report — P3e1 完整 R 有界续段生成器

## Summary

按 Ready [P3e1](p3e1-bounded-rewiring-draft.md) 实现独立显式分段入口、单一共享提案内核、实际 CSC/完整 PCG64 归档、恢复证明、父链及终态准入。正式入口只接受本项目已核验的完整 MaleCNS 母本身份，固定 seed 1901、成功上限 25,582,837 和提案上限 102,331,348；每次显式 1–1000000 提案，保留 1 GiB 额外预留、6 GiB 申请、600 秒及 0.1 秒采样和共享 OS 锁。

原 R 单次入口调用同一提案内核，保留有放回边位置抽样、拒绝次序和列内权重移动。新增完整逐源权重多重集检查，最终保存完整策略输入/DN双多源距离及全部节点正/负/绝对入强度变化向量。中途状态独立 schema/purpose，不能成为脑资产。

本报告不是独立验收。没有启动正式 R 生成、完整图加载、神经模型或设备；首两段实际门槛由协调者在新的独立软件验收及提交后执行。

## Files Changed

- `src/cnsbridge/learning/rewiring_steps.py`：单一逐提案步进、完整 PCG64 验证、严格非布尔累计计数和边界报告。
- `src/cnsbridge/learning/graph_controls.py`：旧 R 调用共享核心；新来源身份覆盖全部生成调用源码，保留旧 R/W/P 公共入口。
- `src/cnsbridge/learning/graph_diagnostics.py`：逐源权重多重集；完整策略输入/DN的两个有向多源最短跳数；逐目标符号强度向量。
- `src/cnsbridge/learning/rewiring_contract.py`：正式完整母本绑定、固定协议、源码/运行环境身份；父仅读取小元数据与流式文件摘要。
- `src/cnsbridge/learning/rewiring_archive.py`：真实 CSC 三数组、PCG64、累计计数及父 context；最后 manifest 封存；worker 恢复逐数组/结构/不变量验证。
- `src/cnsbridge/learning/rewiring_evidence.py`：父独立资源/文件/恢复证明/计数增量/终局证据校验及完整前缀扫描。
- `src/cnsbridge/learning/rewiring_worker.py`：受监督母本加载、提案、归档、最终全部机械诊断和研究资产实际重载。
- `src/cnsbridge/learning/generate_rewiring.py`：显式单段 CLI、共享 OS 锁、原始监督结果保留与父验收；`accepted_candidate()` 作为后续正式 R 的唯一父批准终局准入。
- `tests/test_bounded_rewiring.py`：26 项新低成本算法/持久化/终态检查。
- `docs/BOUNDED_REWIRING.md`：准入、命令、状态及边界。

协调者的 `.ai/prompts/task-handoff.md` 与 `p3c-continuation-20261019.md` 改动不是本实施产出。没有更改 29 个当前 G 训练绑定源码；实施开始及结束依据真实 run 清单逐文件 SHA 核对，mismatches 均为空。没有删除旧探针协议/脚本/数据、环境、许可、历史结果或业务基线；被替代的旧重复交换循环已移除。

## Important Decisions

- 公共旧入口和续段只保留一份提案算法。新测试中的冻结旧算法只作独立等价参考，不是业务实现或替代脑。
- 正式机器登记严格固定当前已核验母本 SHA/source/graph identity；新低成本图夹具采用数值角色，不伪装为完整生物图。原有图工具的旧测试保留原状。
- 每次退出完整封存一次状态；未通过父验收的段不贡献 checkpoint/prefix，包括已经写出 checkpoint 或 candidate 的资源失败段。失败尾部保留，显式继续从最后父批准状态开始。
- 父意外结束而没有 `assessment.json` 时拒绝猜测终态；需先核实原进程/原始结果。这不是自动失败重试入口。
- 后续神经研究工厂必须经 `accepted_candidate()`；通用旧 `load_research_control` 保持既有离线 R/W/P 兼容语义，不能单独作为正式生成终局证据。
- 完整中间 CSC 使用非压缩 NPZ，减少压缩 CPU 开销；每段文件均保留，有相应磁盘成本，不自动清理唯一前缀或失败记录。
- 每个完整图加载、恢复不变量、最终诊断和实际重载都位于受监督 worker；父读小元数据并流式核对来源文件。未引入 Torch、研究 SNN 工厂、优化器或评估改动。

## Validation

- Risk Level：**S3**，持久化、跨进程状态及共享图逻辑；仅相关子系统。
- 数值检查前向协调者申请独占计算窗口，待完整 G `session10217` 已收取 exit 0 后获得许可；没有并行完整脑或数值模型运行。
- `PYTHONPATH=src .venv-train/bin/python -m pytest -q tests/test_bounded_rewiring.py tests/test_graph_controls_v2.py`：**30 passed in 0.43s，exit 0**（19 新 + 11 既有）。覆盖旧提案参考/单次入口/真实新 OS 进程分段的最终 CSC 和完整 RNG/计数精确一致、双边界、零/空图及所有拒绝种类、全局多重集与强度保持而逐源多重集改变的反例、有向距离/不可达、完整强度向量、母本/来源/runtime/RNG/布尔计数/守恒/父 context/数组损坏、manifest-last 保存中断、生产 loader 拒绝中间状态、父链、资源失败及仅退出零不能批准。
- 针对验收要求中尚缺的“重新封摘要后的实际 CSC 结构”和“父终局批准”补齐 7 项检查，仅执行新增项：`PYTHONPATH=src .venv-train/bin/python -m pytest -q tests/test_bounded_rewiring.py -k 'structural_csc or terminal_parent or parent_rechecks_source'`：**7 passed, 19 deselected in 0.07s，exit 0**。覆盖重复/越界 target、零权重、错误 dtype/指针（文件与数组摘要均重新封存后仍拒绝）、缺父 seal/终局后继续/输出改变，以及 worker 退出后再次核对来源。
- 总计 **37 个不同检查已通过**。未重复首轮通过项；没有执行全项目测试、lint、build、完整图或旧模型实测。两个检查命令均直接返回终态，没有遗留 test session。
- 最后阅读范围 diff，并按真实 run 的 29 源绑定 SHA 核对无变化；检查通过后停止验证。

## Known Limitations

软件夹具不证明完整规模的新存档/诊断成本；尚未执行正式首段/第二段 100000 提案跨 worker 门槛。每段规范 CSC 文件约与完整图数组同量级，累计磁盘容量及最终诊断 wall/RSS 要由实际监督记录说明。资源采样不是硬 OS 沙箱。

混合充分性、神经传播、学习收益、G/R 比较、闭环和实体控制均未验证；本生成器不会关闭 AC6。测试中的父终局准入夹具使用明确标为非图的 opaque 文件，验证身份/终态逻辑；完整生物图机制检查仍由 worker 中的真实验证及后续实际门槛承担，不能把该夹具称为完整图成功。

## Remaining Questions

无阻塞实施的问题。请协调者安排新的独立软件验收；仅在其通过并提交后执行协议规定的实际两段门槛。实施者未创建 commit，未修改系统服务，也未启动付费或硬件工作。
