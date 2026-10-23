# M3 冻结缓存恢复返工独立验收

2026-10-23。验收者：`roadmap_m3_cache_final_review`，未参与缓存初版实施、初审或此次返工。结论：**工程 PASS，初审唯一 P1（R1）已关闭。** 风险沿用 S3（训练数据身份与恢复）。只新增本报告，不修改业务代码或科研资产，不 commit/push。

已阅读项目入口、Ready [缓存恢复契约](m3-cache-recovery-contract.md)、[初版实施报告](m3-cache-recovery-implementation.md)、[初审 FAIL](m3-cache-recovery-acceptance.md)与[限定返工报告](m3-cache-binding-repair.md)，复核实际缓存审计返回值、worker 分区读取、选择及读出路径和直接回归用例。本次结论是允许协调者登记后执行一次既定 600 秒恢复，不是实际训练完成或任务能力通过。

## R1 关闭依据

- `learning/x2_behavior_cache.py` 的 `audit_cache()` 保留从原缓存实际读入的 train/validation `bytes`，用这些字节解析并校验 split 文件 hash、原 manifest/registration 身份、完整场景内容、固定选择顺序及与帧/状态的对应；worker 传入封存 `expected`，逐文件 hash 与最终清单均须相等。其返回的 `dataset_bytes` 就是审计所用字节，不是重新读取的路径或延迟读取器。
- `learning/x2_behavior_training.py` 的 worker 现在接收并保留第三个返回值 `cached_dataset_bytes`。实际 `partition()` 对缓存分支只允许 train/validation，直接 `json.loads(cached_dataset_bytes[name + ".json"])`；随后实际训练和验证均调用既有 `select_episodes()`。因此 target、Observation、全部帧及次序与本次审计输入一致，原缓存或新 run 磁盘副本在审计后被替换，不会替换已保留的输入字节。新 run 副本属于归档，不再给缓存读出提供训练数据；本结论不声称磁盘归档不可变。
- 直接回归 `test_worker_uses_revalidated_dataset_bytes_despite_self_consistent_run_replacement` 调用生产 `run_worker()`，用 episode ID/顺序不变但标签、Observation、帧数改变且重算全部自洽身份的磁盘副本复现旧边界。它先确认旧 `load_split()` 会接受该副本，再禁止 worker 调用磁盘分区加载；实际两个分区选择必须完整等于已审字节的选择结果。第二次选择后明确终止于 fixture 边界，检查只打开 train/validation、更新数为 0、神经调用计数为 0。测试替换大文件审计、图和资源边界的限制已准确披露，没有把 fixture 当作真实缓存审计。

## 实验语义与范围

- 固定 r1 失败身份、r2 initial checkpoint、全部七组初始参数、fresh Adam、冻结 theta、50/50 task-round-robin、200 次更新、每次完整验证、最低 loss 与最早平局选择的路径仍在。缓存准入继续拒绝 joint/evaluate、其他配置及 locked-test；没有失败后重算神经特征的分支。
- 此返工不改变神经入口保护、`features_mv` 输入、数值策略、loss、checkpoint 与参数审计。缓存分支仍在策略构造前安装禁止 forward/initial_state/_step 等入口的保护，成功终态仍要求 200 次更新/200 条 history、两个既定分区及零神经调用。
- 未提供 feature cache 时 `cached_dataset_bytes` 保持 `None`，`partition()` 仍执行原 `load_split(directory / "dataset", name)`；本修复没有替换非缓存路径的数据加载行为。
- 对原 r1 只读源码快照与当前训练文件做了一次标准库 AST 静态比较：12 个受约束辅助函数集合完整且全部相同；唯一 `readout_update` 循环 AST 完全相同。生产 `_source_changes()` 的相同准入检查仍保留。没有为了比对调用训练模块、Torch 或缓存审计。

## 检查证据与停止

继承初版的 **8 项 CPU 检查通过（6.26 秒）**，以及返工的 **1 项直接 worker 回归通过（1.21 秒）**；本验收检查了用例内容，没有重复运行。返工首次运行仅因 tuple/list 的 fixture 期望表示失败，修正期望值后只重跑该项，记录没有掩盖此过程。

本验收唯一新增执行检查为上述两份源码的静态 AST 比较，输出 `helper_names_complete=true`、`unchanged_helpers=true`、`readout_loop_ast_identical=true`。未执行 pytest、build、全仓检查、真实约 2.5 GB 缓存审计、图加载、MPS、读出更新、仿真或 VM；locked-test 内容未打开，旧 run 与 checkpoint 未改写。

未发现本次 R1 修复仍有直接阻断。工程验收到此停止。下一步由协调者登记新目录及实际来源身份，仿真关闭时串行执行一次 600 秒恢复；实际准入、200 次更新终态、正式选择和零神经调用只能依据新 run 报告。原 r1 仍为失败；joint 仍待独立决定，P3/P4/P5 与上海真机没有新的通过结论。
