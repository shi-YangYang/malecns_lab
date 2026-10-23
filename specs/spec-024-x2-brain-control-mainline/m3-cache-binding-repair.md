# M3 缓存数据绑定限定返工

2026-10-23。实施者：`roadmap_m3_cache_binding_repair`，未参与前次缓存实施与验收。本次仅处理 [独立验收 R1](m3-cache-recovery-acceptance.md) 的 P1：worker 丢弃已审 dataset 字节、改读新 run 自洽但未绑定原身份的副本。风险沿用 S3（训练数据身份与恢复）；实施完成，等待新的独立验收，**未执行正式恢复**。

## 实际修改

- `src/malecns_lab/learning/x2_behavior_training.py` 保留本次 worker `audit_cache(..., expected=cache_manifest, keep_features=True)` 返回的 `cached_dataset_bytes`。
- 缓存分支的真实 `partition()` 只允许 train/validation，并直接从这些已审内存字节解析完整分区。新 run 的 `dataset/` 副本保留归档职责，不再作为缓存训练的数据输入；审计后替换原磁盘文件或新副本均不会替换已经保留的 bytes。
- 原 `select_episodes()` 继续对完整分区执行同样的 task-round-robin 选择；因此标签、Observation、全部帧和先后次序与本次缓存审计使用的数据相同。没有重写或放宽数据/缓存协议，也没有再读文件形成第二次不受约束的输入窗口。
- 不提供 feature cache 的路径仍调用原 `load_split()`。未改 policy、encoder、loss、200 次更新循环、验证/最早平局选择、预算、source allowlist 或任何缓存数值身份。

业务代码只改上述训练文件；测试只新增 `tests/test_x2_behavior_cache.py` 的一个直接回归，没有修改缓存模块或其他实施者文件。

## 定向检查

执行的唯一检查范围为：

```sh
PYTHONPATH=src .venv-train/bin/python -m pytest tests/test_x2_behavior_cache.py::test_worker_uses_revalidated_dataset_bytes_despite_self_consistent_run_replacement -q
```

最终 **1 passed，1.21 秒**。首次同一命令失败于测试期望值的 JSON 表示：内存 Observation 的 tuple 与解析后的 list 被直接比较；将期望值改为从原审核字节解析后，只重跑该项。生产修复没有因此扩大。未重复此前已通过的 8 项测试。

回归直接调用生产 `run_worker()`，经过实际缓存返回值接收、两个 `partition()` 和原 `select_episodes()`。临时 train/validation 各 50 条、任务分组顺序的资料被复制后，同时改动标签、Observation、帧数，并重算 sequence hash、split hash、dataset ID 和 manifest；episode ID 与选择顺序不变。首先确认原生产 `load_split()` 能接受该自洽替换，再禁止 worker 再次调用此磁盘读取函数。最终两个实际选择结果逐项等于原已审字节中的完整数据，且保留 round-robin 选择语义。

为隔离本次边界，测试替换了缓存大文件审计、图/checkpoint 构造、资源采样和父锁检查，仅用 CPU 头部 fixture 通过初始化；缓存审计 stub 返回固定的已审 bytes，未冒称重新验证缓存本体。第二个分区选择完成即用明确 fixture 异常终止 worker，检查 `opened_splits == [train, validation]`、`optimizer_steps == 0`、神经入口计数为零。没有执行任何读出更新、完整图/MPS 计算、真实缓存审计或仿真。

## 限制与交付

已核对本次补丁范围，仅数据字节保留/消费分支及直接回归发生变化。旧 r1、r2 initial、约 2.5 GB 状态资产和 locked-test 均未读取或改写；测试只在临时目录写 fixture。未 commit/push，没有产生新模型或行为能力结论。

R1 的工程修复须由新的独立验收者复核，通过后才可由协调者登记并运行一次 600 秒恢复。本报告不把原 r1 改为成功，不替代正式恢复结果，也不改变 joint/P3/P4/P5 或上海现场门槛。实施到此停止。
