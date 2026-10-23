# M3 冻结缓存恢复独立验收

日期：2026-10-23。验收者：`roadmap_m5_video_final_review`，未参与 M3 缓存恢复实施。结论：**FAIL，存在 1 项训练数据使用阶段未绑定登记身份的问题；修复前不运行正式恢复。** 风险 S3。仅新增本报告，不修改业务代码、旧 run 或训练资产，不 commit。

已阅读 Ready [契约](m3-cache-recovery-contract.md)、[只读可行性](m3-cache-recovery-readiness.md)、[实施报告](m3-cache-recovery-implementation.md)，并审阅 CLI、训练编排、缓存模块、相关 CPU 测试及被调用的数据/checkpoint/模型构造路径。

## R1：worker 使用的新 run 数据没有绑定已核对的数据身份

**[P1] 阻断工程验收。** `learning/x2_behavior_training.py:675–676` 的 worker 重新核对原缓存并得到已验证特征，但丢弃 `audit_cache()` 返回的 dataset 字节。随后 `partition()`（717–720）从新 run 的 `dataset/` 重新读取数据。当前 `load_split()`（`learning/x2_behavior_data.py:200–213`）仅核对该目录自身 manifest 与 split 的一致性，不接受或核对新 registration 中的预期 dataset 身份。

缓存分支在 `x2_behavior_training.py:840–841` 只比较 episode ID 顺序，没有比较实际 Observation、target、完整 frame 序列或 dataset ID。因此，在父进程完成复制后、worker 使用前，如果新 run 的 `dataset/train.json` 与其 manifest 被一并替换为内部 hash 自洽的版本，只要保留 episode ID/顺序，原缓存复验仍通过，随后训练却能使用不同标签或帧。`cached_outputs()` 还用 `zip(episode['frames'], used)`，帧数变化可被截短而不由该分支拒绝。最终 200 次更新/200 条 history 计数不能补上数据身份核对。

这违反本次恢复必须保持原 50/50 场景、818 帧完整对应及“使用已经核对的文件字节”的约束；即使特征本身全部正确，也不能把不同 Observation/标签配到原神经特征后宣称复现原冻结读出实验。该缺口不要求读取 locked-test 或修改原失败 run。

本验收只为这个疑点执行了一次最小 CPU 复现：在临时目录建立五个任务的短数据文件，记住原 manifest 和 episode 顺序；把首条 target 从 HOLD 改为 ADVANCE，重算该目录自身 split hash、dataset ID 和 manifest；调用**当前生产 `load_split()` 与 `select_episodes()`**。输出为：

```json
{"replacement_accepted_by_load_split": true,
 "cache_episode_order_check_passes": true,
 "registered_dataset_identity_unchanged": false,
 "loaded_changed_target": "ADVANCE"}
```

该复现只读当前代码、写随后删除的临时小文件，没有读取真实缓存、加载图、导入 Torch、执行训练或计算任何神经状态。它验证数据加载与当前唯一 episode 顺序门槛会接受替换；没有冒称启动了完整 worker。

限定修复要求：缓存 worker 的训练和验证必须从本次复验已经校验的同一 dataset 字节解析并使用，或在实际读取时把同一字节与 registration/cache manifest 的预期身份和完整文件 hash 严格绑定；不能只提前检查磁盘文件后再次不受约束地读取。保留原选择、读出循环、参数、预算、旧 run 失败及禁止 locked-test 规则。增加一个针对“新 run split 与 manifest 一起自洽替换、ID 顺序不变”的最小拒绝/隔离用例即可，不要求重跑完整缓存或已有数值检查。按 AGENTS 交新的实施者限定返工，再由新的独立验收者复核；本验收不代修。

## 已审阅且未发现另一项直接阻断的问题

- **原失败与初始化**：只允许固定 r1 wall_limit 且无 worker_result，原目录只读；固定 r2 initial 的 checkpoint ID、七组参数、无 optimizer 数组。新 output 排他创建、新 registration；恢复后 fresh Adam、冻结 theta，从第 1 步做固定 200 次，不取旧第 149/168 步参数。
- **缓存本体完整性**：原 registration canonical ID、37 份来源快照及允许的新来源差异受约束；policy/input-map、runtime、配置与 dataset 身份绑定。按固定顺序审计全部 818 frame / 918 state，核对数组名、dtype/shape、有限性、NPZ/逐数组 hash、Observation/编码、decision/checkpoint、session/步数及逐帧状态链。100 个初始状态有相同数组身份及登记预热步数。这里只判断代码路径，不冒称本轮实际重新审计了 2.5 GB 真实状态。
- **特征使用与零神经调用**：worker 保留从已核对的同一 NPZ 字节提取的 features_mv，原状态仅审计；保护在真实图/策略构造之前安装，拒绝 policy forward/initial_state/_step 和 brain reset/step 等入口。构造路径本身不创建 LIF 动态状态；缓存分支不调用预热或特征重算函数。成功终态要求神经调用计数为零。
- **固定读出语义**：AST 约束保留原 loss、数值辅助函数、cached_outputs 及整个读出更新循环；按 50 个训练 episode 四轮共 200 次更新，每步完整验证，严格小于才替换最低 loss，平局保留最早。最终恢复所选 checkpoint 再评分。R1 未关闭前，这不保证实际传入的数据仍是原 402 验证帧。
- **预算及原路径**：缓存要求 600 秒，父准备/准入已计入开始时间，文件审计前检查剩余预算，再把剩余时间交给既有 worker supervisor；未增大原运行预算。没有 feature_cache 时保留原数据生成、特征计算及其他模式分支；缓存只允许 readout，不进入 locked_test 分支。系统预留、单完整脑锁及仿真预检保持原路径。

## 证据、限制与停止

继承实施者一次 `tests/test_x2_behavior_cache.py` 的 **8 项通过、6.26 秒**，并核对这些测试内容；没有重复执行。完整 fixture 覆盖原缓存本体与 sealed 清单，CPU 读出循环覆盖 200 次更新/最早选择，但没有覆盖本次发现的新 run 数据替换边界。本验收只执行上述一个有具体疑点的小型 CPU 复现，未运行 build、全仓测试、真实缓存审计、完整图、MPS、训练或仿真。

正式恢复尚未运行；原 r1 仍失败，locked-test 未打开。工程 FAIL 不改变已有 M5b PASS，也不影响其他未修改阶段；P3/P4/P5 与上海真机均没有新的通过结论。问题已具体定位，本报告完成即停止。
