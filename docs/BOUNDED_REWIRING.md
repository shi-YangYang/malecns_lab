# 完整图 R 的显式续段生成

> 命名迁移：本文操作入口已更新为 `malecns_lab`。涉及旧 run／checkpoint 的目录保留为参数示例和历史证据；不能直接用新源码续跑或重新登记旧实验。须使用匹配的源码身份与新输出目录，精确恢复旧实验需原 commit。见[迁移说明](MIGRATION.md)。

本工具只生成离线拓扑对照，不训练 SNN、不连接设备。使用当前已核验的完整 MaleCNS 母本，固定 seed 1901、最多 25,582,837 次成功交换或 102,331,348 次提案，先到任一边界结束。交换次数和图边重叠不证明充分混合；零成功不构成有效拓扑反事实。

按 [P3e1 协议](../specs/spec-008-trainable-snn/p3e1-bounded-rewiring-draft.md)，软件独立验收及提交后，首段和第二段各 100000 次提案；第二段必须在新 worker 中验证真实恢复状态。两个实际段经过新的独立验收后，才能扩大每段额度至最多 1000000。没有自动循环或失败重试。

```bash
PYTHONPATH=src .venv-train/bin/python -m malecns_lab.learning.generate_rewiring \
  --graph data/malecns_v2/full_reference.npz \
  --out data/topology_generation_main_20261019_r1 --proposals 100000

PYTHONPATH=src .venv-train/bin/python -m malecns_lab.learning.generate_rewiring \
  --run data/topology_generation_main_20261019_r1 --proposals 100000
```

以上是命令示例，不是运行记录；目录必须是新目录，续段不能换母本、seed、总预算、源码或运行环境。每次调用共享训练 OS 锁，申请 6 GiB RSS，额外预留 1 GiB，墙钟上限 600 秒，每 0.1 秒采样。父进程仅检查文件及身份，完整图和机械计算在受监督 worker 中进行。保持 Noria 和 Docker 运行。

每段保留登记、资源记录、worker 原始结果、监督原始结果和父验收。`checkpoint/arrays.npz` 保存真实规范 CSC，`checkpoint.json` 最后写入，绑定完整 PCG64（含缓存）、累计计数、父 checkpoint、已接受前缀、源码、运行时和母本。恢复前重新检查实际数组、逐源权重多重集、自环、符号、度数、摘要和 RNG，`resume.json` 记录任何新提案之前的实际状态。

中途状态的 schema/purpose 与脑资产分开。失败或未封存的尾部全部保留，不因为磁盘上存在 checkpoint 或 candidate 就算通过；只有父 `assessment.json` 完成并认可的状态可续接。监督失败的段不贡献任何前缀。父进程意外终止、缺少 assessment 时入口会拒绝猜测终态，需要先核实原进程与原始证据。

到总边界后才保存 `candidate.npz`，并完成实际重载校验。最终报告包含完整节点、边、自环和逐源权重多重集不变量、边重叠、类型配对变化、全部节点正/负/绝对入强度变化。`mechanical_vectors.npz` 保存全部向量；路径采用完整 `type == R1-R6` 的 3377 个输入与 `superclass == descending_neuron` 的 1314 个 DN，分别计算每个 DN 从任一输入的最短有向跳数、每个输入到任一 DN 的最短有向跳数（反向邻接多源查询）。`-1` 表示不可达，不是全对路径或功能传播结果。

后续研究工厂应调用 `generate_rewiring.accepted_candidate(run_directory)` 获取唯一父批准的终局路径，不直接使用 worker 文件夹中的 NPZ。这个准入检查不加载脑，也不表示任何神经或任务优势。既有通用 `load_research_control` 保留旧 R/W/P 离线资产语义，不能单独证明本正式 R run 已完成。
