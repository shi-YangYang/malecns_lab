# Implementation Report — P0 完整神经元参考图

## Summary

已按作者 Methods 的操作性 neuron 定义，从官方 v1.0 完整 flat-connectome 母本构建独立 V2 参考资产。身份依据来自作者 bioRxiv v2 / PMC Methods，而非旧图数量或未取得的 Cell 全文。所有非空 superclass 的 body ID 与独立身份清单逐项一致；不再加 type、status、侧别或度数过滤，不剪弱边、自环或孤立节点。

真实构建结果如下，均来自本次运行，并非预设验收数字：

| 项目 | 实测 |
| --- | ---: |
| 神经元 | 166,700 |
| 全部内部有向边 | 25,582,938 |
| 精确整数突触数 | 124,177,617 |
| 保留自环 / 自环突触 | 101 / 474 |
| 保留孤立节点 | 217 |
| 最小 / 最大原始正权重 | 1 / 2,591 |
| 原始 segment 边 / 突触 | 151,856,684 / 311,833,243 |
| 原始边端点的不同 segment ID | 88,384,522 |
| 构建耗时 | 95.7004 秒 |
| 峰值进程 RSS | 1,307,279,360 bytes，约 1.22 GiB |
| 参考 NPZ 大小 | 84,301,745 bytes |

原始端点按身份分为 166,576 个作者定义神经元、3,292 个标注 Glia、21,828 个其他已标注但不符合 neuron 定义的 body，以及 88,192,826 个未标注 segment。后者不能被称为额外完整神经元。所有原始边和节点均有可回查索引；124 个参考神经元没有任何原始表端点，仍保留于图中。

## Files Changed

- `src/cnsbridge/brain/metadata.py`：完整 11 文件来源/用途登记、当前消费的三张表 MD5/SHA256、外部身份清单核验、nullable 注释、NT 符号假设、精确群组与输出 body ID。
- `src/cnsbridge/brain/reference_audit.py`：按原始全表行号保存 uint8 边分类；64 个磁盘桶对原始端点去重并保存身份类别，控制内存；记录逐批偏移及未出现在原始边中的注释 ID。
- `src/cnsbridge/brain/reference.py`：独立构建 CLI、int64 CSC、完整图和元数据身份、加载校验、拒绝已有输出目录、失败报告、实际统计与资源记录。
- `src/cnsbridge/brain/connectome.py`：增加原始 nullable 注释、显式字段选组；V2 保存 int64，旧格式继续可读；突触求和避免旧 float32 总和舍入；保存接口可拒绝覆盖。
- `src/cnsbridge/brain/brain.py`：生产加载及直接 Brain 构造拒绝 `research_control`；完整参考加载/直接构造验证身份，错误使用 V1 群组配置失败，裁剪后继承旧来源身份的图也失败。
- `configs/malecns_v2.yaml`：独立 V2 参考路径，R16 使用 rootSide、HAL 使用 subclass/rootSide，其余显式 somaSide；DNg02 工程偏置 8 mV。
- `tests/test_full_reference.py`：小型真实列模式 fixture 仅验证保留规则、精确整数、nullable 元数据、来源/ID 失败、保存/加载及生产隔离；不替代实际脑实验。

旧 `data/malecns_brain.npz`、`data/core.npz` 和所有官方源文件未覆盖。没有改旧默认配置、恢复 MiniFly、连接硬件或提交 Git。

## Important Decisions

1. 身份入口要求独立证据目录中的 ID 清单 SHA256、当前注释 SHA256 和 generation/MD5 一致，再按完整注释逐 ID 对账。官方定义应用到 v1.0 得到 166,700，而非按旧数量筛到该结果。缺 type 的 2,194 个、非 Traced 的 2,094 个、tbc 的 94 个均保留。516 个 Traced 但无 superclass 的边界记录随身份报告保留。
2. 原始边的 `raw_edge_classes.u8` 每字节对应一个零基源行；分类码由 pre/post 四种身份及正/零权重组成。`graph_ledger.json` 保存源 URL、generation、校验和、批偏移和完整选择规则，`code == 0` 恰为参考图全部正内部边。可从原始 feather 精确回查受影响边，无需复制 151M 条边表。
3. 完整 11 份发布数据检查文件存在、大小、原校验记录和固定 generation；本次实际使用的注释、递质、完整 weights 再算 MD5/SHA256。其余大型文件保留原已验证校验记录，不无理由重 hash 31 GB；重复表示和官方派生子集不重复加入图。
4. 原始 `consensus_nt` 与 null mask 单独保存。3,177 个未识别/缺失递质节点显式假设正号，其中 178 个值缺失；胺类没有额外弱化幅值。符号是模型假设，原始突触数量仍可通过有符号 int64 的绝对值恢复。
5. 保存 `type/class/subclass/status/statusLabel/somaSide/rootSide/superclass` 的原始值与缺失掩码，不互相填充。其余原始字段仍保存在不可改写的完整注释母表中，可用 body ID 回查。
6. R16 左右覆盖为 1,112/2,265；HAL 为 104/101。HAL 主实验禁用由配置和群组协议明确，实际输入编码将在 P1 实现。六个输出群仍为 15/14、1/1、1/1，33 个 body ID 数值升序固定；没有裁掉其余脑区。
7. 真实图本次峰值内存约 1.22 GiB，没有资源理由触发删边。未做阈值 3 优势、行为作用或实时性能推断。

## Validation

- Risk Level: **S3**，依据为核心图资产/身份持久化和跨 Brain 加载边界，验证限于相关模块。
- `.venv/bin/python -m pytest tests/test_full_reference.py tests/test_malecns_loader.py tests/test_brain.py -q`：**9 passed**，4 条已知旧 HAL/fixture 空群组警告。含旧真实图短步/状态隔离检查。
- 随后发现直接构造裁剪图可能保留原完整图身份，增加直接 Brain 身份校验及对应断言；只重跑 `.venv/bin/python -m pytest tests/test_full_reference.py::test_complete_reference_retains_edges_identity_and_nullable_metadata -q`：**1 passed**。未重复其他已通过检查。
- `.venv/bin/python -m ruff check ...`：**未执行成功**，环境没有 ruff 模块；未为本批安装依赖。`git diff --check` 无差异格式错误。
- 实际完整构建仅运行一次：

  ```sh
  .venv/bin/python -m cnsbridge.brain.reference \
    --data-dir data/malecns_v1 \
    --identity-dir data/audits/malecns_v2_identity \
    --out-dir data/malecns_v2
  ```

  同一目录已存在时拒绝重建。需要复现时指定新的输出目录，不删除或覆盖参考资产。
- 实际保存后加载校验一次：图哈希、nullable 元数据哈希、来源身份、选组和整数计数一致；使用 V2 配置成功创建 Brain，无空群组。未推进 V2 仿真或发设备指令。证据 `data/malecns_v2/load_check.json`。
- 来源 ID：`873a58e99f5347f8bd5d3bee76e718bdccb340a67f070dc4f837f02262b41342`。
- 规范图哈希：`af12b8dc8775f828740221f92c870d76450266759bca6f1540ff5e27002ac156`。
- 参考资产 SHA256：`98f83158ed12f0c836fb085c2e286e54f9458630c2a158bee3164af52648f6f3`。

完整事实产物位于被 Git 忽略的 `data/malecns_v2/`：`build_report.json`、`graph_ledger.json`、`group_mapping.json`、`download_manifest.json`、`full_reference.npz`、`identity_evidence/`、`raw_edge_classes.u8`、`raw_nodes/`、`annotated_nodes_without_raw_edges.npz`、`isolated_neuron_ids.npy`、`load_check.json`。

## Known Limitations

完整指当前已发布 minconf 0.5 数据下、作者操作性 neuron 集合的全部内部非零连接；不表示阈值前原始观测、每个细胞形态完全、全部生物神经元或完整生理动力学。官方 Methods 身份依据的 Cell 全文访问限制保留在身份报告中。

P0 只交付完整参考图和可追溯边界。P1 的因果输入/共同事件/神经时间契约、P2 的掩码学习、P3 的对照及真实动作证据未由本批实现。M1–M4 不应因 P0 构图通过就全部勾选。研究专用图加载与仿真入口留待 P3，当前生产拒绝是已落实的边界。

## Remaining Questions

本批无需要用户决策的阻塞。待新的独立验收 Agent 审核；实施者不填写最终验收 PASS。
