# Implementation Report — P3 独立图变换工具

## Summary

实现以已验证完整参考图为母本的 R 重连、W 源内权重置换，以及 P 出边干预与精确权重匹配对照。所有派生文件标记 `research_control`，只能由显式离线加载器结合独立母图校验后读取。本批不运行神经活动、训练、硬件或动作评测。

这是图工具实施报告，不代替独立验收，也不证明真实拓扑优势或交换链充分混合。

## Files Changed

- `src/cnsbridge/learning/graph_controls.py`：R/W 变换、P 配对、来源身份、不可覆盖写入、离线专用加载与 `python -m` 命令。
- `src/cnsbridge/learning/graph_diagnostics.py`：节点与元数据一致性、精确不变量、边集与类型间计数变化、可选定向最短路径诊断。
- `tests/test_graph_controls_v2.py`：11 个算法契约测试。小图仅作为离散算法 fixture，未作为脑模型或完整数据的替代。
- 本报告。

未修改 Brain、Connectome 容器、LIF、P1 感觉/重放器或 P2 训练接口。

## Important Decisions

### 身份与加载

公开构造入口首先调用 `load_reference` 验证完整参考图，不接受旧 V1、研究图链式派生或缺图回退。节点全集、升序 body ID、全部可空注释、缺失掩码与输入输出群组原样保留。

派生 provenance 保存母图 `source_id` 与 graph SHA256、变换/种子/显式参数、协议 ID、诊断、源码 SHA256 与派生图哈希；派生 `source_id` 覆盖整个 provenance。离线加载器重新验证母图、派生图内容、节点/群组/角色及相应变换不变量。没有新增通用 `allow_research` 开关。现有生产 `load_connectome` 对派生研究图的拒绝行为通过定向测试确认。

### R 与 W

R 均匀提议两个边位置，再拒绝同源、自环、不同源符号、同目标、新自环或重复边。成功时交换目标，权重留在原源节点；CSC 列保持排序，无逐边 Python 对象集合。尝试上限与目标成功数均必须显式传入，允许 0 次成功并如实记录，不因不利结果筛图。

R/W 均精确检查逐节点入/出度、逐目标正负入度、每源有符号及绝对出强度、全图有符号权重多重集、原自环与其权重。W 另要求拓扑完全不变，逐源只置换非自环权重，记录不可置换源及实际改变比例。

记录 R 尝试、成功、拒绝原因及 `2×成功交换数/非自环边数`；该比率是平均有效边移动次数，不冒充每条边的完整交换轨迹或混合证明。类型间诊断比较全部 cell-type 配对的边数，报告改变配对数、L1 变化及最大 20 项，缺类型保持空标签类别。

`input_output_paths` 提供逐输入群组到固定输出 body ID 的精确定向、无权最短路径；`with_paths=False` 的产物明确记录未执行。本批工具不记录神经放电，字段明确要求后续离线重放。

### P 边干预协议

本批采用协调者确认的边协议：预声明 `source_body_ids`，删除这些源的全部出边；是否包含自环须显式提供 `include_self`。全部节点继续保留，不称为删除/匹配节点。

配对控制只从非目标源的边中，按 **有符号整数权重** 分层、无放回抽取同数边。候选不足立即报错，不放宽匹配、不更换种子挑成绩。自环排除模式也对配对候选生效；包含模式不承诺自环数匹配。两图共享 pair/protocol/parent 身份并分别保存内容哈希；记录移除边数、突触数、符号×权重直方图、受影响源/目标数和自环数。

`intervention_mode=acute|retrained` 只登记未来运行意图，`fit_status=not_executed`。不承诺受影响节点数、解剖、路径或自环数量相匹配；节点级删除协议与实际急性/重训实验仍属后续工作。

## Public Interfaces

```python
create_research_control(parent_path, out_path, *, transform, seed, protocol,
                        max_attempts=None, target_swaps=None, with_paths=False)
# transform=R 时两个预算必须显式给出；W 不接受交换预算。

create_intervention_pair(parent_path, out_dir, *, source_body_ids, include_self,
                         seed, protocol, intervention_mode, with_paths=False)

load_research_control(path, parent_reference)
# parent_reference 为完整图路径或已加载的 Connectome；后者仍重新校验。
```

`python -m cnsbridge.learning.graph_controls` 要求 parent、out、transform、seed、protocol；R 另需两个交换预算；P 另需固定 source body IDs、自环策略与急性/重训意图。不会默认生成多个图或启动训练。

## Validation

- Risk Level：**S2**。新图变换、持久化来源与离线检查相互关联，修改限于独立新增模块；未扩大到全项目检查。
- `.venv/bin/python -m pytest tests/test_graph_controls_v2.py -q`：**11 passed，4.58 s**。覆盖 R/W 精确不变量、自环、确定性、零成功、不可交换/空边图、路径方向、母图不变、来源/群组篡改拒绝、生产拒绝研究图、不可覆盖、禁止研究图作为新母本、P 精确匹配及候选不足失败。
- 完整图工具实测一次，命令：

```bash
.venv/bin/python -m cnsbridge.learning.graph_controls \
  --parent data/malecns_v2/full_reference.npz \
  --out data/malecns_v2/controls/W_seed201_toolcheck.npz \
  --transform W --seed 201 \
  --protocol p3-complete-graph-toolcheck-not-neural-evidence \
  > data/malecns_v2/graph_controls_toolcheck.json
```

实测结果（来自运行产物，非预期数填充）：

| 项目 | 结果 |
|---|---|
| 母图 source ID | `873a58e99f5347f8bd5d3bee76e718bdccb340a67f070dc4f837f02262b41342` |
| 派生 source ID | `a5508faf4e730eb3b8b6a7031312e2f0735c10a479ac856415f06af5fb8876c1` |
| 派生 graph SHA256 | `d03930f39e60c1724c5957b15d3b1ca2a9cd161523ae48e1158fe37ba858f436` |
| 保留边 | 25,582,938 / 25,582,938；无新增/删除 |
| 固定自环 | 101 |
| 实际权重改变边 | 19,634,129（全部边的约 76.747%） |
| 可置换源 | 165,086；不可置换源 1,614 |
| 类型配对边数变化 | 0，与 W 拓扑不变一致 |
| 耗时 | 16.996145875 s，含母图加载、检查、诊断与保存 |
| 进程峰值 RSS | 1,749,647,360 bytes，getrusage 进程生命周期峰值 |
| 路径/神经/混合 | 路径未执行、神经未执行、混合未建立 |

在定向检查和一次完整图工具实测通过后停止本批验证，未运行全项目测试、完整 R/P 科研批量或重复已通过检查。

## Known Limitations / Remaining Questions

- 完整真实图实测本批选择 W；R/P 的算法与身份契约由小 fixture 验证，未声称完整真实图上的 R 交换链混合或 P 神经效应已测。
- 生成图不意味着已用于公平训练。P1/P2 后续集成须走显式离线研究入口，同图/输入种子单独拟合；生产接口继续拒绝。
- 用户尚无真实录像与动作标签，因此无动作准确率、测试集比较、通用解码或跨设备效果结论。
- 路径诊断默认不运行昂贵的全群组搜索，但函数及显式开关可用；未执行状态进入 provenance，不伪报结果。
- 本批未提交 Git；由协调者组织新的独立验收。
