# Implementation Report — P3 离线研究接入与实验登记

## Summary

完成显式研究图重放、协议模板/登记/只读验证/执行链路；复用已经完成的 P1 重放存储与 P2 掩码学习、独立拟合和配对评估接口。实际完成一次完整 W 图的短接通，未执行真实录像训练或科学效果比较。

用户已确认没有录像，先完成诊断和工具。本次没有制造动作数据、容差或科研结论，没有硬件、MiniFly、阈值 3、B/C 参数训练或 commit。

## Files Changed

- `src/cnsbridge/learning/research_replay_v2.py`：专用 `ResearchNeuralReplayV2`、共享 P1 输出的 `record_replay` 与显式 CLI。
- `src/cnsbridge/learning/experiment_protocol_v2.py`：未填值模板、完整素材/图/代码身份冻结、split/种子/预算/图角色检查。
- `src/cnsbridge/learning/experiment_v2.py`：register/validate/execute 入口、逐图×种子提取与独立拟合、P 急性专用读出应用、失败与覆盖记录。
- `tests/test_experiment_v2.py`：研究网络替换、冻结/泄漏/预算约束、逐图种子独立 fit、P 专用路径和失败不丢失的合同测试。
- `docs/EXPERIMENT_V2.md`：使用方式、协议条目、证据与限制。
- 本报告。

未修改已 PASS 的图变换模块、P1 模块或 P2 模块；P2 CI 三态解释由其独立返工负责，本批只调用最终 API。

## Important Decisions

### 真正替换网络，保持共同计算

研究构造器先按 P1 加载完整真实母图，再调用 `load_research_control(control, parent)` 验证节点、群组、角色与变换身份。随后释放母图 LIF 对象，**以派生 `control.weights` 创建新的 LIFNetwork**，恢复相同偏置、群组、输入映射与只读输出探针并 reset 全部状态；不是只改 metadata 或图名称。

新的模型身份含派生 graph/source、母图 source、变换/图种子/参数与研究适配器源码。`computation_identity` 去除 graph/source/边数和专用研究字段后，保留全部共同编码、事件协议、LIF、偏置、群组、特征、时间及代码身份；不同图运行必须一致。

不增加生产 Brain 的通用允许研究开关。CLI 必须同时给 `--parent`、`--control`，与源材料、输入种子、预算和新输出目录。实际记录调用原 P1 `run_segment`、`source_coverage` 和写入函数，格式继续为 `cnsbridge.neural_replay_run.v2` / `cnsbridge.neural_activity.v2`，可直接由 P2 读取；没有动作标签或设备实例化。

### 注册冻结与材料隔离

模板中的容差、δ、预算、种子和材料留空，不能直接执行。用户提供真实视频与独立标签后，登记完整图/control 身份、图种子、共同输入种子列表、素材与标签哈希、场景/会话/素材族分区、段边界、任务语义、SI 误差尺度、δ、bootstrap 和固定九个岭候选。相关代码哈希一并冻结。

恰有一个 G；研究角色必须与派生 provenance 一致；同一图身份不得重复计为独立条件。`visual/mean/zero` 是保留基线名称，拒绝同名图 ID，防止比较项指向歧义。P 的 target 与 matched 图须成对登记。跨 split 的场景/会话/族/源 ID/视频内容重复被拒绝。

登记期间读取文件哈希、材料 metadata 与标签结构，未计算动作分数；不假称登记就证明数据此前未见。pilot 的测试材料只冻结身份，不提取或评分；confirmatory 执行必须有登记，拟合/参数选择依然只用 train/val。

### 独立拟合与配对

执行顺序为 G 后其他图，每图×输入 seed 独立重放。先严格检查 P2 行身份、监督掩码、有效性、目标和全部共享视觉特征一致，再调用 `for_input_seed → fit_readouts → save_fit`；G/R/W 不能复用别图的拟合参数。V/M/Z 使用相同有效评分清单与训练/验证预算。

急性 P 是唯一明确的固定读出应用：要求登记角色 P_target/P_matched 和 `intervention_mode=acute`，应用本次登记 G 的同输入种子读出，严格保留输出顺序与时间、相同评分行，报告训练与评估两个来源；不保存冒称在 P 上拟合的新模型。R/W 无法进入该路径。retrained P 走独立完整九候选 fit。

按每图报告跨输入 seed 的场景指标，并保留所有图的单独结果与图种子差异。配对区间复用 P2 的三态结果，但本工具顶层仍标为 pilot 探索或登记描述性比较，不自动宣称确认性优势：尚未实现多重比较校正或随机图排名检验，也不以单张 W 测试代替随机图研究。

### 失败与预算

保存完整执行计划再工作。任何重放失败、预算耗尽或零有效覆盖不删图；发生至少一项重放未完成时，不对成功子集拟合。不同图有效行不一致直接终止并保存失败清单，不取行交集。独立 fit 失败时保留各条件状态，最终不产出部分成功条件的“优势比较”。

原 P1 产物保留失败帧、未处理尾段与全场景覆盖；顶层计划保存全部注册任务及未执行原因。`run_summary.json: incomplete` 或 `run_failed.json` 不能作为完整实验结果。

## Validation

- **Risk Level：S3（具体依据）**：接入新的真实研究网络传播路径、跨 P1/P2 的来源/持久化协议，以及多图训练隔离与预算/失败契约；检查仅覆盖新增集成模块，不运行全项目测试。
- `.venv/bin/python -m pytest tests/test_experiment_v2.py -q`：**7 passed，0.23 s**。fixture 只验证算法与编排合同，不是替代脑或经验动作数据。
- 检查中发现图 ID 与保留基线名称可能歧义，局部加拒绝规则后只运行对应检查：`.venv/bin/python -m pytest tests/test_experiment_v2.py::test_registration_rejects_cross_split_family_and_changed_search_budget -q`：**1 passed，0.11 s**。未重复整组。
- 小 fixture 检查真实 LIF 对象的 `W` 与 `_data` 确实换成 control 权重、共同计算身份不变、偏置与 reset 生效；专用 loader 在测试中被隔离，图 loader 本身已有独立验收，本批不重复验证它。
- 独立 fit 测试确认 G/W 的两个输入种子分别调用拟合，共 4 次；急性 P 两次不调用 fit，保存原 G 训练来源，且拒绝 W 和不同特征顺序。额外测试确认一项重放失败保留全部任务且不拟合成功子集，评分覆盖不一致在 fit 前失败。

### 一次完整 W 图的实际接通

P1 独立 PASS 后执行一次：

```bash
.venv/bin/python -m cnsbridge.learning.research_replay_v2 \
  --parent data/malecns_v2/full_reference.npz \
  --control data/malecns_v2/controls/W_seed201_toolcheck.npz \
  --out data/malecns_v2/research_replay_W201_static \
  --stimulus static --run-seed 0 \
  --observation-seconds 0.1 --budget-seconds 180
```

运行产物实际记录：

| 项目 | 结果 |
|---|---|
| 状态 | complete |
| 模型 source ID | `07f7ede8f761e2d00d58eaa251879a0e8cb85e80e41a05861341e27d59c68a6d` |
| 共同计算 ID | `ea1b4fcecce4d97360481148fd61461b75e715ea0bcc8a921e9fa25a2d2af790` |
| 图 source ID | `a5508faf4e730eb3b8b6a7031312e2f0735c10a479ac856415f06af5fb8876c1` |
| 图 SHA256 | `d03930f39e60c1724c5957b15d3b1ca2a9cd161523ae48e1158fe37ba858f436` |
| 完整图节点/边 | 166,700 / 25,582,938 |
| 输入有效帧 | 42 / 42；失败 0；未处理尾段 0 |
| 预热/可预测帧 | 40 / 1；最后一帧无片段内未来目标 |
| 模拟时间/步数 | 2,100 ms / 4,200 |
| 放电次数 | 9,274,281 |
| 曾放电节点 | 43,438；没有把其余节点裁掉 |
| 累积突触事件 | 2,701,548,759 |
| 墙钟时间 | 27.2049405 s，含母图/派生图加载与检查 |
| 进程峰值 RSS | 1,762,639,872 bytes |

实测只是完整 W 实际进入 LIF 的工具接通；没有动作标签、G/W 同任务评分、训练或统计比较。本批没有重复此完整图实验，也没有新增 R 混合研究、全刺激重跑或硬件运行。

## Commands / Outputs

```text
python -m cnsbridge.learning.research_replay_v2 --parent ... --control ... --out ... --stimulus static --run-seed 0 --observation-seconds 0.1 --budget-seconds ...
python -m cnsbridge.learning.experiment_v2 template --out ...
python -m cnsbridge.learning.experiment_v2 register --protocol ... --out ...
python -m cnsbridge.learning.experiment_v2 validate --registration ...
python -m cnsbridge.learning.experiment_v2 execute --registration ... --out ...
```

完整说明见 `docs/EXPERIMENT_V2.md`；真实接通证据在被忽略的 `data/malecns_v2/research_replay_W201_static/`。没有实际动作协议或数据时，登记/执行工具不宣称已运行真实动作研究。

## Known Limitations / Remaining Questions

- 还没有用户录像、动作标签、任务容差及足够独立场景；不能执行真实动作拟合，也不能勾选 M4 研究证据。
- 实验层保留每图描述性结果和条件于已登记种子的区间，未自动实现随机图集合充分混合、多重比较校正或确认性随机化检验；不声称已经验证通用解码器或果蝇认知。
- 静默、不可达或低分图不能按结果剔除；数值/覆盖失败以未完成实验保留，后续需根据具体问题另立修复/数据协议。
- 只读 validate 检查冻结文件哈希及登记身份；不能证明外部研究人员此前从未接触测试数据。
- 无协调侧阻塞，实施完成后停止验证，交新的独立验收 Agent 审查。
