# Implementation Report — P3e2 正式 R 研究策略工厂

## Summary

已实现 Ready [P3e2 工厂规格](p3e2-research-factory.md)的软件边界：从正式生成 run 的唯一父批准终局取得轻量准入，构造精确现有 `LaminaTonicLIF` / `NeuralPolicy`，在首次神经状态/计算前安装真实研究图的整数计数、Torch 正向/伴随 CSR 及共享 SciPy 前向缓存，绑定真实研究身份与明确的共同初始数值契约。

没有运行完整 R 神经模型、正式 R 训练或研究参数/Adam存档。正式 R 尚未终局，软件夹具不代表生物图实测。本报告是实施结果，不是独立最终验收。

## Files Changed

仅新增：

- `src/cnsbridge/learning/research_admission.py`：父层只读正式终局准入、生成/母本/诊断/完整机械向量与工厂源身份、小 NPZ 元数据和数组头验证；不导入 Torch 或加载矩阵。返回快照不能代替重读原始证据。
- `src/cnsbridge/learning/research_cache.py`：私有首次构造缓存安装、完整分块结构/权重/转置/共享内存核对和完整输入输出/背景成员核对。
- `src/cnsbridge/learning/research_contract.py`：共同初始数值契约及全字段比较；验证 live 动力学/相机/感觉配置和时钟与 identity 一致，保持数值来源与适配来源分开。
- `src/cnsbridge/learning/research_policy.py`：唯一公共正式 run 工厂；真实 G 初始化后先释放旧缓存/图再加载 R，完整安装后重绑 graph/model/map/policy 身份，再次检查缓存/准入才返回。
- `tests/test_research_admission.py`、`tests/test_research_policy.py`：仅新增边界的元数据及 9 节点数值夹具。
- `docs/RESEARCH_POLICY.md`：接口、资源、身份与旧入口限制；示例不是已批准的完整图运行命令。
- 本实施报告。

工作区内协调者的交接、研究协议和验收文档不属于本实施修改。未修改 G29 或 R13 绑定源；未删除仍服务当前实验的旧模型/训练/生成模块。

## Important Decisions

1. 保留 `accepted_candidate` 原严格准入，不复制一个宽松父链验证器；新层交叉绑定最终候选的正式 R provenance、终局参数、母本、计数、诊断与工厂源。所有公共准入始终从 run 目录重新读取，不信任自洽 JSON。
2. 工厂只接收 run 和明确配置，不接收已有 brain、policy、state 或延迟队列。先用真实 G 创建精确原类，不执行 reset/step/预处理；在构造内部完整替换实际 R 缓存，不临时改变 R 的角色、不继承伪装、不 monkeypatch 原数值方法。
3. 正向和伴随保留原操作顺序：整数计数先转 float32，再乘 `weight_mv`。逐边检查全部 CSC/CSR 指针、索引与值、SciPy 指针/shape/dtype 和只读共享视图；数值夹具另外实际执行原 forward/全细胞 adjoint 和神经 step，不能只靠 ID 不同证明换图。
4. `reference_*` 仍指 G 母本，实际 R 与准入放入 `research`；真实 R 的 ordered-counts 摘要单独计算。全部 L1–L5 固定 rho=0/8mV、完整元数据/缺失/节点顺序及输入/DN保持。
5. 共同契约只用于初始未训练 policy，明确列字段和初始数组，不递归删除图字段制造相同。工厂源另绑；核心配置/runtime/源、感觉和时钟、读出数值全部进入共同契约。
6. 小元数据先验证 NPY 头、shape、dtype 和声明字节数，再读固定上限的实际 payload；ZIP 文件长度本身不能防止伪造头触发大分配。
7. 旧参数/Adam/训练存档不放行研究身份；本批不实现 P3e3。文档明确 `NeuralController` 只是类型检查，并非研究角色隔离器，没有新增设备接线。

## Validation

- Risk Level：**S3**，限定在新增终局准入、完整缓存和身份边界。协调者在每次检查前确认完整图作业已终态，并明确给出独占数值时段；未与完整图作业重叠。
- 首次命令：

  ```text
  PYTHONPATH=src .venv-train/bin/python -m pytest -q tests/test_research_admission.py tests/test_research_policy.py
  ```

  session `32134` 已收取 exit 1；**30 passed / 1 failed，1.08s**。失败为负例本身尝试把已只读的 SciPy 视图改为可写，NumPy 已立即拒绝，未到达被测校验器。修正该项夹具：使用同一 Torch storage 的新可写 NumPy view，以便实际测试新校验器的只读约束；业务实现没有因该失败修改。
- 仅重跑该失败项：

  ```text
  PYTHONPATH=src .venv-train/bin/python -m pytest -q 'tests/test_research_policy.py::test_complete_cache_checker_rejects_wrong_storage_and_graph[writable]'
  ```

  session `39949` 已收取 exit 0；**1 passed，0.79s**。其余已通过 30 项未重复。
- 静态复核发现并局部收紧两处具体边界后追加最小相关检查：live config 与 identity 对应，以及小 NPZ 伪造头分配量。新增 4 个 live 配置反例、2 个元数据头反例，并复核受实现改变直接影响的 4 个原契约/合法准入用例：

  ```text
  PYTHONPATH=src .venv-train/bin/python -m pytest -q tests/test_research_policy.py -k 'common_contract or same_graph_install or different_graph_actual' tests/test_research_admission.py -k 'common_contract or same_graph_install or different_graph_actual or snapshot_rechecks or declared_oversized'
  ```

  session `37802` 已收取 exit 0；**10 passed / 27 deselected，0.78s**。该命令最后一个 `-k` 表达式决定筛选；未运行无关已过项目。
- **累计 37 个不同检查通过**，不是一次完整新套件重跑。检查包括：合法元数据交叉绑定；W/P/探针/非终局/源/图/选择/父失败/文件替换/伪造快照拒绝；全新 OS 进程禁止 Torch/模型导入；全部缓存结构/权重及共享存储；旧正向、旧伴随、旧 SciPy、符号、转置、dtype、指针、可写视图反例；实际 forward 与独立矩阵参考、包括静默细胞的 all-cell adjoint；相同图与不同图的数值/身份区分；G 原缓存和图对象在 R loader 前释放；body/group/input/tonic/rho 错配；跨图神经/策略状态拒绝；无运行中换图入口；旧参数/生产 map 的研究角色拒绝；共同初值及配置变更拒绝。
- 数值检查只有已有 PyTorch 稀疏 CSR beta 提示，未观察到数值失败或非有限结果。
- 使用纯标准库读取实际 `data/neural_training_joint_20261019_r1/run.json` 与 `data/topology_generation_main_20261019_r1/run.json` 的 `implementation_sha256`，逐文件计算当前 SHA：**29/29 与 13/13 全部不变**。该核对未导入 Torch 或加载图。
- 仅检查本批新增文件/差异和直接调用链；未运行旧 37 项生成器检查、全项目测试、build/lint/typecheck、完整图模型、设备或付费作业。没有操作 Noria/Docker；没有 commit。

## Known Limitations

- 准入合法路径使用明确标注的元数据夹具替换原父链门槛以测试新交叉绑定；这些夹具的声明尺寸和 header-only payload 不是完整生物图。原严格门槛的既有验证不被该夹具替代，实际正式 R 终局/工厂门槛仍未执行。
- 数值夹具保持 `numerical_test_only`，通过私有安装路径验证真实缓存逻辑；公共工厂不提供数值图入口。当前证据不证明完整 R 的构造峰值、前向成本或学习效果。
- 实际构造仍须重新加载完整母本进行机械校验，可能有临时多个稀疏表示；未根据 G 训练内存推断 R 构造会满足预算。
- 新工厂复用原私有构造接口及缓存结构，其源身份已绑定；不是向外承诺这些内部接口稳定。公共输入不会接受半成品对象，但 Python 本身不提供进程内任意对象修改的权限隔离。
- 没有研究参数/Adam/完整状态持久化、研究训练/评估 runner、硬件接线，未关闭 Spec008 AC3–AC6。

## Remaining Questions

无阻止本软件边界交付的已知问题。等待新的独立软件验收；正式 R 终局、实际受监督工厂门槛和后续研究持久化由协调者按各自协议推进。
