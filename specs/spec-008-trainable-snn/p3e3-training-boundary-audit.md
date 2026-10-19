# P3e3 — 研究训练持久化与窗口边界审计

状态：**Draft / 只读设计审计，未 Ready，未实施，不是验收或运行结果。** 2026-10-19。本稿供后续 Spec/Plan 选用；遵循已 Ready 的 [G/R 科学协议](p3e3-topology-study-protocol.md)，不新增 R 训练许可，不改变现有 G29/R13 绑定来源。

本次只读源码、规格和 G 历史第一窗 JSON/ZIP 成员表；未导入 Torch/NumPy、未加载图或数组、未运行测试、未修改业务代码。开始时工作区无未提交修改。审计期间 G 可由协调者独占运行；不得据本文启动并行数值作业。

## 结论

可完整保留原 LIF、代理梯度、稀疏正向/伴随、感觉驱动及 `NeuralPolicy` 前向；原 NumPy 损失、裁剪和一般步 Adam 独立复算也可直接复用。R 真正需要新增的是：严格研究身份验证、参数/Adam/完整状态格式，以及使用这些边界的短两帧更新编排。

不能直接复用现有 `training_window.compute_window`：它在窗口开始及结束调用 `optimizer_snapshot`，继而调用 `training_optimizer._policy_parameters` → `policy_checkpoint._arrays` → `_validate_identity`。后者同时拒绝研究扩展字段和 `research_control` 角色。`production=False` 只影响旧 runner/evidence 的部分历史检查，既不能解决该限制，也不是合法研究准入。

在“不能修改冻结源码、不能 monkeypatch/冒充角色”的约束下，原 `compute_window` 没有可注入的 optimizer/persistence 回调。因此不存在零编排重复的合法复用办法。可接受的最小差异是一份约百行、逐步骤对齐的研究两帧编排；它调用原数值核心及独立复算，不复制整个注册、扫描、监督、训练循环或评估系统。若连这份必要编排也禁止，则当前冻结约束下无法实施 R 更新，不能假装用包装函数解决。

## 精确复用和阻断位置

以下路径均位于 `src/cnsbridge/learning/`，除明确标出的 brain/senses 实现。

| 位置 | 可直接复用的边界 / 不能直接复用的原因 |
| --- | --- |
| `neural_policy.py:203–257` | `_validate_parameters`、`_drive_for`、`initial_state`、`forward_tensor` 不硬编码 full_reference；真实 R 工厂已返回精确原类型。保持每场景一次 initial_state、每帧 500 步、初始 600 步和全部 2628 特征。 |
| `research_policy.py:109–125` | 唯一公开 R 构造 `from_terminal_research_run` 可直接复用；它重验终局并安装真实缓存。禁止新增裸图/半成品 brain 的公开入口。 |
| `research_policy.py:94–105` | 原工厂已经把 `research_factory` 放入 model/policy，并把 `common_numeric_contract` 放入 policy；后续训练源应放入独立训练绑定，不改已有 policy/model ID 或补写 `_identity`。 |
| `research_contract.py:26–80` | `common_numeric_contract` **只允许未训练初值**，要求 theta=0 和 seed 生成的原始读出。首次构造时计算并封存；恢复前可在新建初始 policy 上重算，随后再加载参数。不能在训练后的 policy 上调用它验证续训。 |
| `research_contract.py:83–89` | `require_common_numeric_contract` 比较明确契约，允许原样使用；两个自报字典相等仍不能替代对真实构造/终局证据的核验。 |
| `policy_checkpoint.py:127–195` | `_validate_identity` 首先限制字段集合（134），随后限制 role（137）；不是只加一个 role 就能兼容。`_validate_model_semantics` 也限制 model 字段（83–86）。整个旧参数 reader/writer/restore 不可直接用于 R。 |
| `policy_checkpoint.py:31–40,212–222` | JSON、规范摘要、参数数组/声明是角色无关的小工具；但整个模块导入 Torch，不能为了三个小 helper 让轻量父进程导入它。父端已有 `training_contract` 的 JSON/数组/hash helper。 |
| `training_optimizer.py:18–26,73–93` | `optimizer_contract` 和 `validate_moments` 可原样复用；后者只检查命名字段、shape/dtype、正 step 和非负二阶矩，不检查图 role。Torch-free 父端可用 `training_contract.optimizer_contract` 得到同一配置。 |
| `training_optimizer.py:29–70,96–110` | `_policy_parameters`、make/validate/snapshot optimizer 都经旧参数身份限制，需新的严格绑定适配；仍使用精确 `torch.optim.Adam`、原 `ADAM_OPTIONS` 和参数对象身份/顺序检查，不引入新优化算法。 |
| `training_checkpoint.py:72–177` | `_validate_context`、`_state_metadata`、`_state_arrays`、`_validate_state`、`_build_state`、`_declarations`、`_no_gradients` 均可在 worker 原样调用，实际 identity 保持 R。它们已经覆盖全部延迟队列、感觉相位/计数、dtype、时钟及 detach 条件。 |
| `training_checkpoint.py:180–301` | save/read/_restore/load 经旧参数格式、旧 Adam 校验和 full_reference 构造，不能直接使用。新格式须独立 manifest/schema/purpose，不把研究档包装成旧部署档。 |
| `training_archive.py:24–54` | `checkpoint_state_digest`、`validate_state_digest` 是 Torch-free 且角色无关，可直接复用；摘要检查本身不代替实际数组的有限性、范围与完整清单检查。 |
| `training_archive.py:57–119` | `inspect_checkpoint` 绑定旧 schema/purpose、旧三个存档源码、旧 nested 参数格式；即使没有单行 role 判断，也不能拿它接收新研究格式。 |
| `joint_probe_worker.py:27–52` | `snapshot`、`raw_counts_digest`、`state_digest` 不限制 role，可在 worker 原样调用。它们是数值快照工具，不承担研究身份准入。 |
| `training_math.py` | `clipping`、`loss_metrics`、`adam_moments`、`adam_parameters`、`moment_check`、`validate_window_numbers` 均可原样用于严格核验后的真实 R identity。它们不证明数据/图来源，由新外层负责。 |
| `training_math.compare_r2` | 仅对初值/输入、DN/动作、原始梯度和更新值作历史比较；**没有覆盖裁剪梯度、Adam 矩和完整末态**。不能单独充当 P3e3 的 G 新边界等价门槛。 |
| `training_contract.py:98–121,180–209` | 原 optimizer contract、active_keys、固定 dataset 核验、cursor、next_selection、prefix_line、bytes_sha 可直接复用；新研究契约只选 joint。register/validate_run/history 绑定 G policy、原历史和来源，不可泛化成 R 注册。 |
| `training_evidence.py:34–55` | `sampled_boundary` 与图无关，可直接用于同一 1GiB/6GiB/600s/.1s 资源合同。 |
| `training_evidence.py:17–31,58–209` | segment/resume/window/scan/completion 硬编码旧 schema、参数档、r2 预处理摘要或 G 原始 counts。未来研究 runner 需新的严格薄编排；不得 `production=False` 跳过首窗来源后冒称父验收。 |
| `training_worker.py:39–138` | execute_windows 绑定旧 compute/save；run_worker 又固定 full_reference。不得复制该文件整体或复用它时篡改全局 imports。 |

## 建议的一次完整软件切片

建议先实现 **P3e3a：严格研究参数、Adam、完整状态与真实两帧更新 API**，暂不实现完整 160 窗 CLI/链扫描或 R 闭环评估。单一实施者负责所有耦合边界。该切片必须真的能执行 loss.backward/Adam.step、保存与新进程恢复后再更新；只有新的 role checker 或转发 wrapper 不算完成。

可按五个职责模块组织，名称由最终 Plan 冻结；以下是具体可实现的接口边界而非既有 API：

1. `research_training_contract.py`：Torch-free 的严格用途/来源/identity/context 验证。以真实终局 admission、科学协议、已验证 G 初值契约及源清单构造绑定；调用方字典必须重读实际证据核对。显式允许 `terminal_R_joint` 和 `G_boundary_equivalence` 两个不同用途。数值小电路只走私有测试辅助，不是第三个生产用途。
2. `research_training_archive.py`：Torch-free 参数与完整训练档读取/校验。提供独立版本的参数 manifest 与训练 manifest，固定文件名、未知/重复字段拒绝、allow_pickle=False、文件/数组摘要、完整嵌套 policy、实际 role、绑定 ID、源码/runtime/线程和真实数组校验。可同模块封存已验证快照，避免再建独立重复父 reader。
3. `research_training_optimizer.py`：worker 的 `make_optimizer`、`validate_optimizer`、`snapshot_optimizer`；复用原 joint 配置与 validate_moments，替换旧 full-reference-only 参数准入。拒绝错误对象/重复绑定/额外参数/错参数组及任何 frozen/acute 冒充。
4. `research_training_checkpoint.py`：worker 的 `save_checkpoint`、`restore_checkpoint`、`load_checkpoint`，先使用新 archive 验证，再复用旧角色无关 state codec。所有参数与矩/状态张量准备好后才一次性修改对象；任何晚期失败均不得留下部分恢复。生产 R load 重新走唯一终局工厂；G 诊断 load 走原 G 工厂并保持 G 身份。不得隐式 initial_state/重放前缀。
5. `research_training_window.py`：新 `compute_window` 使用相同两帧顺序、前向前构造正则、原 forward_tensor、同 mask/scales、loss.backward、原 clipping、原 Adam、detach/zero_grad，再调用原 validate_window_numbers。只替换严格参数/Adam/存档边界和证据用途；阶段顺序及所有浮点运算次序维持原样。

新参数档可用 `cnsbridge.research_training_parameters.v1`；完整档可用 `cnsbridge.research_training_checkpoint.v1`。purpose 必须区分正式 R 和 G 等价诊断，加载时与外部 expected binding 精确相符。G 诊断档不能作为 R 初始化/续训档，R 档不能被旧部署 reader 接受。原四个 context ID 仍可交给旧 `_validate_context` 校验；研究 study/admission/common-contract/purpose/source 等额外身份放在其外层不可分割绑定中，不为了兼容向旧函数塞未知 context 字段。

这五模块只解决可训练状态和窗口运算。后续 runner 必须再实现严格段注册、event→prefix→checkpoint→window seal、独立父扫描、源和资源失败分类；当前切片不能冒称已经有可执行的正式 R 连续训练 CLI。参数-only reader 将来也可供独立研究评估复用，但不能直接送进仍只接受 G 的 P3d loader。

## G 等价用途必须真实、且走新路径

G 诊断从已验证的 G run/图/协议和历史第一窗构造精确原 `NeuralPolicy.from_reference`。其 policy、model、graph、map、state 始终保持真实 full_reference；不调用 R 工厂伪造 admission、不删字段重算假研究身份。用途是一次边界兼容诊断，不计新的 G 科学训练候选。

诊断使用**同一新增** optimizer/window/checkpoint 边界。若仍调用旧 compute_window 得到 G 对照、而 R 用另一份编排，则没有验证新增编排。G 单窗模式不开放任意训练预算；另立受监督实际协议后仅计算同初始 combined 0/1。诊断之后的新数值/状态/参数适配源若改变，原等价通过不能自动覆盖新版本。

实际可用的历史锚点为 `data/neural_training_joint_20261019_r1/segments/000001/windows/000001/`。本次只读取其既有声明和 ZIP 字段，不声称重新独立验证全部字节：

- G run ID：`ccb4a20b84269f9ec7f5f49233e3dc202729f61434ba7d1ad0772ede5b2273f0`。
- window ID：`491c6d0b6b841610121c7889a3a5761b047c06c542fdc4f244dff67ea2f9895e`。
- evidence.npz SHA 声明：`c618ec6dbca47b3bdd1b510a4629f47507ae09db992d81bc8e43543aeb17566a`。
- checkpoint ID：`1387b7b3da06366fc0b0530bd6de3b0228db99684cdd56580a989038e75dc375`。
- checkpoint/training.npz SHA 声明：`5e5baef7a6c6366c590ee2a7b4024f392b33190885660f8dcbd842da66702644`。

`evidence.npz` 已有全部 before/after 参数、gains/rho、输入、DN、动作、pre_clip/post_clip 三参数及全部 adam_after 命名状态。`checkpoint/training.npz` 已保存 v/g/refractory_until、18×166700 的全部 pending、3377 phase/events 和全部 Adam。首窗 end frame=2、brain/drive step=1600。它比只引用 P3b r2 的 compare_r2 覆盖更完整。

实际协议应逐项冻结上述真实文件及父 seal/登记/终态/hash 链，再比较：

- 输入/初值/shape/dtype/离散时钟与整数数组 exact；G identity 也应真实相同。
- 全部 DN、动作、原始/裁剪梯度、更新数值及已存在末态浮点数组按已 Ready 科学协议 `rtol=1e-6, atol=1e-7`，同时报告 exact/maxdiff；不能只比较摘要后声称容差通过。
- Adam 必须另作原 `training_math` 一般步独立复算，保留 moment rtol=1e-6/最小 float32 次正规数 atol，以及参数 rtol=5e-6/atol=1e-9。历史 G vs 新 G 的成对比较与这些内部算术检查分别记录，不相互代替。
- 起始预处理历史若只有完整状态摘要，明确按摘要 exact 检查，不能宣称拥有未保存的逐数组误差值。末态已有实际数组，必须全量逐数组比较。

G 实际 gate 无需等待 R 终局，但必须在本切片软件验收、最终来源封存、独立实际协议审查和取得串行完整图作业时段后运行。本稿未运行它。正式 R 首窗仍必须等待真实最终 candidate、另登记构造 gate、G 实际等价通过及研究 runner 完整准入。

## 绑定和拒绝要求

- 明确区分四类源：原数值/训练 G29、生成 R13、工厂四文件、新研究适配文件。实际使用的 frozen helper 亦须列入绑定。以运行原始 manifest 的逐文件 SHA 为权威；“实验已经结束”不授权编辑这些冻结文件。
- R policy 全部嵌套 role 保持 research_control，`reference_*` 仍指 G，`research` 指真实 R。验证完整字段集合、所有 canonical IDs、factory/admission/common-contract 关系、机械规模、真实 nullable 注释/选择/全部 L1–L5 固定背景；不做递归删除 graph/source 字段的宽松比较。
- 原 `common_numeric_contract` 只封存初值，学习后的参数值另由参数 checkpoint ID/实际数组摘要绑定。不能用相同 policy ID 证明相同参数，不能把当前 trained theta 误与原始 zero 检查混在一起。
- R/G 续训都保存完整原始 state dtype/shape/values；scene 内只 detach，scene 末保存旧 scene 的最后状态，下一窗口才 reset。参数/Adam 跨 scene 保留。真实全无效首窗仍拒绝零 step 存档，不伪造 step=1。
- 新正式 R 只支持 joint；独立 G frozen_brain 仍使用自己的从初值训练的原 run。acute_restore 是保留 joint 最终读出并恢复初始 theta 的评估干预，不是训练模式，不得让三者在 loader 或档名上混淆。
- 旧 reader/full_reference loader 必须继续拒绝 R，新 reader 必须拒绝旧部署档/错误用途/错误 admission/未知字段和同尺寸异图 state。`NeuralController` 仍只是类型检查；本批没有新增硬件路径，也不能宣称所有内存控制器都具备角色隔离。

## 与切片相称的直接检查

风险 S3，限新持久化及更新边界。无需重跑旧 37 项工厂或整个项目：

1. 明确 numerical_test_only 私有夹具：原 window 与新增 window 使用相同初值/实际旧矩，至少两次有效更新逐值比较；覆盖非零延迟 pending、感觉 phase、连续时钟，第二窗验证真实旧 Adam 传递。只对新增边界对照，不冒充正式 R。
2. 真实两个 OS 进程保存/恢复后继续，和不中断路径比较全部参数、Adam、动作/DN、完整 state、cursor；验证 scene 切换恰好一次、场景末旧态不被错误标成新场景。
3. 验证注册的三参数对象绑定/配置，错组、重复对象、额外参数、step/矩 shape/dtype/非有限值拒绝；保留零二阶矩与非零梯度下溢语义。原数学已验收，只检查适配层是否正确调用并传入实际旧矩，不重跑其全部旧算术实验。
4. 坏 role/purpose/admission/common contract、缺/重复/未知字段、坏父/参数/源码/runtime/文件 SHA、同形异图 state、错 phase/pending/clock、未封存尾部拒绝。安排一个所有早期检查已通过后的晚期错误，证明目标 policy/optimizer 完全未变。
5. 轻量父模块 subprocess import 后无 Torch；新旧档/loader 双向隔离。G 诊断保持真实 G identity，正式 R 准入仍调用真实终局检查，不能测试里 monkeypatch 后把结果当正式许可。
6. 实施者仅核对一次实际 G29/R13 源字节保持和最终 diff；通过后停止。独立验收阅读上述证据，不无理由重跑。

软件检查之后的 G 全图 gate、R 首窗/第二新 worker gate 是不同的预登记实际实验；不能用小电路软件 PASS 代替，也不在本审计中运行。完整配对训练、闭环成绩、拓扑收益和 AC3–AC6 仍未由本稿完成。
