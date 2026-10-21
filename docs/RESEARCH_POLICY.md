# 正式 R 研究策略工厂

> 命名迁移：本文操作入口已更新为 `malecns_lab`。涉及旧 run／checkpoint 的目录保留为参数示例和历史证据；不能直接用新源码续跑或重新登记旧实验。须使用匹配的源码身份与新输出目录，精确恢复旧实验需原 commit。见[迁移说明](ARCHITECTURE.md#环境迁移与历史来源)。

`learning.research_policy.from_terminal_research_run` 只用于离线拓扑对照。它构造精确现有 `NeuralPolicy` / `LaminaTonicLIF`，以真正的 R 整数图构建正向 CSR、全细胞伴随 CSR 和共享正向存储的 SciPy 缓存。它不训练、不生成参数档、不执行神经前向，也没有设备入口。

本批只有软件夹具证据；正式 R 仍须达到已登记终局。完整图的构造内存、前向及学习效果尚未实测，不得根据软件测试直接运行正式 R。实际构造须先完成新的受监督协议和 G/R 比较登记，参见 [P3e2](../specs/spec-008-trainable-snn/p3e2-research-factory.md)。

## 轻量父进程

```python
from malecns_lab.learning.research_admission import admit_research_graph

snapshot = admit_research_graph("data/<正式R生成run>")
```

这会重新调用 `generate_rewiring.accepted_candidate` 验证完整父批准链、源码/runtime、实际文件摘要和唯一终局，然后交叉核对候选来源、诊断与机械向量摘要、全部计数和生成身份。只读取 JSON、小 NPZ 元数据和数组头，流式计算文件摘要；不导入 Torch 或加载矩阵。

返回值是审计快照。修改字典并重算自身摘要不能获得准入；`revalidate_admission` 会重新读取原始证据。裸候选文件、生成探针、中间状态、W/P 图、失败父终态以及变化的源/文件均不能作为正式 R 入口。

## 工厂接口

以下说明接口，不是已批准的完整图运行命令。只能在另行登记的串行资源监督 worker 中使用：

```python
from malecns_lab.brain.trainable import TrainableLIFConfig
from malecns_lab.learning.neural_policy import NeuralPolicyConfig
from malecns_lab.learning.research_policy import from_terminal_research_run
from malecns_lab.senses.spatial_drive import SpatialDriveConfig
from malecns_lab.senses.target import TargetCamera

policy = from_terminal_research_run(
    generation_directory,
    config=NeuralPolicyConfig(frame_ms=50, precondition_ms=60, control_dt_s=.1, readout_seed=1703),
    camera=TargetCamera(),
    drive_config=SpatialDriveConfig(phase_seed=1702),
    lif_config=TrainableLIFConfig(),
)
```

背景固定为全部 L1–L5 的 8 mV，CPU float32。输入保持全部 3377 个 R1–R6，输出保持全部 1314 个 DN。原始计数、全部节点/边、自环、缺失注释和顺序保留；没有可达性或活动裁剪。

构造先用真实 G 初始化原类的非图参数；未创建任何神经状态。随后先释放 G 的 SciPy 视图、Torch 正向和伴随缓存及旧整数图，再加载并校验正式 R。新缓存逐边核对指针、索引、值和转置关系，权重遵循先转 float32 再乘 `weight_mv` 的原顺序。SciPy 的三个数组必须与正向 Torch 共享且只读。任何失败都不返回半安装的策略。工厂不接受现有 brain、policy、state 或延迟队列，不提供运行中换图接口。

完整母本和候选的加载/校验仍可能同时保留多个稀疏表示，实际峰值不能由上述释放顺序推断。继续使用 1 GiB 系统额外预留、申请 6 GiB、`min(6 GiB, available−1 GiB)` 有效上限、600 秒 / 0.1 秒监督；不与其他完整图任务并行，不调整 Noria/Docker。

## 身份与共同数值

研究 graph/model/map/spatial/drive/policy 都有与图内容对应的身份。`reference_source_id`、`reference_graph_sha256` 和 `reference_provenance` 仍指 G 母本；R 自身及生成准入放在 `research` 字段。`raw_ordered_counts_sha256` 来自真实 R 数组，不能沿用 G 摘要。

`research_contract.common_numeric_contract(policy)` 明确列出共同动力学、背景成员、类型分组、输入输出、空间分配、相位初始化、相机/时钟、特征/动作语义及初始 theta/读出数组摘要。它只接受初始参数，不是训练状态格式。`require_common_numeric_contract` 比较全部明确字段；不递归删除未知字段来制造一致。适配器来源单独绑定，原数值来源保持。共同契约相等不意味着 G/R 的连接或身份相等，更不意味着行为等价或学习收益。

软件小电路都明确为 `numerical_test_only`；元数据夹具中的完整尺寸只是格式字段，不是真实图。它们没有公共替代入口，不能通过正式生成准入。

## 现有边界

旧 `policy_checkpoint`、Adam/训练存档和 full_reference 加载器继续拒绝研究身份，不能用旧训练 CLI 直接训练此 policy。下一批须单独实现严格研究持久化/训练边界。本批未增加或开放任何设备接线。

现有 `NeuralController` 只检查 `NeuralPolicy` 类型，因此不是研究角色权限隔离器。返回精确该类型不等于全应用禁止任意内存对象接线；设备路径当前未使用本工厂，调用者必须保留离线用途。G/R 同维度的 `PolicyState` 和 `LIFState` 仍由各自真实 ID 相互拒绝。
