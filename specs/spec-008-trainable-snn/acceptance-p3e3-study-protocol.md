# Acceptance — P3e3 完整 G/R 配对研究协议

## Result

**PASS** — 2026-10-19，新的独立验收 Agent `topology_study_protocol_accept`。只验收 [P3e3 科学契约](p3e3-topology-study-protocol.md)的可实施性、既有事实一致性和比较边界，不验收 P3e2 待完成的软件，不声称已生成最终 R、训练 R 或完成闭环比较。

本登记明确 G 已开始训练且其前进饱和已知，没有把本方案冒称为在全部 G 结果之前预注册。冻结点在首次正式 R 神经结果与主配对比较之前；未来档案、runner、首次完整图构造和数值等价运行仍须各自满足登记与验收门槛。

## Spec Coverage

- **完整母本与正式 R 唯一性：满足。** 与实际 `data/topology_generation_main_20261019_r1/run.json` 的固定规模、seed 1901、25,582,837 次成功交换上限、102,331,348 次提案上限和先到即止规则一致。中间档、成本探针或换种子不能替代最终父批准 candidate；保留不足交换、拒绝和失败，零成功不算有效反事实。控制逐源权重、度数/符号与自环，同时公开入强度、路径和类型对变化，不把该扰动效果强行解释为某条通路的作用。
- **相同学习机会：满足。** 实际 G 登记与 `training_contract.PROTOCOL/SCENES` 支持四个 train 场景的 combined/forward/vertical/yaw 顺序、各 80 帧、160 个两帧窗、唯一最终档、同一 dataset ID 和 manifest SHA。参数集合、学习率、Adam、独立稳定裁剪、损失、状态延续、场景重置、无效帧与失败规则均承接 P3c2，没有为 R 增加训练或选择机会。R 从原始参数、感觉配置和种子开始，禁止复制训练后的 G 状态。
- **新研究边界的实际数值等价：满足。** 不是只比较源身份或类型名称；正式 R 首窗之前必须让新边界以 G 的历史首窗输入/初值真实运行，比较 DN、动作、原始/裁剪梯度、更新、Adam 和已有末态。初值/输入/离散项 exact，数值容差沿用既有标准，超差不能解释后放行。历史字段和 SHA 留待实际协议逐项绑定是明确的后续门槛，不是本协议欠缺的软件成果。
- **跨进程与终档准入：满足。** R 的首两窗是唯一正式训练前缀且分别在新 worker 中执行；恢复前检查实际参数、全部优化/神经/感觉状态及游标，再推进更新。实际独立验收后才开放最多四窗段。主比较只用两侧第 160 窗的最终参数，每个评估场景重置状态，不复制训练末隐藏状态。
- **场景、主指标与 test 门槛：满足。** 四 val、六 test 的 ID/顺序逐项符合 `scene_registry()` 和 `evaluation_contract.VAL/TEST`。唯一主指标固定为六 test 合格率等权均值及 G 减 R 差值，每场景分母固定 80；真实物理终止与资源/数值/证据不完整明确区分。缺失不填零、不以少数完成场景冒充完整主均值。R 的 test 和配对汇总须等待 G 五条件全部 val 与 R 四 val 封存；同时保留既有 P3d 的独立执行顺序和真实时间，不回写旧协议。
- **原评分语义与监督：满足。** 任务沿用 `follow_task._errors/_metrics` 与 `evaluation_rollout.step_record`。其中“有效记录”的 MAE 范围按既有实现解释：水平/垂直是目标位于相机前方、对应几何误差非 null 的实际记录；距离是全部实际记录的欧氏距离误差。两者都不是按 `observation.valid` 过滤的检测误差。检测有效才可能 qualified；覆盖、有效观察、实际记录数量、失败与 MAE 并列。相机/阈值/80 帧计划和丢目标后置零监督一致，不能以监督行为充当神经能力。
- **对照和结论范围：满足。** G 的独立 frozen_brain、acute_restore、direct_visual、zero 保留各自不同问题；G/R 主比较仅使用两个 joint 最终档，不把 acute 当作独立冻结训练，也不声称没有做过的 R 冻结组或训练×拓扑交互。单种子、单扰动、不保证混合、不推断随机图总体或显著性、离线分数不证明部署实时性/实飞等边界明确。
- **资源与来源：满足。** 保持 1 GiB 额外预留、6 GiB 申请及有效预算、600 秒/0.1 秒监督、全图任务串行和 Noria/Docker 保留；G 29/R 13 绑定源不变，新适配身份分开。源码/数值检查、完整图构造成本、研究行为与最终设备目标没有混算。

## Validation

- Risk Level: **S0**，只审科学协议文档。
- Evidence Reviewed / Checks Performed: 阅读项目治理、Spec 008、相关 Decision 007/008/010/011/012、P3c2、P3d、P3e2；只读核对实际 G/R `run.json`；检查 `follow_scenes.py`、`follow_env.py`、`target.py`、`follow_task.py`、`training_contract.py`、`training_optimizer.py`、`evaluation_contract.py`、`evaluation_rollout.py` 和评估汇总/门槛源码。
- 没有运行测试、build、lint、模型、训练、全矩阵读取或新的实验。只读 JSON 的轻量命令不导入 Torch；没有操作 Noria/Docker。
- 未修改研究协议或业务代码，未提交；仅新增本独立验收报告。

## Issues

无必须修复项。

## Regression Risks

本 PASS 只确认研究契约足以约束下一阶段。具体研究存档与 runner 尚需实现；相同初始化及共同数值契约须通过真实数组和实际 G 等价门槛证明，不能由本报告替代。最终图、两侧完整训练和逐场景结果缺一仍是 incomplete；AC3–AC6 与实体目标保持未完成。

## Required Rework

无。协调者可将此科学契约标记 Ready，并继续独立的软件与实际子批。
