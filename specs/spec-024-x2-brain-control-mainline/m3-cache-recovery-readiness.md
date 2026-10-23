# M3 冻结特征缓存恢复可行性

2026-10-23。只读评估；结论：**缓存现存证据足以支持一次严格复用的实现，但当前入口尚不支持复用，不能直接续跑或部署中断候选。** 本次仅写报告（S0），没有改源码、原 run 或 checkpoint，没有启动 Torch/MPS、完整图、训练或仿真，没有读取 locked-test 文件。

## 原 run 的真实终态

`data/x2_behavior_readout_20261023_r1` 保留为失败：supervisor `wall_limit`，5399.7657199589885 秒，returncode −15；外层 summary 5400.137580125011 秒，无 worker_result。registration ID 为 `d051209beb7573fd64dfe4da9795866da2f15083cb0e0fec1a3debf9bbc7cccd`。

- `updates.json` 有 **167 条完整训练加验证 history**；phase 指向第 168 步，不能只据 phase 判定已更新。
- `gradient-audits/0168/audit.json` 实际记载 backward、optimizer.step、参数校验均完成，`optimizer_steps_after=168`。该步尚未完成并归档验证 history。故实际已完成 168 次参数更新，只有 167 次完整更新后验证；没有完成登记的 200 次。
- 168 号 `parameters-after.npz` SHA256 为 `453764028b2c769f9165084605eef6729fae49bf94866e660b251161803ffbbd`，本次重新核对匹配；7 组参数全部有限，theta 与 r2 initial 逐元素完全相同。全部 12 个已保存 checkpoint 的 theta 也与 initial 相同。这是已落盘状态的证据；无正常 worker 终态，不能声称已完成最终全程封存或全部 Adam 状态审计。
- 已落盘 history 中最低验证 loss 出现在第 **149** 步，`3.036942720413208`；checkpoint ID `79482e9529d5b308b9126fc9a25e1d5eee227b93625a60a5b27081f7e7d521f7`，arrays SHA256 `fcc4c9ab0b718f975f56e83682301599c2fb3f75cd5abb9ed6aa829968959183`。这是中断 run 的**诊断候选**；`selection.json`、`validation-scores.json` 均不存在，不可作为正式基线或部署候选。本次没有补算任何分数。
- peak observed RSS **1,619,968,000 bytes**；MPS current peak **1,100,497,408 bytes**，driver peak **1,525,497,856 bytes**，分别报告，不相加。

## 已核对的缓存证据

只用 stdlib 与 NumPy，一次顺序审计耗时 **24.6007 秒**。37 份来源快照 SHA256 全部符合原 registration，当前对应 37 份源码也仍相同。registration、policy、input_map、checkpoint、dataset 身份均自洽；checkpoint metadata/NPZ hash 已核对。初始 checkpoint 固定为：

`data/x2_behavior_probe_20261023_r2/initial-checkpoint`

ID `394e39ec8eed0b622712fe2ffeca20d247e1ed437a9e6945479716eb007f3959`，NPZ SHA256 `c25fd6aa725968de8fcd15363e41eaa69f88f6411d7c619d6648608da67045b7`。其 7 组初始参数逐数组身份和有限性通过，没有 optimizer 数组。

| 内容 | train | validation |
|---|---:|---:|
| 固定顺序的完整 episode | 50 | 50 |
| frame JSON/NPZ 对 | 416 | 402 |
| state JSON/NPZ 对（含预热态） | 466 | 452 |

- 对 train/validation 文件核对 manifest 文件 hash、内容 hash；按登记的 task-round-robin 重新推导前 50 条，与 `selected_episodes.json` 次序完全一致。每 episode 的 sequence hash、全部帧 Observation 一致。
- **818 个 frame NPZ** 全部匹配各自 `arrays_sha256`，无重复/多余数组，全部 float32、有限。实际字段：`encoded_input(3377)`、`features_mv(2628)`、`behavior_logits(5)`、`phase_logits(3)`、`parameter_raw(1)`；记录的三头数值与 NPZ 一致。只能取 `features_mv` 给头训练，不能把任务、标签、旧 logits 或编码直接输入头。
- **918 个 state NPZ、21,114 个状态数组** 的文件 hash、逐数组 hash/shape/dtype、有限性全通过。每个状态的 canonical ID、policy/session/frame/observation 序号与 600+500×frame 神经步数均一致；逐帧 decision 的 before/after ID 串联相邻真实存档状态，checkpoint/policy/input_map/request 身份均正确。100 个 episode 的预热初始状态数组完全相同。归档 host_monotonic 随登记 episode/帧次序严格递增。
- frame NPZ 合计 **8,954,893 bytes**；state NPZ **2,506,266,255 bytes**，无需复制这 2.5 GB 状态进新 run，也无需把它们载入神经状态。校验应逐个读取，训练只保留约 8.60 MB 的 818×2628 float32 特征。
- 目前没有独立封存的 cache manifest；frame JSON 自身也没有统一上层 hash。恢复实现必须在新登记中封存完整路径/文件 hash 清单并绑定原 registration，之后在 worker 使用前复核。现在的只读结论不能代替新 run 的 admission 检查。

这些检查验证归档的身份和完整性，不重新证明每个神经积分步骤；没有重跑脑。当前图文件未在本次只读审计中重新读取，恢复 worker 仍须沿既有逻辑核对真实图 SHA256、完整图结构与 policy 严格恢复。

## 最小实施方案与拒绝条件

1. 在现有 CLI/execute 增加显式缓存来源参数，仅允许 `mode=readout`；新 output、新 registration。固定来源为上述失败 run，不修改其状态、source、hash 或内容。新 registration 写入 cache schema、来源 registration ID/hash、完整文件清单 digest、初始 checkpoint、policy/input_map、两个拆分和确定 episode 次序、缓存生成的源码/runtime、恢复自身的源码/runtime。原 200 次上限、50/50、seed2401、Adam、loss、验证选择方式全部不变。
2. 用轻量 admission/loader 核对上述所有条件，拒绝符号链接/越界路径、缺帧/重复/多余 episode、NPZ 多余字段/对象数组、非有限或错误 shape、来源/观察/序号/状态链/身份/hash 不一致。只允许读 train 与 validation；locked_test 文件不打开、不重生成以求兼容。缓存来源必须是冻结 theta 的 readout、同一 initial，而不是 joint 或任意中间 checkpoint。
3. 保存并验证来源快照；新旧源码不能因新增恢复编排而要求 registration ID 相同。允许明确登记的训练编排/CLI/缓存校验差异；脑、输入编码、数据选择、头/loss、数值 runtime 的语义不得漂移，policy/input_map 必须严格相等。未知差异拒绝。当前源快照完整；恢复实现改动后由新登记记录新来源，绝不改旧 hash。
4. 可保留完整图加载和 `restore_checkpoint`，恢复 **r2 initial 的全部 7 组参数**，然后冻结 theta、创建全新 Adam。不读取原 run 的中间 heads/optimizer，不从第 169 步接续。分支只载入已验证 `features_mv`，沿用现有 `cached_outputs`、200 次完整 episode 更新、每步全验证与最早最低 loss 选择。缓存模式显式禁止 `initial_state`、`forward`、神经 `_step` 被调用，调用即失败；原 cache 生成函数不得被触达。
5. 封存新 200 步完整 history、正式 selection、验证预测、来源 cache manifest、初始和最终 theta 相等、零脑 forward 的执行证据。任何身份/资源/数值失败立即结束；不以跳过坏帧、减少样本、增步、换 seed、调学习率或读取测试来修复。原 168 次更新不合并进新实验计数。

实现涉及训练入口、缓存来源及持久化身份，按恢复边界做定向检查：完整 fixture 可读、缺失/篡改/错序/换 checkpoint 拒绝，以及缓存分支不会调用脑 forward；执行一次最相关训练模块检查并独立审查。无需重跑已有完整图探针、全项目测试或桌面构建。正式运行前登记新目录与一次上限。

## 一次恢复的有限预算建议

原 run 的首个头更新 phase 在 **5255.1596 秒**；168 个更新 phase 之间的 167 个间隔平均 **0.859734 秒**、P95 **1.086012 秒**、最大 **1.932597 秒**。因此 200 次头更新按均值约 **172 秒**，按已观测最大间隔约 **387 秒**；完整缓存审计实测 24.6 秒，完整图恢复前的原启动至首个 episode 约 3.73 秒。

建议新 run **一次、总墙钟 600 秒、固定 200 次**：即使为父/worker admission 各预留约 25 秒、完整图恢复和最终封存合计 30 秒，按既有最慢更新间隔计算仍约 467 秒，剩余约 133 秒覆盖 I/O 与调度。沿用 CPU4/interop1、RSS/MPS min(6GiB,启动可用−1GiB)、系统预留1GiB、无 CPU fallback、唯一完整脑锁，期间仿真关闭。600 秒是新缓存恢复登记的完整预算，不是延长旧 5400 秒；超限仍失败且不自动再试。

新 run 完成也仅能得到模拟反馈/工程教师的离线读出基线；不据此通过 P3 实际反馈、P5 动作或 P6 真机。是否运行后续 joint 应另作决定，不从已见分数更改本次固定恢复方案。
