# M3 冻结特征缓存恢复

2026-10-23。工程实施 Ready，实际运行须在新的独立工程验收后登记。风险 S3，范围为训练入口/登记/缓存校验及相关CPU检查，不改策略、图、编码、教师、拆分或已有run。

## 具体原因与固定实验

`x2_behavior_readout_20261023_r1`已按5400秒上限退出，supervisor为wall_limit、worker_result缺失，正式实验失败。416训练帧、402验证帧均已计算；history完成167条，168号梯度审计完成更新，但没有完成200步和最终选择。不得延长、修改原run或把其诊断最低loss候选直接部署。

新的独立run从同一个 `data/x2_behavior_probe_20261023_r2/initial-checkpoint` 初始化全部参数，重置Adam，使用原50train/50validation顺序及完整缓存，从第1步重做固定200次读出更新。theta仍冻结；loss、lr、seed、每步完整验证、最小验证loss/最早平局选择与最终评分完全沿原readout契约。不能从第149步、末次或“最好分数”续接，不能增加训练步数、换数据、调参或打开locked_test。新run不是原run成功续跑，也不训练内部theta。

## 缓存准入与新身份

- 显式可选的只读缓存参数只允许readout；不提供参数时保留原行为。缓存校验失败必须退出，不能自动重算整套特征或选部分可用帧。
- 原缓存是本地已退出r1。核对其registration完整性、原来源快照、dataset/input_map/policy/initial-checkpoint身份、select_episodes的顺序和train/validation分区；禁止读取locked-test内容。
- 校验全部818个frame JSON/NPZ：NPZ hash、精确数组名/dtype/shape、有限性、实际Observation、episode/frame/decision序号、checkpoint、前后state身份。逐episode核对918个state JSON/NPZ的hash/数组身份、session/step/序列连续性及实际前后链，不能只验文件数量或首尾一帧。所有100个初始状态与登记预热策略相符。
- 原source快照必须与原registration逐项一致；本次允许变化的训练入口/缓存加载实现明确登记为新source。数值策略、编码、图、模型身份等来源必须仍与缓存一致。不能因为新loader必然改变训练source，就忽略所有来源差异或改写旧hash。
- 新registration绑定缓存根目录、原registration ID、初始checkpoint、完整缓存清单/hash和校验版本；实际读取使用已核对文件字节并再次核对必要身份，避免校验后换文件。原约2.5GB状态资产不复制、不改写、不删。
- 允许沿现有严格加载路径构建完整图/恢复策略用于身份与checkpoint校验；**缓存路径不得调用brain forward、warmup或生成新神经状态**。保留各头使用现有`read_features`和既有梯度审计，不以新分类器/CPU近似实现替代。
- 结果明确写出特征来源、冻结theta、完成更新数、验证选择及theta前后一致；原run仍failed。没有实际反馈/P3/P4/P5能力结论。

## 一次预算与检查

原run实测头部约0.86秒/更新，200次约172秒；全部缓存/状态CPU审计约24.60秒。新run总墙钟固定600秒，包含校验、原严格图/策略恢复、200更新和最终归档；不是无限恢复。仍CPU4/interop1、MPS无CPU fallback、单完整脑、至少1GiB系统预留及重新测量RSS/MPS预算，仿真保持关闭。当前所有重资源已退出。

由新实施者完成最小加载/登记路径。一个受影响训练子系统CPU检查覆盖完整缓存准入、身份/内容/次序/状态链损坏拒绝、来源允许范围、禁止locked_test、缓存路径零forward与200步/选择语义；不启动完整图作为软件测试。新独立验收者不机械重复通过项。通过后协调者填写新目录和具体registration，再实际执行一次；失败保留，不自动续训。

联合训练完整40条方案继续Draft，须等待本恢复产生正式有效基线，再按其独立条件决定；本契约不授权改joint调度或延长旧实验。
