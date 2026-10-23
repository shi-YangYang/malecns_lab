# 研究目标、建模假设与数据

当前研究问题是：保留完整 MaleCNS 连接图，训练网络内部受约束参数，能否形成有效具身控制，以及真实连接结构是否提供作用。当前设备目标是[官方 X2 仿真与现场接入](../history.md#spec-015)；无人机目标跟随和单轴稳向作为[历史实验](../history.md)保留。

已有 V2 离线基线固定连接组和神经元动力学，只训练神经活动到平面动作的读出，保留为研究对照；它不代表网络内部训练已经成功。MaleCNS Lab 是计算与控制实验框架；装载真实连接图不等于还原一只果蝇的全部行为或认知。

## 当前模型包含什么

| 部分 | 当前实现 | 需要区分的限制 |
|---|---|---|
| 连接结构 | V2 使用完整发布数据的神经元参考图；旧设备链仍使用过滤图 | 两者必须分别记录身份，不混称完整图 |
| 神经动力学 | 点神经元 LIF、延迟、不应期及泊松输入 | 没有树突形态、缝隙连接、完整调质动力学或突触可塑性 |
| 视觉输入 | 软件计算亮度、光流与膨胀，再编码到指定群组 | 早期视觉计算不由连接组自行完成，视网膜映射近似 |
| 旧读出 | 人工映射、恒定巡航项与阈值逃逸逻辑 | 不应作为“学会飞行”的证据 |
| V2 对照读出 | 掩码岭回归，预测期望机身速度 | 只监督 vx/wz，尚未证明有效实体控制 |

LIF 参数参考 Shiu 等人的工作。现有配置使用 0.5 ms 积分步长，部分输出群组有人工设定的持续偏置。神经递质对突触符号的映射也是模型假设，不完整表达所有真实调节机制。

## 完整 MaleCNS 与旧过滤图

完整图实验使用 `data/malecns_v2/full_reference.npz`：166,700 个神经元、25,582,938 条内部有向连接、124,177,617 个突触计数，保留已发布内部非零连接、自环、孤立节点和缺失注释。来源是官方 v1.0 的 11 份 flat-connectome 文件，神经元与片段边界分别核对。已有资产无需重建；构建命令和账本见 [V2 使用说明](offline-research.md#model-v2-usage)。

旧 `malecns_lab build-brain` 仍服务既有设备实验：默认 `--min-synapses 3`，`--core-hops` 和 `--max-neurons` 还可继续筛图。其输出 `data/malecns_brain.npz` 与 `configs/malecns.yaml` 不属于完整图主实验，不用于覆盖 V2 资产。旧过滤影响、输入和输出映射见 [历史审计](data-model.md#male-cns-audit)。

真实连接结构需由计算模型产生模拟活动，不能把输出称为活体记录。LIF 数值语义、输入编码、偏置和符号映射应随版本记录；新训练内核不能仅因使用相同图就假定与原引擎或生物实验等价。

数据来自 [MaleCNS 下载页](https://male-cns.janelia.org/download/)。输出群为空时检查类型表达式和侧别，名字存在不代表成功匹配。原始数据、完整参考资产、旧图、运行记录都保留在本地，不随源码清理删除或混用。

## 怎样形成可检验的贡献

先登记任务、输入输出、训练参数和评估协议，再比较内部学习与冻结脑、直接视觉及受控拓扑对照。每一项主张需要对应证据：

| 主张 | 所需比较或实验 |
|---|---|
| 学到的读出有用 | 未见场景中与旧手写读出、均值、零动作等比较 |
| 连接组提供有效信息 | 与直接视觉特征控制器、合理随机化的连接结构比较，并报告容量与数据条件 |
| 时间结构有贡献 | 与无时序或神经活动打乱的对照比较，保持其他条件一致 |
| 能有效控制实体设备 | 在可控闭环任务中记录完成率、轨迹、延迟和失败情况 |
| 能迁移到移动底盘 | 冻结共享部分后更换适配器；若微调，报告新增数据与从零训练的对照 |

已有 [V2 学习工具](offline-research.md#learning-v2)提供掩码神经读出、直接视觉岭回归、训练均值和零动作比较；[V2 研究工具](offline-research.md#experiment-v2)提供拓扑对照的登记和执行基础。它们不代表对照实验已取得正结果。内部学习、冻结脑和拓扑对照工具继续保留，实际已执行范围见实验记录；后续机器人任务须在官方 X2 环境中另行定义。

按完整场景划分训练与评估，不能随机拆相邻帧。记录每类动作覆盖、不同方向错误、每段结果和处理耗时。平面任务只报告相关动作轴，避免由固定为零的轴形成虚高分数。相机固定时的成绩也不能推断飞行中自运动画面的表现。

尚不主张“全球首次”或已实现通用机器人控制。论文贡献应由方法、可复现数据和对照结果支持，项目命名或统一接口本身不足以证明新颖性。

## 来源与参考

- [SpikeCalls/FlyDrones](https://github.com/SpikeCalls/FlyDrones)：原始代码基础，保留 MIT 版权与许可。
- [MaleCNS 项目](https://male-cns.janelia.org/)：连接组数据、下载和对应论文；数据作者包括 HHMI Janelia FlyEM、University of Cambridge、MRC LMB 和 Google Research。
- Shiu P. K. 等，*A Drosophila computational brain model reveals sensorimotor processing*，Nature，2024：[论文](https://www.nature.com/articles/s41586-024-07763-9)、[参考代码](https://github.com/philshiu/Drosophila_brain_model)。
- Namiki S. 等，*A population of descending neurons that regulates the flight motor of Drosophila*，Current Biology，2022：[论文](https://www.cell.com/current-biology/fulltext/S0960-9822(22)00019-7)。这是既有 DNg02 读出的生物学参考，不能据此推导本项目读出权重有效。
- Maisak M. S. 等，*A directional tuning map of Drosophila elementary motion detectors*，Nature，2013：[论文](https://www.nature.com/articles/nature12320)。这是方向性视觉群组命名与编码的参考。
- Ache J. M. 等，*Neural basis for looming size and velocity encoding in the Drosophila giant fiber escape pathway*，Current Biology，2019：[论文](https://www.sciencedirect.com/science/article/pii/S0960982219301381)。现有阈值逃逸到无人机升降的转换仍是工程设计。

许可与依赖说明见 [THIRD_PARTY.md](../THIRD_PARTY.md)。
