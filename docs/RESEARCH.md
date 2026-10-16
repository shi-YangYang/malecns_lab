# 研究目标、建模假设与数据

研究问题是：果蝇连接组产生的神经活动，能否通过可训练读出形成有用的运动目标，并在更换设备时复用。

当前阶段固定连接组和神经元动力学，先训练神经活动到平面动作的读出。CNSBridge 是计算与控制实验框架；装载真实连接图不等于还原一只果蝇的全部行为或认知。

## 当前模型包含什么

| 部分 | 当前实现 | 需要区分的限制 |
|---|---|---|
| 连接结构 | MaleCNS 数据构建的有向突触矩阵 | 节点和边数取决于过滤条件 |
| 神经动力学 | 点神经元 LIF、延迟、不应期及泊松输入 | 没有树突形态、缝隙连接、完整调质动力学或突触可塑性 |
| 视觉输入 | 软件计算亮度、光流与膨胀，再编码到指定群组 | 早期视觉计算不由连接组自行完成，视网膜映射近似 |
| 旧读出 | 人工映射、恒定巡航项与阈值逃逸逻辑 | 不应作为“学会飞行”的证据 |
| 新读出 | 标准化岭回归，预测期望机身速度 | 只训练读出，尚未证明有效实体控制 |
| MiniFly | 手工构建的合成网络 | 用于离线软件检查，不是实际果蝇连接组 |

LIF 参数参考 Shiu 等人的工作。现有配置使用 0.5 ms 积分步长，部分输出群组有人工设定的持续偏置。神经递质对突触符号的映射也是模型假设，不完整表达所有真实调节机制。

## 构建 MaleCNS

安装数据依赖后：

```bash
cnsbridge download malecns --dir data/malecns_v1
cnsbridge build-brain --data-dir data/malecns_v1 \
  --min-synapses 3 --out data/malecns_brain.npz
cnsbridge inspect --brain data/malecns_brain.npz
```

已有构建文件时无需重复下载和构建。数据来自 [MaleCNS 下载页](https://male-cns.janelia.org/download/) 所列公共存储，主要输入是：

| 文件 | 用途 |
|---|---|
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | 神经元 ID、类型、侧别与状态 |
| `body-neurotransmitters-male-cns-v1.0.feather` | 神经递质预测 |
| `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | 神经元对之间的突触数量 |

默认最小突触数为 3。`--core-hops` 可提取输入至输出附近的子图，`--max-neurons` 可进一步限制子图；这些设置改变实验对象，必须记录，不能都称为同一个全量模型。

以构建日志中的实际节点、边数为准。输出群组为空时检查类型表达式和侧别；群组名称存在不代表成功匹配了细胞。配置位于 `configs/malecns.yaml`，细胞类型可对照 [MaleCNS 项目](https://male-cns.janelia.org/) 与 [neuPrint](https://neuprint.janelia.org/)。

本地原始数据与构建产物不随源码提交。数据来源、过滤配置、特征模式和时间窗口应随实验记录保存。

## 怎样形成可检验的贡献

先建立无人机的数据与任务协议，再验证共享解码器。每一项主张需要对应证据：

| 主张 | 所需比较或实验 |
|---|---|
| 学到的读出有用 | 未见场景中与旧手写读出、均值、零动作等比较 |
| 连接组提供有效信息 | 与直接视觉特征控制器、合理随机化的连接结构比较，并报告容量与数据条件 |
| 时间结构有贡献 | 与无时序或神经活动打乱的对照比较，保持其他条件一致 |
| 能有效控制实体设备 | 在可控闭环任务中记录完成率、轨迹、延迟和失败情况 |
| 能迁移到移动底盘 | 冻结共享部分后更换适配器；若微调，报告新增数据与从零训练的对照 |

当前离线训练工具已经提供神经读出、直接视觉岭回归、训练均值和零动作比较。其余对照与实体评测仍是后续工作；不能将上述表格写成已完成结果。

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
