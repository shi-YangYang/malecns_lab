# MaleCNS Lab 项目规则

## 工程与资产

- 包、CLI 为 malecns_lab，源码在 src/malecns_lab；learning 可依赖 control 动作契约，control 不反向依赖 learning。
- brain/senses 负责神经计算与编码；diagnostics 负责离线神经观察；visualization/apps/atlas 提供刺激响应与记录回放；experiments/tello 是已有设备实验。
- 当前设备方向按 [Decision 021](../decisions/021-official-x2-and-cleanup.md)：官方 X2 Sim + MC + AimDK，先尝试 Mac Docker。G2、第三方步态及单轴稳向前置不在当前主线。
- data/、logs/、模型、录像、环境、run/sources 不因清理源码而删除或纳入 Git。保留 LICENSE 与 THIRD_PARTY.md。
- cnsbridge.*、flydrones.* 序列化标识是旧数据契约，不是可删除的导入兼容层。模式、特征顺序与来源身份变化必须明确迁移。
- 不改写旧 run/checkpoint/source 哈希以伪装续跑；旧实验使用其记录的源码与环境，新实验重新登记。

## 动作与研究

- BodyVelocity 按 FLU 为 (vx, vy, vz, wx, wy, wz)，平移 m/s、角速度 rad/s，正 wz 左转。它是目标，不是测量结果。
- 旧 FlightCommand 横移/偏航正值向右，转换属于适配器；Tello RC 不能直接称精确 m/s。
- 旧 MotorDecoder、冻结脑岭回归、内部参数联合学习分开标记；人工监督和旧试飞不能证明新模型学会控制。
- 只使用真实连接组资产，禁止 MiniFly、合成网络替代或缺失时回退。完整参考保留已发布神经元之间全部非零连接、自环、孤立节点与缺注释节点；任何简化另立受控实验。
- 分清仿真时间、实际处理耗时、命令频率；训练按完整场景隔离，不把相邻帧随机拆分造成泄漏。教师标签、旧 RC、实际运动不是同一种证据。
- 保留失败结果；连接组优势、实体控制、可靠避障、跨设备迁移和全球首创必须有独立证据。详见[研究边界](../../docs/RESEARCH.md)。

## 验证与设备

按 [AGENTS](../../AGENTS.md) 的 S0–S3 和停止规则执行。硬件动作、大型完整脑运行只在当前授权内；离线测试不触发设备。Noria 不得停止，Docker 不得为本项目擅自重启。当前授权为独立容器中的官方 X2 安装与启动验证，不训练、不连接真机、不推送。
