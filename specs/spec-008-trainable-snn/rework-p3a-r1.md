# P3a R1 — 数据来源与轻量动力学语义校验返工

状态：Done，新实施者[报告](implementation-p3a-r1.md)和新独立[PASS](acceptance-p3a-r1.md)已读取处理，71项受影响检查通过，无返工项。首次独立[FAIL](acceptance-p3a.md)及原49项通过记录保留；随P3a提交，实际训练数据尚待封存，不覆盖首次失败报告。

## 必须修复

1. `neural_data` loader不得只信记录内truth。按已登记scene/camera/dt/tau构造既有轻量FollowEnvironment，逐帧核对像素、truth，使用已核对的executed_action推进，再核对next_truth/连续步号与结束状态。原有同帧教师、行为动作、有效掩码、来源/split/hash检查保留。无需加载神经图，不能通过放宽split来修复。补充真实val轨迹的frames/records/labels移入train、保留train声明并重算hash后的拒绝测试；这是原验收实际复现的缺口。
2. `policy_checkpoint`轻量读取严格校验固定模型键集合及已绑定实现的gain/gradient/event_order等动力学声明；固定tonic变体的formula/range/integration等也必须与现有语义一致，动态成员声明做格式与内部关系检查，实际图成员身份仍由生产参考加载复核。不能只用canonical hash证明语义正确。补充篡改gain等固定语义、重算所有嵌套ID后被读取直接拒绝的检查；保持restore全部验证后才写参数、生产入口拒绝数值角色。

## 范围和验证

仅修改两个未提交的新业务模块、对应两个新测试文件，以及必要使用说明；不改既有brain/neural_policy/环境/空间驱动/旧checkpoint或已完成实验源码。新增implementation-p3a-r1.md，不改原实施与FAIL。本次为两个相关校验缺口S2，定向执行这两个新模块的受影响检查；已有49项不为增加信心单独重跑，相关代码改动后可在这两个文件范围回归。禁止完整图、旧全套测试、硬件或服务动作。

新独立acceptance-p3a-r1.md通过后再提交整个P3a，随后才能封存四场景实际数据，P3b继续保持前置门槛。
