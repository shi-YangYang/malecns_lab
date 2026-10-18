# Acceptance — P3a 实际教师数据 r1

## Result

**PASS**。由未参与 P3a 实施、返工或本次数据采集的新独立验收者完成。确认 `data/neural_teacher_20261019_r1/` 的封存资产与 [实际报告](p3a-data-run-20261019-r1.md) 一致，可作为已经冻结的 P3b 输入。该结论仅涉及实际仿真教师数据，不是神经训练或实体控制验收。

## Spec Coverage

- **身份及来源封存一致。** 独立用规范化 JSON 和 SHA256 计算的数据集 ID 为 `f0fe9735abbffff6135a8a5d7c8f3943d638da90bfe6a40eb21ba53583137076`；manifest 字节摘要为 `2418141cc7711c64cfc27f7bd8056150b8a170dd0ebed3a3079c25bffecb1104`。全部 7 个来源文件摘要、12 个 NPZ/summary/trajectory 摘要均与当前文件匹配。
- **完整四场景及隔离一致。** manifest 的四个完整 scene 定义逐项等于当前 `scene_registry` 中按 ID 排序的全部 train 场景；summary 重复声明一致，没有 val/test 场景。相机为 160×120、BGR uint8，环境 dt=0.1 s、response_tau=0.25 s；轴与单位为 FLU vx/vz/wz、m/s/m/s/rad/s。
- **320 帧完整且可追溯。** 每个 NPZ 均具有严格六字段、已声明 shape/dtype、有限数值；每场景 80 帧、全部有效，时刻精确等于 `arange(80)*0.1`。轨迹同帧 time/valid/teacher/executed 与数组一致；相邻记录 next_truth/truth 连续，最后推进到 8.0 s，所有 failure 为空。没有删帧/补帧的记录缺口。
- **监督与行为分开。** 从每条已存 truth 独立代入教师公式重算三轴原始标签，与 teacher_raw 精确一致；逐轴裁剪到 `[0.3,0.2,0.5]` 后与 teacher_target 精确一致。目标和执行动作均在边界内。全部 320 行执行动作与目标存在至少一个精确数值差异；这不代表显著误差，也不把执行动作当标签。
- **报告统计全部匹配。** 从原始数组重新计算四场景三轴正/负/非零计数、最小/最大值、29 行限幅计数，以及执行动作差异，与 `collection_analysis.json` 逐字段一致；报告中六位小数的范围与原值相符。轨迹 qualified 重算与 summary 一致：combined/forward/vertical/yaw 合格比例分别为 0.75/0.75/0.55/0.8375，success 为 false/false/false/true。ready 表示采集与来源可用，不把前三个行为未达标场景删除，也不声称视觉基线全部成功。
- **P3b 固定输入满足前置条件。** 已冻结协议选择 combined 的索引 0/1，不根据结果改选。两帧均有效，时刻 0.0/0.1 s；目标分别为 `[0.3,0.09630814167877251,0.20954077112171]` 和 `[0.3,0.1026919353212648,0.21961795602601253]`，与实际报告和派生统计一致。这里只确认输入准备，不判断 P3b 更新效果。

## Validation

- Risk Level：**S0（本次运行报告与已有封存资产的事实验收）**。P3a 持久化/来源软件风险及 R1 校验缺口已由 [软件独立 PASS](acceptance-p3a-r1.md) 处理，不重开软件验收。
- Evidence Reviewed：项目规则、使命与技术约束、验收流程、当前交接；P3a Spec、R1 PASS、实际运行报告、P3b 冻结协议；当前 registry、采集/标签和规范化摘要实现；manifest、全部原始数值序列/轨迹/summary 及派生统计。
- 执行一次只读 NumPy/hash 审计，退出 0，输出 PASS。只调用纯场景注册函数取得定义；没有调用 collector、生产 dataset loader、环境推进/重放、测试或神经模型。没有修改源代码、数据资产、系统服务或硬件。
- 协调者提供的一次 CLI exit 0/ready 与随后一次生产 `read_neural_data` 登记环境来源重放成功记录已纳入证据。独立验收检查封存字节、来源定义和报告事实，**未重复该环境重放**，亦不把本次检查描述为重新生成并逐像素复现了四场景。

## Issues

无本批必须返工的问题。

## Regression Risks

本批是已知标记、确定性简化环境中的仿真教师数据；320 帧全部有效，因此本次实际数据不提供丢失目标帧的运行证据。四场景包含共享初始偏差；P3b 选定两帧三轴标签均为正值，不能据此声称负方向、长期闭环或泛化能力。未加载 MaleCNS、运行优化或硬件；总体 AC3–AC6 仍未完成。

## Required Rework

无。由协调者更新实际报告状态并提交本阶段；本次验收停止。
