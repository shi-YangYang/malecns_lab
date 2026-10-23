# Spec024 主线实施进度

更新：2026-10-23。**最新：用户要求先暂停roadmap并打开应用查看，goal已paused，不自动恢复训练/构建/实验。** 当前Electron已打开，候选配置指向读出r2的0149；停在未运行，未启动脑/仿真。此前授权覆盖完整roadmap，不止设计或M1。用户确认“暂时没有真机，之后到上海接入”；先推进全部可在本地完成的阶段，M6硬件验收保持未完成。没有commit/push授权。

| 阶段/门槛 | 当前状态 | 可核对证据/剩余工作 |
|---|---|---|
| D0 | 设计PASS | spec.md、plan.md、acceptance.md；不代表产品完成 |
| M1 / P1 | 工程独立PASS | m1-m2-final-acceptance.md；真实完整映射由M3探针核验 |
| M2 / P2 | HOLD/ADVANCE及两秒双向转向有实测PASS；挥手FAIL | 第七轮左转+0.07167rad、右转−0.05985rad，均停稳；挥手ACK超时、UNKNOWN，无有效挥手证据。整轮179.79秒，资源已关闭；已确认Set请求版本差异，按官方旧版隔离发送方案实施，不直接重试 |
| M3 / P3、P4 | r2冻结读出完成，行为能力不足；后续已暂停 | 818特征全部完成，168更新/167次完整验证；5400秒上限未完成200步。全部818帧/918状态/37来源只读校验通过，原8项及返工1项CPU检查通过，m3-cache-recovery-final-acceptance.md确认R1关闭；新r2完成200更新/200验证，用251.67秒、零神经调用、theta不变；验证序列0/50、行为F1 .15924/阶段F1 .45190。实际结果独立审查被打断，joint仍Draft |
| M4 / P5 | 接口/朝向/反馈来源关联均独立工程PASS | m4-feedback-provenance-acceptance.md确认同步事件与实际入脑快照关联修复；尚无实际新M4运行、合格checkpoint/P5证据，挥手取消资格仍缺 |
| M5 | M5a、M5b工程均独立PASS | m5-video-final-acceptance.md确认末块修复通过，继承4项CPU检查，未重复构建；视频无经验证首帧锚点，仅单独回看。真实采样/编码/联合视频仍未验证 |
| M6 / P6 | 现场准备文档已写，待设备 | m6-onsite.md记录版本矩阵、接入顺序及30轮验收；上海现场证据仍缺 |

## 当前执行约束

- 新schema、模型与控制执行风险按S3审查；实际检查集中在受影响子系统，不默认全仓测试。
- 业务实施者按契约/编码/数据、官方技能、新脑策略三个不重叠文件范围工作；协调者维护规范、安排真实运行；每阶段由新的独立验收者检查。
- 重资源只由协调者启动，同一时间一个完整脑；训练与GPU仿真错开，真实联调才共驻留。默认桌面/预览/离线测试不动作。GPU仿真不用时关闭，旧CPU仿真不启动。
- 每次实际运行先登记模式、输入/模型/源码身份、输出新目录、RSS/MPS/至少1GiB系统预留、墙钟/更新上限和停止条件。有限失败是证据；不得无限续训直到得分通过或反复查看锁定测试调参。
- 保护Noria、默认Docker context、旧run/checkpoint/sources与Spec023未提交工作。没有实机时推进代码、离线与仿真，保留现场验收缺口。

## 当前分工

- 协调者：r1训练session53078及第七轮技能session9289均已退出。独立缓存r2 session10932已完成code0；当前只保留Electron空闲窗口，仿真/私有VM保持关闭、锁定测试未打开，Noria未操作。旧r1不延时、不改成功状态。
- 第七轮新run为skills-20261023-170921-07ed4474；准备时额外一次Pause才明确观察PAUSE，已如实记录流程偏差。截止前51.69秒交接，两秒双向转向通过，挥手ACK超时；m2-wave-ack-diagnosis.md已确认MC的SetMcPresetMotion为6字段，当前SDK为7字段；m2-official-message-compatibility.md确认官方v1.0.0的Set及递归依赖匹配，但缺新版状态类型；按m2-preset-compat-contract.md实施双进程隔离发送，不改厂商源码或立即重试。
- M4反馈来源修复已独立PASS；7项定向CPU检查通过。发现默认1秒转向与第七轮2秒实测不对应，按m4-skill-profile-contract.md由`roadmap_m4_skill_profile`接入固定1/2/2候选profile，8项CPU检查已通过，`roadmap_m4_skill_profile_review`独立PASS；不改模型强度、不宣称全强度标定，无实际运行或新训练数据。
- M5a工程PASS；M5b末块R1限定修复已由`roadmap_m5_video_final_review`独立PASS，仅继承4项CPU检查，未重复检查或启动资源。真实录制与同步仍待验。
- 缓存初版8项CPU检查通过；m3-cache-recovery-acceptance.md独立FAIL发现worker重读新run数据未绑定已审字节。新实施者`roadmap_m3_cache_binding_repair`已完成限定修复，1项直接worker回归通过（首次fixture比较问题修正后重跑），已由`roadmap_m3_cache_final_review`新的独立复验PASS；实际命令/新r2目录见m3-cache-recovery-run.md，已完成session10932；用户要求暂停后续工作。旧实施者`roadmap_m5_capture`已转M2官方消息兼容边界，不再改训练/桌面。
- M2挥手接口只读核对已结束：m2-wave-stop-boundary.md确认公开模式切换可返回MotionBusy，未找到保持平衡并取消owned挥手的官方契约；JD不是动态平衡取消方案。自然完成探针保留，wave_cancellation_verified=false，现场问项已明确。
- m3-real-feedback-readiness.md已完成：M2单技能不能伪补完整互动样本，后续真实数据从M4实际入脑观测/执行反馈关联导出；v1和locked_test保持原样，尚未实施导出或生成新数据。
- M3联合训练准备审查已结束：现50条轮转200更新只覆盖299/416不同帧；完整40条（五任务×八情境）需175窗口、325帧，加完整验证预计超过7200秒。m3-joint-complete-contract.md保留9000秒有限方案Draft；原r1超时失败，先取得正式缓存恢复基线，再决定joint，不与当前缓存实现混改。
- M5展示操作与恢复说明已写在m5-showcase-runbook.md，三个流程均只提交任务；没有完整展示实测。
- 初版16项M1、13项M2、8项M3定向检查是已有证据，不能覆盖独立发现的具体问题；返工只补直接回归，随后由新验收者审查。

目前已有四项技能实测和一次联合更新探针证据；尚无合格专项模型、完整P2或桌面P5产品验收结果。后续以真实产物逐项更新。
