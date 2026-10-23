# 当前交接 — X2脑控整条主线实施中

更新：2026-10-23。工作目录 `/Users/yang/project/malecns_lab`。

## 当前状态

**2026-10-23 最新交付要求：先本地commit保存当前修改，再讨论按AGENTS的目录职责调整。用户首先询问旧CPU Docker文件是否能删；本轮只核对依赖，不删除/重构，不恢复训练和实验。当前按钮到对应动作没有跑通，原因尚未定位，不得仅归因于模型能力；本次提交是未完成阶段快照。**

**最新用户决定：先暂停roadmap训练、构建和实验，急需打开应用查看。活动goal已标记paused，不自动恢复。** 已创建忽略的`data/x2_behavior_desktop.json`，指向读出r2的`checkpoints/0149`（ID `3f0c556df9cf8f306f38b328ac25b6154838aa17b0305621dbe113f4fc1ac5b4`），只作qualification候选；入口校验通过，Electron已打开且停在“未运行／准备验证”，未自动启动脑或仿真。读出r2 session10932已完成code0，总251.6703秒、200更新/200验证、零神经调用、theta不变；验证完整序列0/50、行为F1 0.15924、阶段F1 0.45190，不能称模型合格。实际结果独立审查被用户打断，尚未完成。M4固定1/2/2 profile工程已独立PASS；M2兼容发送实现17项CPU通过，独立验收及prepare/build/inspect均未进行。不要据旧下文“运行中”重启任何任务。

最新用户要求“把你刚才说的roadmap主线都跑了”。当前活动目标覆盖[Spec024](../../specs/spec-024-x2-brain-control-mainline/spec.md)的M1–M6实施与运行，不再止于设计或单一阶段。用户确认暂时没有真机，之后上海接入；M6保持待现场，不能把本地阶段完成称为整条主线完成。当前M1契约/编码/数据、M2官方技能执行器、M3新脑策略分开实施；进度和证据以[progress.md](../../specs/spec-024-x2-brain-control-mainline/progress.md)为准。有限本地训练与GPU仿真在新授权内，无需重复询问；实际运行按登记预算串行安排。

首批行为为HOLD、ADVANCE、TURN_LEFT、TURN_RIGHT、WAVE_RIGHT。待命/前进及两秒双向转向均已有单次实测通过；一秒左转失败保留。挥手首次实际请求ACK超时、UNKNOWN，尚未完成或确认取消。预设动作不具有连续速度的控制源优先级；SDK定义或服务ACK不能当作动作完成。

新模型要增加任务/反馈工程编码和行为/阶段/参数读出，跨行为保留状态。读出只接神经特征；外层执行器不得按任务脚本选择下一动作。普通“结束互动”入脑，独立紧急停止与保护另记。旧theta仅作明确初始化候选，旧无人机三轴读出和policy身份不直接复用。

实施顺序按[路线](../../specs/spec-024-x2-brain-control-mainline/plan.md)：M1新行为/观测/执行契约与数据，M2官方技能核验，M3专项学习，M4桌面联动，M5展示准备，M6上海现场。P1契约工程已PASS，P2–P6仍未完成。不要因阶段通过、单个checkpoint或预算耗尽而标记总目标完成；也不为主线自动无限续训或绕过现场设备条件。

Spec023评估保留为历史限制证据：[实测](../../specs/spec-023-x2-task-evaluation/actual-evaluation.md)15帧全FORWARD、STOP0/9，工程验收PASS、任务行为FAIL。原始run `data/x2_task_eval_20261023_r1/` 与未提交源码保留；不复跑、不把这15帧当主线训练/验收集。此前“下一步只训练红目标前进/停止”的建议已被用户最新方向替代。Spec023与本轮设计改动均未commit/push。

## 可用工程基线

桌面 [Spec022](../../specs/spec-022-electron-workbench/spec.md) P1/P2 已完成并[独立PASS](../../specs/spec-022-electron-workbench/p1-p2-acceptance.md)，本地提交 `6ba3fb4`，未推送。保留用户认可的左右布局，不恢复已删除的大标题、底部播放条及历史小字。左侧官方GPU仿真，右侧同一冻结MPS脑的真实22群组/32选样输出神经元遥测；每50步采一批，无另起CPU脑或全脑逐步归档。

原生桌面两轮前进0.381m/0.446m后由控制流程逐轮停稳，证据在[正式接入实测](../../specs/spec-022-electron-workbench/actual-integration.md)。该动作链路成立不等于模型自行判断停止；性能配对、失败记录及旧数值证据留在对应Spec，不重跑已通过检查。

## 入口与边界

- 已构建开发入口：`cd apps/desktop && npm start`。首次准备/修改构建按 `apps/desktop/README.md`。Electron 45.0.0官方本地依赖，尚无安装包/签名/跨平台交付。
- 当前候选验证入口 `.venv-train/bin/python -m malecns_lab.experiments.x2_behavior_desktop --execute --sim-only --qualification --output <新desktop-run>` 由主进程封装，需宿主固定候选配置；缺候选明确失败。旧x2_desktop保留历史职责；renderer无任意命令或路径权限。普通 `.venv` 不含Torch。
- 私有runtime `data/x2_sim/gpu_probe/runtime-rebuild`，容器 `malecns-x2-gpu-trial`，noVNC 6081；不接管已运行资源，不启动旧CPU仿真，不切换Docker context。用户要求不用仿真时关闭以减少发热。
- 既有桌面实测是里程计虚拟标记、纵向离散动作、停稳后计算；新的X2专项读出r1超时失败，正实施严格缓存恢复，尚无通过行为验收的候选。真实相机、连续控制、真机与仿真视频归档未完成，SDK未改造。GUI帧率不是脑计算频率：历史桌面共驻留每帧约3.64s，Spec023无仿真离线均值2.371s，不作跨环境性能对比。
- Spec023结束时GPU VM与旧CPU仿真关闭、完整脑已退出；Spec024已进入实施，具体运行状态见progress.md。不动Noria，不重启Docker。用户要求不用仿真时关闭以减少发热。
- 模型、资产、依赖安装目录、日志及全部run保留在本地且不入库；不覆盖或清理科研资产。Spec023的11项定向检查和唯一完整图评估已完成，不机械复跑。Spec024新协议、控制与训练按S3检查受影响子系统。

## Spec024 当前实施证据

- M1/M2工程已独立PASS；第七轮skills-20261023-170921-07ed4474中两秒左转+0.07166989rad、右转−0.05984628rad均通过实际运动与停稳。挥手ACK超时，无task_id/PLAYING，UNKNOWN；整轮179.79秒后资源关闭。准备时额外一次Pause的偏差已记录，不冒称原流程可靠。静态诊断已确认MC随包6字段与当前SDK7字段的预设请求结构差异，官方v1.0.0 Set及递归依赖匹配但缺新版状态消息，正按m2-preset-compat-contract.md实施隔离旧版发送进程；未改厂商源码或重试。历史七轮全部保留，见m2-actual.md。
- M3稳定探针已完成一次有限跨窗更新，前向等同、梯度有限；候选不部署。正式冻结读出r1触及5400秒失败，session53078已退出：818帧缓存全部完成，168参数更新/167完整验证，未完成200及最终选择，见m3-readout-actual.md。全部818帧/918状态/37来源只读校验通过；缓存初版8项CPU检查通过，独立FAIL发现新run数据未绑定worker已审字节，roadmap_m3_cache_binding_repair限定修复后1项worker回归通过，新的独立复验PASS。新读出r2 session10932已启动，同r2 initial+fresh Adam固定200步、600秒、不再脑forward，不把原run改成功或部署149诊断候选。locked_test未打开。joint完整40条方案仍Draft，等正式恢复基线再决定，不同时改joint。
- M4接口、固定相对朝向及实际反馈来源关联均已独立工程PASS，7项最新定向CPU检查通过；目标不是视觉识别人，尚无新M4实际run。M5a、M5b限定修复均已独立PASS；M5b的stop前排队块重复final问题已关闭，继承4项CPU检查，没有重复构建；仍无真实采集/编码/解码及首帧锚点。wave取消资格保持关闭，官方语义缺口及现场问项见m2-wave-stop-boundary.md。当前只有独立缓存读出r2运行，GPU仿真/私有VM关闭，默认Docker context仍desktop-linux，Noria未操作。
- 当前分工、实际run目录及下一步登记都在progress.md与对应报告；不重做通过的检查，不自动续训、调参或打开锁定测试。

## 保留的工程基线

Spec016–018 官方动作/停车契约与适配器、Spec019冻结推理、Spec020精确等价优化、Spec021两轮纵向反馈均已PASS。Spec021本地提交 `fbc5b2d`，原实测0.41884m/0.42901m后逐轮停稳；完整166700节点、25582938非零边、124177617原始连接计数。旧基线数值与失败记录见各Spec，不以桌面完成扩大科学或真机结论。
