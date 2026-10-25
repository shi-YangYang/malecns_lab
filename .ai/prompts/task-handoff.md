# 当前交接

更新：2026-10-25。工作目录 `/Users/yang/project/malecns_lab`。

## 最新用户方向（2026-10-25）

用户要求把前进、左右转、挥手和普通结束互动经真实果蝇脑输出打通，优先复用已有模型做接口适配。桌面默认 `runtime.x2.behavior`，本地候选 `data/x2_model_redesign_20261024_r1/b_current_history_v1`；完整冻结 MaleCNS + 原 B 三成员读出，权重/教师/标签/评分未改，未训练，没有 task→action 旁路。普通结束互动入脑、急停独立；遥测来自同一脑实例。

最终 FIFO 修复版实测：`080524-3f93cf12` 的 GREET 单次挥手→停稳，以及 SHOW_FORWARD 前进 0.401150714 m→挥手→停稳均完成；`082249-9cc89f32` 左/右实际转 +0.226847081/−0.232037011 rad，但脑在当前完成反馈后输出 HOLD，未达原 ±0.45 rad 目标且未继续挥手，完整请求未通过。最终代码普通 END 尚未提交实测；FIFO 修复前 `073307-3cbd114c` 曾实际通过，不能跨版本冒充本版验收。采集已可运行，两场分别因保留的运行/录像期限结束，host 原 failed/cancelled 状态保留，停车与资源清理均确认；当前 GPU/脑已关闭，桌面保留停止页。B 原离线资格仍 FAIL，不称正式部署合格。下一直接问题仅为左右转任务的实际完成反馈与脑 HOLD 输出，以及最终 END 验收缺口；不自行重训或扩展动作。本轮证据、软件检查及未通过项以根 [GOAL_STATUS.md](../../GOAL_STATUS.md) 顶部为准，下面旧记录不构成自动继续作业的指令。所有修改未提交。

## 历史状态（不覆盖以上最新方向）

- 文档已精简：全部旧 Spec 与决策汇总到根 `history.md`，`specs/` 已删除；重复手册按主题合并。以后每任务最多一份 Markdown，原地更新，只留有用信息，规则见 [AGENTS](../../AGENTS.md)。
- CPU 仿真容器/专用镜像已删除，GPU 部署及运行目录已整理；目录与文档整理随本次本地提交保存；此前阶段快照为 `414d3f5`，没有 push。原始数据、模型、日志、sources 和官方软件保留。
- **用户已重新授权按钮排障、完整脑短训练及仿真点击验收。** 当前任务与验收证据统一记录在 [X2 桌面互动](../../docs/tasks/x2-interaction.md)，一次真实点击前进已取得正向4.079cm、停稳及反馈回脑证据，但整体会话超时，模型多行为与同步回放仍未通过；历史阶段见 [history.md](../../history.md#spec-024)。

## 运行入口

- 桌面：`cd apps/desktop && npm start`，仅打开界面。当前默认后端为 `malecns_lab.runtime.x2.behavior`；使用说明见 [桌面 README](../../apps/desktop/README.md)。
- 本地 `data/x2_behavior_desktop.json` 当前选择 `data/x2_model_redesign_20261024_r1/b_current_history_v1`，仅 qualification 调试候选；原0149和所有训练资产保留，不改旧 checkpoint 或来源身份。
- GPU 安装在 `data/x2_sim/gpu_probe/runtime-rebuild`，容器 `malecns-x2-gpu-trial`，端口 6081；维护见 [部署 README](../../deploy/x2/README.md)。不用仿真时关闭，不动 Noria、默认 Docker context 或其他服务。
- 当前仿真、私有VM、脑与桌面均关闭。theta-only短训练328.84秒完成48帧但效果FAIL；后续局部导数诊断53.879秒完成6帧，基线精确，小扰动已跨放电事件分支，仍不能分离代理估计偏差。不追加epsilon/同方案训练，不部署。
- 桌面隐藏时不刷新/旧snapshot覆盖状态及紧停被普通pending禁用的问题已修复、独立审查PASS，最新实际准备状态正常。`desktop-20261023-201434-aca12633`首次前进4.112cm并反馈回脑，第二请求方向失败，整场仍FAIL。52脑批与171张新v2采样已完整封存且独立档案校验通过；CUA未能切换下拉框，新联合回放UI仍待验证。默认0149已恢复，当前更改未提交。
- 已查明旧资格只测强度1/0.2m/s，当前模型为.5/0.1m/s。低速probe显式强度及有界重复已实施、4项CPU检查和独立审查PASS；低速2秒连续两段仍次段失败。官方本地文档说明静止起步有门槛，示例非零forward最小.2，不能线性缩至.1当同等资格。
- 固定`skills-20261023-205604-c569d60d`同一SIM两次(.2,0,0)×1秒完成，净正向40.677/39.722cm，原方向与停稳规则及独立资产审计PASS，仅为技能资格。固定档位已接入新SIM候选`data/x2_behavior_sim_execution_20261024_r1`（78138c/574ede），6项CPU检查、独立审查、实际完整脑严格加载均通过；参数原样复制、0训练，保留原Decision .5并另记固定速度。旧档案/默认0149保持，不重复技能probe。
- `desktop-20261023-212914-8aa7551a`唯一SHOW_FORWARD被脑选TURN_RIGHT/START，未触发新前进档位；实际右转后回摆、净角不足FAIL，停车与资源清理确认。只读对照确认参数/数值初态相同，首帧standing从旧True变为新False，训练未覆盖新组合；缺稳定窗口内部历史，尚不能断言唯一根因。唯一任务MD已登记Ready的2次前向standing单因素对照，180秒/1GiB限额，基线精确复现才解释反事实，不训练或启VM。
- standing对照r1因JSON list/tuple比较误失败，0帧；r2限定修复后CPU封存/绑定及新独立审查PASS，但启动worker前被内存保护拦下，仍0帧。后验available2.704GiB，需至少3GiB满足2GiB脑预算+1GiB余量。未降低阈值或关闭其他服务；r1/r2原样保留（r2已有execute.claim，不直接重跑）。下一次内存满足后建新目录同协议尝试；反事实结果尚不存在。脑、VM和桌面均关闭，默认0149不变。
- 随后目标轮起始内存短暂3.072GiB，r3仅换输出目录封存，但启动仍被同一内存预检拒绝，0帧、无worker；后验2.592GiB。第三个连续目标轮实测2.857GiB，留1GiB后不足2GiB脑预算，且本项目runtime/learning/私有VM进程为空，目标按规则标记blocked，尚未完成。r3已有execute.claim及preflight-failure.json，不能原地重跑。须系统释放内存、稳定达到原预算后另建同协议新尝试；不反复创建空运行、不降保护、不停其他服务。用户恢复blocked目标时重新计算阻断轮数。
- 真机暂缺，之后上海接入。旧工程检查与纵向实测不代表当前多行为按钮可用，新 GPU Dockerfile 未重建验证。
