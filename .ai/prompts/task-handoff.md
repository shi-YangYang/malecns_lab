# 当前交接

更新：2026-10-26。唯一任务文档 [GOAL_STATUS.md](../../GOAL_STATUS.md)，原始资产与失败记录全部保留。当前无commit授权。

## 目标与边界

- 长期目标：真实MaleCNS选择前进≥1m、左右90°、挥手；任意连续、END后重启、长期稳定。四动作16对及60分钟/100请求仍未完成，不预称全可用。最近工具状态为paused，本轮按用户“快修”执行限定修复，未自行恢复长期Goal。
- 不重训、不改权重/输入/历史/标签、不把HOLD改WAVE、不在动作间Reset、不操作Noria/默认Docker。仿真不用时关闭私有GPU。
- 用户已明确“必须±5°，继续解决转向精度”；不得放宽评分或改旧成绩。v5/v6/v7已实际失败，不再重复或扫参数。

## 已确认的改善

- continuous v2在真实主动作COMPLETED后关闭同请求后续技能准入，继续记录原始脑输出、消费反馈与ACK后解锁。040524场GREET→FORWARD→END→GREET通过；额外原始WAVE保留、未执行。不代表脑学会单动作结束。
- busy三层保护、超时停车、首因报告、媒体正常封存后继续实时源、有限遥测等已有定向检查/只读审查，不重复绿测。软件fixture不能当物理长时证据。
- UI已左右各半宽满高度、自动message、真实群组/采样脉冲平滑显示；无MB/CX/AL表面或膜电位遥测，不伪造。

## 最新真实场次

- 旧v8/official混合054057-5f166662：前进1.226411/1.212555m后GREET失败wave_yaw_limit。第二前进COMPLETED到下一请求首观测14.307秒继续偏航13.44°，MC全STAND/player0且无非零命令；旧停止条件不代表朝向稳定。
- 隔离配置`--sim-profile noslip1`只改noslip_iterations=1，派生XML/YAML与原/派生hash留在新run，不改官方资产/默认。11项配置检查及独立只读审查PASS。此前r1–r4准备失败均0动作，不算求解器成败。
- **064146-55921afe** / `data/x2_continuity_20261026_noslip1_r5`：noslip1/v8/continuous，FORWARD→FORWARD→GREET 3/3通过；1.220029/1.203665m，25.644/25.464/28.279秒；总241.995秒。实际MJMODEL.TXT:63读回Noslip=1。第二前进后12.923秒偏航1.18°，比旧13.44°小但非零；挥手自然完成，执行peak yaw .020435rad、归臂误差.031822，stop/resources=true。单场不证明根因/全部组合。
- **064808-ec3b4935** / `data/x2_continuity_20261026_noslip1_turns_r1`：LEFT真实START，终态88.741°、crossed=true、STOPPED/base_stop=true、无motion failures，但旧执行器>=90°判FAILED。RIGHT/GREET未运行，总197.161秒，stop/resources=true。

## 本轮修复与最新结果

- v9 `quarter-turn + balance-sway`统一真实跨目标且停稳后的±5°终态判据，旧v4/v8/default不变；20项基础及2项stale-cross保护case通过，独立只读复审PASS。
- 070504-851b6d02：LEFT88.6225° PASS，RIGHT95.6046° FAIL；右收到跨目标反馈7.772ms才发零。限定v9 callback pump提前结束，保留原tick MC/odom/TTL保护、50Hz及全部动作参数；新增14case通过、只读审查PASS，不重复旧22case。
- **072823-21ef494f** / `data/x2_continuity_20261026_noslip1_v9_pump_r1`：同noslip1/v9 LEFT→RIGHT→GREET **3/3通过**，88.942250°/94.531215°，请求26.360/25.605/30.422秒，总404.401秒含准备；cross→zero .494/.554ms。挥手自然完成，peak yaw .01568949rad、归臂.0322153rad。无中途Reset，stop/resources=true。证明延迟缩短，单场不能把所有角度改善归因该变更。
- **073800-a004ef83** / `data/x2_continuity_20261026_matrix_v9_r1`：默认19请求矩阵首失败退出，共277.382秒。前进1.214021/1.208438m通过；第三LEFT脑d9 START、请求正常接纳，终**100.635176° FAILED**，其余16未运行；本场仅1/16对通过，stop/resources=true/runtime_alive=false。
- 本次LEFT cross90.477762°→zero仅.716ms，wz1.337432；零后LOC20097.5199°、STAND98.6486°、终100.6352°，5.365秒又+10.1574°。成功LEFT零后−1.1947°。二次前进完成→LEFT接纳17.463秒漂1.3085°但新角基线已锁定，未直接计入新转角。初态和发零时平移速不同，不能猜内部根因。排除本次为回调迟发零/按钮忽略；不重复候选、不扫参数/补角、不放宽评分、不先长测。
- 官方locomotion.html:640–645明确速度级开环、不保证终点精度；STAND无动态制动保证/目标yaw字段，公共导航无已验证的本机相对yaw入口，不凭接口存在追加试跑。
- v10有界积分候选（P2/D.25/I1、最大±.8、实际非零10秒、总34秒、一次停稳纠偏）25项新定向case均有通过记录、独立只读审查PASS，旧检查不重复。旧profile/default、脑和±5°评分保留。
- **090516-fb03c490** / `data/x2_continuity_20261026_integral_v10_r1`：FORWARD→FORWARD→LEFT→RIGHT **4/4通过**，1.216936/1.197785m、91.763645°/90.638429°，217.628秒；stop/resources=true。I实际进入指令，首零角速约±.09rad/s、0次纠偏；尚无先停滞后脱离的证据，不能断言积分独立因果。
- **090934-4dcb9cd8** / `data/x2_continuity_20261026_matrix_v10_r1`：422.679秒首失败退出。F→F→L→F前四通过，1.236535/1.222678m、91.629391°、1.208043m；第五RIGHT脑d17 START真实执行，一次纠偏后79.164660°，`integral_turn_corrections_exhausted`，后14未运行。仅3/16对通过，不先长测、不重跑同候选。stop/resources=true、runtime_alive=false；次生`bridge_summary.json:ValueError:Transport packet exceeds budget`另行只读定位。
- v10第五RIGHT两段均约87.5°低速发零，之后分别回摆8.8470°/8.3557°，一次纠偏误差只改善.4807°；非预算耗尽，不能推断固定跨run补偿。
- 传输超限修复只去重邮箱中PID/阶段/历史数组，完整原事件先flush入JSONL、引用身份+SHA，保留终态/角度/保护/预算，256KiB上限不变。5项新检查及独立只读审查PASS，不重复绿测。
- v11显式`compensated + balance-sway`：首次真实停稳残差(5°,15°]才一次内部reference=原目标+残差，原90°评分/脑输入不改；>15°失败、实测|角|>105°锁停车，沿v10 10/22/34/12预算。23新case有pass记录及只读审查PASS。
- **093738-5a66765f** / `data/x2_continuity_20261026_matrix_v11_r1`：471.073秒、前7 F→F→L→F→R→F→GREET通过，1.210727/1.184321/1.202326/1.211936m，91.545702°/93.716738°。RIGHT确实触发一次残差补偿（首停−1.312587rad→参考−1.829006→终−1.635666rad），原±5°通过。第8GREET后LEFT脑d30 START接收后约.504秒发生`startup_transition_timeout`，最终−.069588°、无STARTED，非PID stall。后11未运行，6/16对，不先长测/重跑。
- FAILED event23实际送达邮箱，无packet-budget错误；stop/resources=true/runtime_alive=false。v11候选尚未设桌面默认。
- 启动计时修复：0.5秒从首条成功有效非零发布返回后起计，ZERO/异常不计、后续不续期；首发等待另限请求起0.5秒，claim/TTL/动作总预算不改。30项定向检查及只读审查PASS。首场`102013-b916de86`为0动作初始化失败：ready需23.617秒、host15秒先超时；仅桥初始化等待15→30秒、保留总截止/进程/guard/真实ready校验，4项检查及只读审查PASS。失败资产保留。
- **103028-ccb4411f** / `data/x2_continuity_20261026_startup_clock_r2`：同v11/noslip1/continuous，GREET→LEFT→RIGHT→GREET **4/4通过**，左88.294492°（一次实际残差补偿）、右90.483790°（无补偿），两挥手归臂.032488/.032158rad。总354.148秒，stop/resources=true、runtime_alive=false、exit0，无动作间Reset。本场启动LOC/100在首发后159/247ms、无MC200；仅回归通过，不宣称新锚点的物理因果。初始化10.868秒，新增等待余量未实际用到。全部16对/长时仍未完成；无训练、无commit、未设桌面默认，当前无本场仿真/脑进程。

## 固定入口

- `apps/desktop` npm start只开界面；B模型`data/x2_model_redesign_20261024_r1/b_current_history_v1`，配置`data/x2_behavior_desktop.json`。旧开发710/722、22/27、FINISH误运动3仍FAIL。
- 私有GPU`data/x2_sim/gpu_probe/runtime-rebuild`，容器`malecns-x2-gpu-trial`，VNC6081。运行状态以本场最终summary为准。真机未接入。
