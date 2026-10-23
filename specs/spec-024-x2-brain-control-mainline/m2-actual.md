# M2 实际技能运行记录

## 首次启动：界面准备超时

目录：`data/x2_sim/agibot/data/var/skills-20261023-140845-6f93be12`。源码/官方资源身份与日志保存在该目录。

主机成功启动GPU Sim/MC并进入await_ui_reset_run；协调者首次使用浏览器时遗漏了通用Tab坐标操作文档，重新读取接口文档期间超过80秒UI预算，未写continue、未启动任何技能probe。画面当时机器人倒地，尚未执行Reset/Run；不能把这张初始化画面算作站立或技能失败。

`host_summary.json`：state=failed，reason=`TimeoutError:Official UI preparation timed out`，probes=[]，wall_seconds=92.166，vm_seconds=91.950，resources_stopped=true，errors=[]；默认Docker context前后均desktop-linux。

## 第二次启动登记

原因是已定位的UI准备超时，现已取得并读取CUA Tab.click/getAXStateAndScreenshot接口，可直接操作可见官方Reset/Run。仅重复启动准备和原计划首次技能探针，不改动作阈值、源码、预算或成功定义，不覆盖首轮记录。仍是HOLD、ADVANCE、TURN_LEFT、TURN_RIGHT、WAVE_RIGHT顺序，任一实际probe失败停止后续；240秒总预算、80秒UI等待和30秒清理保留。挥手只自然完成probe，取消资格不变。新目录和实际结果运行后补记。

第二次目录：`data/x2_sim/agibot/data/var/skills-20261023-141343-a8172c4d`。已通过可见界面的Reset/Run复位并运行；画面恢复直立后写continue，进入第一个HOLD probe。ROS进程0.153秒后报`TypeError: dict() got multiple values for keyword argument 'monotonic'`；没有skill事件或接受记录。原因是运行日志封装遇到实际payload同名时刻字段，不能称HOLD动作能力失败或通过，其他四项未运行。

主机wall_seconds=47.597，vm_seconds=47.439，resources_stopped=true，errors=[]，默认context前后desktop-linux。日志与源快照保留；新实施者仅修记录器同名字段，针对真实payload做最小检查，再由新验收者审查后决定下一次运行，不自动放宽动作阈值。

## 第三次启动登记

已按[m2-runtime-fix.md](m2-runtime-fix.md)把记录接收时刻与源monotonic分开，真实tick载荷CPU回归1项通过；[新独立验收PASS](m2-record-acceptance.md)。第三次仅验证此具体运行错误修复后原定五技能probe，保持原顺序、阈值和240秒预算，任一probe失败即停止后续。首次完整脑探针已退出，实际运行不与完整脑共驻留；旧两轮失败目录保留。

第三次目录：`data/x2_sim/agibot/data/var/skills-20261023-143217-57b4dc21`。再次在80秒官方UI准备等待内超时，尚未写continue，probes=[]。本轮未检验记录器修复后的实际动作；不得算技能失败或通过。wall_seconds=91.940，vm_seconds=91.773，resources_stopped=true，errors=[]，默认context前后desktop-linux。原始host日志与summary保留。

## 第四次准备登记

第三次说明CUA界面观察、操作与协调者工具往返不能可靠落入80秒准备窗口。下一轮单独使用180秒UI准备、360秒总墙钟、30秒清理保留；仅调整操作准备预算，不改变动作参数、每技能40秒上限、下个技能至少需41秒余量、成功/停稳定义或任一probe失败即停止规则。预算必须由显式参数登记进run；默认旧入口预算保持80/240。新实施者限定修改host预算与最小CPU检查，新的独立验收通过后才启动。浏览器接口文档在启动前读取，不让运行资源等待接口学习。

## 第四次实际技能结果

预算修复已[独立PASS](m2-ui-budget-acceptance.md)。目录`data/x2_sim/agibot/data/var/skills-20261023-151800-93a1e497`，实际启动命令为：

```sh
PYTHONPATH=src .venv-train/bin/python -u -m malecns_lab.experiments.x2_skill_host --execute --sim-only --wave-probe --skill HOLD ADVANCE TURN_LEFT TURN_RIGHT WAVE_RIGHT --ui-timeout 180 --wall-timeout 360 --output data/x2_sim/agibot/data/var/skills-20261023-151800-93a1e497
```

官方UI实际Reset/Run后画面恢复直立，才写continue。UI准备约121秒，在本轮180秒登记内；并非动作超时。各probe以实际MC/里程计与停止反馈判定：

| 技能 | 结果 | 实测 |
|---|---|---|
| HOLD | 通过 | 观察到站立并确认，probe 1.360秒 |
| ADVANCE | 通过 | 沿初始朝向前进0.442807883米，横向−0.008368964米，最终水平停稳，probe 10.813秒 |
| TURN_LEFT | 失败 | 最终相对yaw −0.047328933rad，未满足左转方向；记录过运动并最终停稳，probe 10.081秒 |
| TURN_RIGHT、WAVE_RIGHT | 未运行 | 遵守前项失败即结束，不把未运行记作通过 |

主机总160.431秒、VM160.104秒，resources_stopped=true、errors=[]，Docker默认context前后desktop-linux。保留实际完整源/事件/命令/状态及官方资产哈希。这是无脑的执行能力核验；HOLD/ADVANCE单项成功不代表P2整体或脑控产品通过。左转需先只读核对实际发布命令、符号、反馈与停止区间，禁止仅因负角反转符号或放宽成功阈值。挥手取消仍未验证。

## 第五次登记：已有上限内的两秒转向

[只读诊断](m2-turn-diagnosis.md)未发现符号/TTL/里程计解析错误：左转50次+0.2rad/s发布调用成功，非零脉冲1.006秒；站立切入步态初期存在负yaw瞬态，前进无yaw目标时也有类似瞬态。下一次仅检验延长至核心既有2秒上限后能否建立实际正确方向，不能把这个解释提前写成已证实原因。

新实施者只透传probe CLI `--command-seconds 2`，默认保持1、仅允许1/2、全程入档，速度/方向/停止/成功阈值不变，独立审查后才运行。一个新run依次请求TURN_LEFT、TURN_RIGHT、WAVE_RIGHT：左转固定+0.2rad/s两秒，通过后首次请求右转−0.2rad/s两秒，之后首次自然完成挥手。任一项失败即停；不重跑已经通过的HOLD/ADVANCE。wave仍使用原官方1002/area2自然轨迹，连续速度脉冲参数不改变其动作或取消资格。

UI180秒、总360秒、清理30秒、单probe40秒及下项41秒余量保持。左转最终净角必须>0.02rad、右转必须<−0.02rad，并满足原角速率/漂移/站立停稳标准；第二秒角增量只用于解释，不替代最终结果。禁止继续加时、改符号、加速或临时放宽阈值。结束关闭全部本次资源，不与完整脑共驻留。

第五次目录`data/x2_sim/agibot/data/var/skills-20261023-152628-a91d265b`：持续时长透传已独立PASS，但此前浏览器tab6已失效，清点/重建tab7后才操作Reset/Run；continue写入晚于180秒UI截止，主机已经退出，probes=[]，resources_stopped=true。因此两秒转向尚未实测，右转/挥手仍未运行。失效标签的恢复开销没有被预先排除，不能将此次记成动作失败，也不删除迟到的continue文件来修饰记录。

第六次仅重新执行尚未开始的第五次动作登记，预算与所有条件不变。先在仿真关闭时确认并markHandoff现有tab7（当前1280×720视口），再启动私有runtime；启动后只重连、检查当前画面和Reset/Run，避免运行资源等待标签发现或创建。不复跑HOLD/ADVANCE，不额外增加动作时长。

第六次目录`data/x2_sim/agibot/data/var/skills-20261023-153231-c295ed80`，仍未进入probe。实际重连后视口变为1284×1344，因此按新截图坐标操作；直接Reset/Run后的画面仍倒地，不能据点击推定复位成功。随后Pause→Reset明确观察到暂停且直立，但再Run前后已到180秒准备截止，未写continue。不能断言之前是输入未生效或复位后快速倒地，当前证据只确认暂停复位状态。probes=[]、总190.443秒、VM190.280秒、resources_stopped=true、errors=[]、默认context未变。

停止相同启动循环；180秒内的浏览器往返与复位流程仍不可靠，需要先解决准备流程，再运行原登记两秒动作。不因这两轮未执行而改方向、速度或成功标准，也不再临时加UI时长。当前转向能力结论仍只有第四轮一秒左转FAIL，右转/挥手未测试。先推进已独立准入且不依赖仿真的读出基线。
# 第七轮：两秒双向转向通过，挥手未取得 ACK

2026-10-23，原run：`data/x2_sim/agibot/data/var/skills-20261023-170921-07ed4474`。等待唯一读出训练监督进程退出后才启动，无完整脑共驻留。使用已登记的 `TURN_LEFT TURN_RIGHT WAVE_RIGHT --command-seconds 2 --ui-timeout 180 --wall-timeout 360 --wave-probe`；没有重跑HOLD/ADVANCE，也未变更动作、方向或停稳标准。

| 技能 | 实际结果 | 原始反馈 |
|---|---|---|
| TURN_LEFT | PASS | 两秒正角速度目标，最终净角 `+0.07166989080758994` rad；实际运动与持续停稳均确认，9.542698213秒 |
| TURN_RIGHT | PASS | 两秒负角速度目标，最终净角 `−0.05984628296939161` rad；实际运动与持续停稳均确认，9.582330172秒 |
| WAVE_RIGHT | FAIL / UNKNOWN | `preset_ack_timeout`，无task_id、无查询RUNNING/播放器PLAYING证据；关节最大变化仅0.000974647rad，未达到实际挥手阈值。未确认取消/恢复，14.487010382秒 |

左右转向的完整请求、接受、实际运动、零目标及停稳事实保存在各自 `probe-00-turn_left`、`probe-01-turn_right` 的原始事件与summary。两次均是隔离技能探针，不是脑已学会方向任务，也不证明达到±0.45rad目标。左转开始阶段仍曾出现负角瞬态，但最终方向达到原标准。原一秒失败保留，不被覆盖。

本次CUA连接后观察到机器人倒地；依次点击Pause与Reset后画面直立，但Run仍选中，未观察到PAUSE。再次读图仍直立运行，随后额外点击一次Pause，明确看到直立与PAUSE，再点击Run并确认退出PAUSE且直立。**这比预登记单次准备事务多一次Pause，不把它写成原流程已可靠复现；未再次Reset，也没有改变技能标准。** 在同一`.venv-train`宿主时钟中检查剩余UI时间51.688899秒、仍为awaiting_simulation、无summary后原子写continue；宿主确认进入技能核验。CUA图像仅在会话中实际观察，本run没有保存独立视频/截图文件。

注意准备时读取普通`python3`得到的monotonic约0.020秒，与宿主运行时的349929秒显然不属可直接比较的时钟；没有使用该值做交接判断。最终截止保护使用与宿主相同的`.venv-train/bin/python`，实际now为350057.901125791，ready截止为350109.590024666。各原始时钟保留，不伪造GUI与MC时间对齐。

整轮179.787630208秒、VM179.539751167秒，未延长原预算；挥手失败后结束。`host_summary.json`为failed、resources_stopped=true、errors=[]、前后Docker context均为desktop-linux。随后`lighter status`明确not running，默认context仍desktop-linux。无GPU仿真、完整脑或旧CPU仿真继续运行，Noria未操作。

当前P2更新为HOLD、ADVANCE、TURN_LEFT、TURN_RIGHT各有单次实际通过证据；WAVE_RIGHT自然完成与取消均未通过。下一步先只读诊断本次ACK超时的请求/服务/模式证据，不直接重复启动或把未知状态改称成功。
