# M2 第五、六轮准备超时：只读诊断与下一次准备登记

日期：2026-10-23。范围仅为已有日志、当前宿主入口与随 SDK 提供的官方文档。未启动脑、GPU VM、容器、Electron 或设备，未改业务源码、训练来源、官方文件或历史 run，未运行测试。本文为 S0 文档，验证仅检查本文 diff。

## 结论

第五、六轮的直接失败是 **180 秒 UI 准备截止前没有被宿主接受的 continue**。两轮都已经拿到 SD 成功回应，MC 也记录了进入 STAND_DEFAULT；不能归因为 MC 服务始终未就绪。第六轮唯一明确观察到的复位成功状态是“Pause → Reset 后机器人直立且显示 PAUSE”，之后没有在期限内完成运行与交接。

当前证据不足以判定直接 Reset/Run 是坐标输入未生效、复位后再次倒地，还是其他同步问题。因此没有已证实的物理或 SDK 修复。下一次只允许一次预先固定的 **Pause → Reset → 确认直立且暂停 → Run → 确认运行 → 有截止保护的 continue** 准备事务；不从直接 Reset/Run 失败后再临时切换流程，不延长预算，不重复点击至“碰巧成功”。该流程待实际核验，不提前称为稳定修复。

## 三轮可核对的事实

根目录均为 `data/x2_sim/agibot/data/var/`。运行编号是目录名，不用其命名时间代替日志时刻。

| 证据 | 第四轮 `skills-20261023-151800-93a1e497` | 第五轮 `skills-20261023-152628-a91d265b` | 第六轮 `skills-20261023-153231-c295ed80` |
|---|---|---|---|
| host 开始至 awaiting_simulation | 9.487 秒 | 9.235 秒 | 9.485 秒 |
| SD | 第一次失败、第二次成功 | 第一次失败、第二次成功 | 第一次失败、第二次成功 |
| MC 进入 STAND_DEFAULT 完成日志 | 06:55:00.671670 | 07:26:37.760648 | 07:32:41.283945 |
| awaiting_simulation 至下一 lifecycle 事件 | 121.044 秒，开始 HOLD | 180.006 秒，开始清理 | 180.003 秒，开始清理 |
| continue | 有，被接受 | 有，但迟到，未被接受 | 无 |
| probe | HOLD、ADVANCE 成功；一秒左转失败 | 空 | 空 |
| 总耗时 / VM 耗时 | 160.431 / 160.104 秒 | 190.261 / 190.017 秒 | 190.443 / 190.280 秒 |
| 清理 | resources_stopped=true、errors=[] | 同左 | 同左 |

时长来自同一宿主 `host_monotonic_ns` 与该 run 的预算记录；MC 时间是日志原值，不能直接与宿主 monotonic 相减。三轮默认 Docker context 均未改变。第五、六轮 `host_source.py` 逐字节相同；三轮 `official_assets.json` 文件 SHA-256 都是 `dc9e5d01d99a6a6ce747bb005e47d601d485cff91d14c5c5b759e7c980e07671`，所记录的官方资产身份没有在两次失败间改变。

### SD 与实际站姿是不同证据

三轮 `sd-01.json` 的进程返回码都是 0，但 stderr 明确输出 `Failed (code=0)`；现有 `wait_for_sd()` 同时检查成功文字，因此没有误将第一次当成功。`sd-02.json` 才输出 `Robot mode set successfully`，随后才产生 ready.json。

三轮 MC 都记录 `PASSIVE_DEFAULT → JOINT_DEFAULT → STAND_DEFAULT` 完成，但同一启动区间仍有 `Detect Falling Action`。例如第六轮最后一条这类记录在 07:32:43.256313，早于 180 秒 UI 准备结束。它证明启动早期曾检测到倾倒，不证明后来每次 Reset/Run 的效果；也不能用“后面没有 falling 日志”推定已经站稳。

第五、六轮没有技能 probe 的 odom/状态归档；Sim 日志中的 `Subscribe ... Joint Position` 是订阅到的关节命令记录，不能当作基座实际位置或站姿。没有逐次 GUI 点击事件时间、逐帧画面或准备阶段连续 odom，无法从这些日志恢复倒地发生在具体哪次点击之后。

### 末尾异常不支持“启动时崩溃导致超时”

第五轮 Sim 末尾确有 Rosetta allocation error，第六轮末尾有 `Aborted / terminate called without an active exception`，均保留。两者前面已有 `X connection to :99 broken (explicit kill or server shutdown)`；最近带时间的日志分别在约 07:29:38 和 07:35:41，与各自 SD 成功后约 180 秒、宿主开始清理的区间相符。第四轮成功执行技能后清理时也出现 X 断连及无效发布警告。

这支持将它们视为清理附近的异常线索；跨时钟和缓冲日志不构成精确因果排序，不能进一步断言“只是清理错误”，也不能把它们倒推为较早的 GUI 准备失败原因。本次不据此改 Rosetta、图形栈或重启 Docker。

## 启动顺序审查

当前 [x2_skill_host.py](../../src/malecns_lab/experiments/x2_skill_host.py) 的 `start_simulator()` 先发送 Sim 的 `docker exec -d`，紧接着发送 MC 的 `docker exec -d`，再等待 SD。`-d` 返回仅说明后台命令已派发，没有等待 Sim 窗口、模型或物理状态就绪。`simulator_ready=true` 在这里表示进入人工准备阶段，绝不表示直立、已经 Run 或技能可执行。

随本地 SDK 提供的官方 `data/x2_sim/packages/aimdk/docs/cn/_build/html/zh-cn/dev/sim_contents.html` 第 999–1010 行要求先启动 MuJoCo，窗口显示机器人后再启动 MC。当前实现只保证派发顺序，缺少该就绪间隔。实际日志如下：

| 轮次 | MC startup completed | Sim initial qpos setting complete | MC 提前 |
|---|---|---|---|
| 第四轮 | 06:54:54.743049 | 06:54:58.644260 | 3.901 秒 |
| 第五轮 | 07:26:31.853623 | 07:26:35.881881 | 4.028 秒 |
| 第六轮 | 07:32:35.248559 | 07:32:39.445847 | 4.197 秒 |

这是可证实的启动流程差异，**不是已证实的倒地根因**：成功执行技能的第四轮也存在同样顺序。现有第一轮 SD 早于完整初始化而失败，第二轮有界重试成功，因此没有必要增加 SD 重试或额外睡眠。

`deploy/x2/start-component.sh` 正确使用官方模拟配置并 unset `AGIBOT_SOFTWARE_HOME`；Sim 的 `start_sim.sh -s` 中 `-s` 意为清理型号/配置路径缓存，**不是暂停启动**。本地官方指南未给出“Pause/Reset/Run 自动保证站稳”的接口契约，不能宣称有未经核实的无界面复位 API。

如后续准备准入仍失败，可单独登记宿主流程修复：Sim 模型/窗口就绪门禁后才派发 MC，并归档实际准备阶段的模式、odom 与 UI 阶段。该修改应由实施者另做定向验证、独立审查后再运行；本次没有实施，也不把它作为本轮诊断已经解决的问题。

## 下一次最小准备流程（待执行）

保持既有两秒技能登记：TURN_LEFT → 成功才 TURN_RIGHT → 成功才 WAVE_RIGHT；不复跑已通过的 HOLD/ADVANCE。UI 180 秒、总 360 秒、清理预留 30 秒、单 probe 40 秒和下一项至少 41 秒余量均不变。速度、方向、最终净角和停止门槛不变。

1. **资源关闭时完成所有准备。** 等当前唯一读出训练退出；不与完整脑共驻留。确认要用的浏览器 tab 仍存在、已有 CUA API 文档、markHandoff，并准备好新 run 路径与 continue 写入步骤。标签或 API 不可用就不启动 VM。不要沿用旧截图坐标。
2. **启动后一次重连并取得新画面。** 以 ready.json 的真实 `ui_deadline_monotonic` 为准；从新截图识别 Pause、Reset、Run 及当前缩放。第六轮已证明，启动前 1280×720 不保证重连后仍是该视口。准备期间不查接口文档、不重建标签、不展开日志研究。
3. **首先暂停，再复位。** 使用所见明确的 Pause 控件，随后 Reset；同一已识别布局下可在一个 CUA 调用顺序完成两次点击并取截图，减少工具往返。必须从返回画面看到机器人直立且 PAUSE，再进入下一步；点击完成本身不算成功。不再先试直接 Reset/Run。
4. **运行并确认。** 在刚确认的同一布局点击 Run，取得新画面确认退出 PAUSE 且机器人直立。若仍暂停、倒地、断连或控件歧义，本轮准备失败，不补发 continue，也不在同一 run 改成另一套复位方案。
5. **立即交接，避免迟到信号。** 仅在上一步已观察成立后读取宿主当前 monotonic、ready 截止和 desktop_session；要求仍为 awaiting_simulation、无 host_summary、距离 UI 截止至少 10 秒，才原子写入新的 `{"command":"continue"}`。不向旧目录补写、不修改 ready、不删除失败证据。然后确认 lifecycle 已进入 running；真正的技能成功仍由原执行器的实际反馈判断，不由 GUI 判断替代。
6. **有界失败即退出。** 如果准备窗口剩余不足 30 秒而尚未看到“直立且暂停”，放弃本轮动作；调用既有宿主停止通道或让既有超时清理执行，不再 Run 或写 continue。任何准备失败仅归档失败，不在同一工作段立即再启动一轮。结束须沿用原资源关闭确认，其他服务不动。

与第六轮的差别是：Pause→Reset 从开始就是唯一流程；观察门槛发生在 Run 前；已知几何、继续信号和剩余预算都作为交接条件；没有“直接复位失败后临时救援”的多轮往返。以上是一次具有判定点的实验流程，不是复现可靠性已经通过的结论。

下一次至少记录三个准备事实及相应宿主时刻：首次连接后的实际视口、Pause/Reset 后直立且 PAUSE、Run 后退出 PAUSE 且直立。GUI 图像可使用 CUA 返回的实际截图；如现有工具支持保存，则以该新 run 内文件归档，不能用旧截图或事后补造时间。宿主 lifecycle/ready/continue、原 Sim/MC 日志和全部 probe 原始反馈继续保留。若再次出现“Run 后倒地”，优先补齐对应区间的真实姿态与时序证据，再决定上面的启动门禁改动；不改变转向验收标准。

## 本次交付与限制

只新增本报告，未执行下一次准备。当前 M2 能力结论仍是 HOLD/ADVANCE 单项通过、一秒左转失败、两秒转向与右转/挥手未执行。本文不提供真机资格、不证明挥手可取消，也不改变 M3 正在运行的训练身份或预算。
