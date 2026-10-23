# 正式接入实测

2026-10-23。风险 S3，受影响子系统定向验证；不代表训练效果或真机表现。

## 完整脑遥测开关配对

首轮独立审查确认旁路与保护无阻断离线 probe 的问题后，串行执行两次受监督完整图推理；VM和旧CPU仿真均关闭，未运行训练。输入/原baseline固定为 `data/x2_neural_inference_20261023_r1`，新结果分别为 `data/x2_telemetry_off_20261023_r1` 与 `data/x2_telemetry_on_20261023_r1`。两次均退出0、比较PASS，当前wrapper源码与runtime一致。

| 证据 | 实测 |
|---|---|
| 全部状态/输出 | 两次各自4个递归边界与3个完整输出对同一原baseline逐元素精确一致 |
| 计算墙钟（预热+3帧） | 关闭9.882742秒；开启9.899205秒；增加0.166587% |
| 峰值RSS | 关闭1650425856 bytes；开启1715896320 bytes；增量62.4375 MiB |
| 采集范围 | 22个真实参考群组；32个选定真实输出神经元 |
| 完整性 | 42/42批，丢批0，errors=[]，complete=true，435440 bytes |

满足提前冻结的≤15%计算增幅、≤128MiB额外RSS、精确等价与完整性门槛。这是一次同机顺序配对，不是重复统计性能结论。保存汇总 `logs/desktop_workbench/telemetry-pair.json`，原始全部身份、状态、输出、资源和遥测在各新run内。未重跑原baseline。

## 桌面集成

正式 Electron 空闲首屏左侧未运行、右侧真实静态胞体图，未读取旧CPU诊断活动。截图 `logs/desktop_workbench/idle.jpg`。首屏 SVG 默认填充问题和独立审查发现的实时 summary 错误/完整性展示问题均已限定修复，并经新的独立验收者复核通过。

### 成功的原生桌面运行

通过实际 Electron 界面点击准备，等待官方 SD 确认，在左侧官方 noVNC 中依次 Pause、Reset、Run，然后点击桌面“开始”。没有修改 ready、伪造观测或绕过姿态门槛。记录为 `data/x2_sim/agibot/data/var/desktop-20261023-034602-5d90cd78/`，run_id 为 `e2e5ba357a674e8895c8279ce58264e9`。

| 项目 | 实际证据 |
|---|---|
| 桌面双侧 | `logs/desktop_workbench/live-running.jpg`：真实 noVNC 站立场景与同 run 的 12 个预置脑批次；随后右侧更新至第32批、群组13.3Hz和真实脉冲 |
| 两轮反馈 | `bridge_summary.json` success=true；新观测来自前一轮动作之后；两轮 FORWARD 位移分别0.381063869m、0.446197222m |
| 逐轮停车 | 两轮均为 owned_core_STOPPED；final_stopped=true、active_forward=false、core_failures=[] |
| 冻结脑 | worker_result completed、2帧、parameters_unchanged=true；每帧计算3.646406s、3.642355s |
| 同一身份 | 桥、worker、telemetry、desktop_session 的 run_id 一致；遥测 execution_id 与 worker 一致 |
| 真实活动 | 22群组、32选定输出神经元；32/32批、1600神经步（160ms），零丢批/错误、complete=true、428612bytes |
| 脉冲事件 | 第26/27/28/31/32批分别1/1/5/3/1个真实选样事件，共11个，不是全部166700节点的逐步归档 |
| 资源与时限 | desktop_session completed、stop_confirmed=true、resources_stopped=true；宿主87.508s、VM80.332s，满足240s/120s预算 |

官方 SD attempt 1 返回码0但明确失败，未被当成成功；attempt 2 返回明确 `Robot mode set successfully` 后才放行。两次完整返回已保存于 `logs/desktop_runs/desktop-20261023-034602-5d90cd78/sd-attempt-*.json`，实际验证了限定重试路径；不能据此证明所有历史 SD 失败的根因。

汇总保存为 `logs/desktop_workbench/desktop-integration.json`；原始数值和桥证据保留在运行目录。截图仅记录所见窗口，不代表视频逐帧与遥测时间严格对齐。两轮闭环结论来自桥/里程计/停车记录，不能仅凭站立截图宣称动作成功。

### 离线回放与最终资源状态

运行结束后，从 Electron 的实验记录中选择本次成功记录，逐批切到第28/32批。实际界面显示135–140ms、DNg02_L群组53.3Hz和5个真实脉冲，与第28批原始文件一致，截图 `logs/desktop_workbench/replay.jpg`。回放不启动完整脑或仿真；左侧明确无归档视频，未拿历史图片冒充同步回放。

完成后只读检查：私有 Lighter 返回 `lighter is not running.`，旧CPU容器为 exited，默认 Docker context 仍为 desktop-linux。成功运行日志已确认自有 GPU 容器和VM清理；未操作 Noria、重启 Docker 或变更默认 context。桌面保留在离线回放，未再次启动重资源。

### 之前的失败记录

以下记录全部保留且 resources_stopped=true，未计入成功闭环。前四次是旧编排/交互阶段的具体失败，后续只针对失败重试，没有扩大数值或动作预算。

| 目录后缀（均为 desktop-20261023-） | 结果 |
|---|---|
| 024153-8e625418 | user_stop，未记录动作，stop_confirmed=false；只证明资源清理，不能称停车通过 |
| 025314-d9fe9b98 | 官方UI准备超时；曾实际看到双侧连接和12批预置活动，无动作，stop_confirmed=null |
| 030410-f331c380 | 工具交互间隔导致官方UI准备超时，无动作，stop_confirmed=null |
| 031503-c92ea250 | 官方SD未确认成功；旧实现未保存SD响应，根因未证明；随后限定增加预算内重试及返回记录并独立通过 |
| 034424-ffa085b8 | noVNC就绪/布局变化时误点了旧坐标，先开控制桥后完成复位，initial_stand_timeout；未记录动作，stop_confirmed=false |

成功运行后停止重型验证。异常输入、窗口关闭/renderer崩溃、路径隔离和资源所有权等故障场景沿用已完成的定向注入检查；没有为了重复证明这些路径再次启动完整脑或仿真。本交付仍限本机开发运行、里程计虚拟标记与两轮纵向离散控制，不包含训练、真实相机/设备、连续控制、安装包或仿真视频归档。
