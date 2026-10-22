# 官方 X2 动作实测 — 2026-10-23

状态：实施完成，[独立验收 PASS](acceptance.md)。仅验证官方仿真动作链路，未接 MaleCNS、未训练、未连接真机。最终 GPU 容器、私有 VM 和 CPU 仿真均已停止。

## 实际修改与检查

使用已有私有 runtime-rebuild、malecns-x2-gpu-trial 和原版 AimDK/MC/Sim/模型，没有修改官方组件、物理参数、业务代码、依赖或部署配置。实施者新增只读 ROS 采集器，协调者维护本 Spec/Plan/报告、交接与文档索引。

动作集成按 S2，只检查这条链路；只读脚本按 S1，仅一次 AST 检查通过，并在本轮实际订阅中无异常；文档按 S0 审 diff。没有全仓测试、构建、FPS 重测、commit 或 push。

## 操作与失败保留

1. 准备时关闭仿真和 VM，只在动作窗口启动。原始 PASSIVE 无力矩，机器人先倒地。
2. 官方 set_mc_action SD 返回成功；MC 自动规划 PASSIVE → JOINT → STAND，但机器人仍躺地，高度约 0.077 m，body_status=2。没有继续前进，服务成功不作为站立通过证据。
3. GUI Pause → Reset → Run，保持官方 STAND 控制器，恢复初始姿态后站稳。没有验证倒地自主起身。
4. 官方 set_mc_action LD 返回成功；没有速度输入时 MC 会回到 STAND。官方速度示例输入 0.2、0、0，约 5 秒后持续发布零速度。第一轮采集器恰在运动前达到 240 秒上限，保留 attempt1/ 和 velocity.log，该轮不作为完整运动验收证据。
5. 针对具体采集缺口补测一次：新采集器 attempt2/ 最长 300 秒，GUI 重置并启用跟随镜头；确认直立后再运行相同官方速度示例。既有输入源 ADD 拒绝后，示例自动 ENABLE 成功。未改变控制参数。18 秒 timeout 发 SIGINT 前，示例已正常发送零速度约 12.2 秒。
6. 补测完整覆盖站立、前进、归零后停止，随后关闭 GPU 容器与私有 VM，离线分析。

## 定量结果

原始数据为 attempt2/samples.jsonl，计算结果见 logs/x2_docker/20261023_motion_mvp/analysis.json。阶段用同一容器内官方发布日志与接收 wall time 对齐；经过时间用接收 monotonic time。没有测量物理仿真时间与墙钟倍率。

| 阶段 | 实测 | 模式与姿态 |
|---|---|---|
| 运动前站立 | 9.886 s；水平漂移 0.000555 m；高度 0.653098–0.653103 m | 62 个 MC 样本全部 STAND、body_status=1；最大倾斜 3.676° |
| 非零速度发布 | 日志持续 4.999 s；轨迹覆盖 4.980 s；沿起始身体前向位移 **2.118 m**，水平位移 2.130 m | STAND 切到 LOCOMOTION；body_status 全部为 1；高度 0.632–0.676 m；最大倾斜 4.541° |
| 零速度发布 | 持续 12.200 s；轨迹覆盖 12.178 s；归零后仍有 **0.206 m** 水平位移 | LOCOMOTION 回到 STAND；body_status 全部为 1 |
| 归零末 3 秒窗口 | 143 个样本，端点跨度 2.972 s；水平漂移 **0.000347 m** | 高度 0.653341–0.653362 m；最大倾斜 3.820°；最终截图直立 |

运动与停止全段 819 个里程计样本，最大相邻间隔 0.0237 s。采集上限 odom 50 Hz、MC 10 Hz，不是物理或控制频率。两个采集器均按时限正常退出，errors 为空。截图只证明对应时刻；连续姿态结论来自轨迹和 MC 状态。

**速度跟踪尚未校准。** 0.2 m/s 是发送值，按接收时间算前进阶段平均前向速度约 0.425 m/s。原版 rl/foundationwalk_config.yaml 有 min_run_velx: 0.5、run_scale: 0.9，但执行语义和物理时钟尚未验证，不能直接归因。接入 MaleCNS 前需要明确 MC 速度映射、时钟和停车余量，本轮不调官方参数。

## 证据与复现

以下路径相对项目根目录；运行资产仍位于忽略目录：

- logs/x2_docker/20261023_motion_mvp/：vm-start.log、set-sd.log、set-ld.log、analysis.json、shutdown.txt、observe.py、observer-implementation.md，以及 stand-from-floor.png、stand-after-reset.png、recorded-motion.png、recorded-stop.png。
- data/x2_sim/agibot/data/var/x2-motion-20261023/：sim.log、mc.log、velocity.log、velocity2.log、observer.log、observer2.log、attempt1/ 和 attempt2/ 中的 samples.jsonl、latest.json、summary.json、phase.txt。
- forward.png 的机器人已出窄视野，不作为直立证据；recorded-motion.png 实际拍在停止附近，不宣称抓到了非零速度阶段。

Docker CLI 显式使用私有 --host unix:///Users/yang/project/malecns_lab/data/x2_sim/gpu_probe/runtime-rebuild/docker.sock。官方包挂载为 /workspace，项目 agibot 挂载为 /agibot。使用已有 x2-component 启动器，Sim 的 VirtualGL 参数 -d egl0 -ms 0，环境 DRI_NO_MSAA=1、VGL_SAMPLES=0、LIBGL_KOPPER_DISABLE=true，组件选择均输入 0。既有 ROS domain 47、localhost 隔离。容器 bash -lc 中执行：

```sh
timeout 15s ros2 run py_examples set_mc_action SD
timeout 15s ros2 run py_examples set_mc_action LD
# 两次速度示例相同；第二次日志为 velocity2.log
printf '0.2\n0\n0\n' | timeout --signal=INT 18s ros2 run py_examples mc_locomotion_velocity
```

停止调用确认 GPU container exited，私有 lighter stop 返回 Stopped.，CPU container exited，默认 context desktop-linux。未操作 Noria、其他服务或默认 context。Sim 在本次主动停止窗口 18:18:44 UTC 记录 Publish type unregistered、terminate、Aborted，因此不宣称优雅退出；此前运动停止成功不等于进程清理无错误。本轮未修复或重跑官方清理路径。

## 结论边界

实施证据满足本 Spec 的站立、前进、最终稳定停止门槛，独立验收已确认。未验证自主起身、精确速度跟踪、转弯、长期稳定性、MaleCNS 闭环或真机。速度差异和退出错误保留为后续问题，不把阶段成功推广为完整系统完成。
