# X2 最小动作适配器实施与实测

日期：2026-10-23。风险 S2；最终独立结论见 [acceptance.md](acceptance.md)。

## 实际交付

新增 `control/x2.py`、`experiments/x2.py` 和 `tests/test_x2_motion.py`。纯状态机只允许 STOP / 固定 MC 目标 `(0.2,0,0)`；默认不连接、不发布。显式仿真入口在 domain47/localhost1 下注册 `malecns.x2_mvp`，按约50Hz发送，结合新鲜里程计、MC、姿态、控制权与有效期约束动作。支持注册后空源的有界 CLAIMING 和官方起步 TRANSITION；均限于首次请求起0.5秒，计入单次5秒上限，不随续期延长。

失效后清除意图并持续归零，恢复不自动重走；确认停车要求停止/最近窗口重置之后两路都推进的新反馈、STAND/RUNNING、自有输入源、直立、水平速度<0.02m/s持续至少1秒，并核对位置变化。发送、反馈和日志错误保留失败，不以退出或零变量代替停稳。既有 BodyVelocity 契约及官方 SDK、MC、Sim、模型和配置未改。

## 验证与修正

仅运行本子系统检查。最终 `.venv/bin/python -m pytest -q tests/test_x2_motion.py` 为 **106 passed in 0.13s**，通过后未重复。默认无ROS、异常/瞬时故障、TTL、控制权、5秒上限、两路反馈屏障、日志失败退出均有定向检查。详细实施与有限返工见本目录 `implementation-code.md`、`implementation-bootstrap.md`、`implementation-transition.md`、`implementation-stop-window.md`；未运行全仓测试、build、lint或typecheck。

真实集成保留全部失败：

1. 最初启动覆盖 PYTHONPATH，导入 rclpy 前失败，无动作；后改为 source ROS 后 prepend 源码路径。
2. 第一轮注册成功、1033次零发布，209条MC仍为空源，无非零动作，按超时失败。随包官方仲裁文档确认零不切源，修正有界CLAIMING。
3. 第二轮遇到官方 STAND/TRANSITION=200 后过早归零，约0.10秒非零、总位移0.0456m；虽确认停稳，仍明确为动作失败。按消息定义增加有界起步过渡。
4. 第三轮跑完2秒并回报成功；原始分析发现旧MC可回填停车窗口起点。该轮不作为AC3通过；增加重置时间及双路序号屏障，位置异常立即返回。
5. 第四轮使用修正源码完成下述实测。没有因主观信心重复启动。

## 最终第四轮

源码与原始证据：`data/x2_sim/agibot/data/var/x2-adapter-20261023-stop-window/{src,evidence,sim.log,mc.log}`。命令、源码散列、分析和关闭记录：`logs/x2_docker/20261023_motion_adapter/` 下 `runner-stop-window.log`、`source-manifest-stop-window.json`、`analyze.py`、`analysis.json`、`shutdown-stop-window.txt`。旧轮分别保留无后缀、`-retry`、`-transition` 目录与独立快照。

在官方SD及GUI Pause→Reset→Run恢复站立后执行：

```bash
docker --host unix:///Users/yang/project/malecns_lab/data/x2_sim/gpu_probe/runtime-rebuild/docker.sock exec malecns-x2-gpu-trial bash -lc 'source /opt/ros/humble/setup.bash && source /workspace/aimdk/install/local_setup.bash && PYTHONPATH=/agibot/data/var/x2-adapter-20261023-stop-window/src:$PYTHONPATH ROS_DOMAIN_ID=47 ROS_LOCALHOST_ONLY=1 python3 -m malecns_lab.experiments.x2 --execute --sim-only --forward-seconds 2 --startup-timeout 12 --stop-timeout 12 --output /agibot/data/var/x2-adapter-20261023-stop-window/evidence'
```

退出0；`success/forward_accepted/forward_started/stop_confirmed` 均true，failures为空。状态为 STOPPING→CLAIMING→FORWARD→STOPPING→STOPPED。

| 实测 | 结果 |
|---|---|
| 非零目标 | 99次，首个非零至首零1.99346秒 |
| 发布间隔 | 中位20.126ms，最大21.748ms，样本354个 |
| 整轮初始身体前向净位移 | 0.90069m |
| 非零命令期间前向位移 | 0.66811m，命令边界用相邻接收里程计线性插值 |
| 首零至最终位置水平净位移 | 0.23401m |
| 姿态 | 最大倾角4.357°，最低骨盆z=0.64276m |
| 首零至确认停车 | 5.09292秒 |
| 最终连续新样本窗口 | 1.00677秒，249条odom、11条MC |
| 窗口水平速度/漂移 | 最大0.01723m/s，净漂移0.00784m |
| MC窗口 | 全部STAND_DEFAULT/RUNNING、自有源、直立；间隔满足阈值 |

离线分析只执行一次，8项预设谓词全部true。事件文件 SHA256 `646b38190ce4464d9394132bcc348d453798172d97fd407af4aed92153a45ff5`。控制核心 SHA256 `e0e25488248eeb8bc76d4d116bdfd268f593dd2b8db093ead701a177800dfd8b`；ROS入口 `e79190ac4fdabcea2e28eb3bcd9450c7b5d97160d3becc08da0d1dfd086d5afa`。

第四轮结束后立即停止本项目GPU容器与私有VM。`shutdown-stop-window.txt`对应本轮：GPU容器exited、VM Stopped、CPU容器exited，默认context仍desktop-linux。未操作Noria、重启Docker、运行完整脑、训练、真机或push。最终Sim日志尾部仍有主动关闭后的 `Publish type unregistered`，不声称优雅退出。停车截图 `stop-window-confirmed.png` 视角较近、腿部部分出框；连续直立结论来自记录的姿态与MC，不能仅凭该截图。

## 结论边界

本阶段证明受控意图到官方X2仿真的有界前进与反馈停稳，不证明精确速度、固定刹停距离、硬实时、无人起身、完整MaleCNS控制或真机适用性。0.2是MC目标，约50Hz是本轮发送节奏，历史60FPS只是GUI渲染读数。下一阶段需要接入真实完整脑及明确的传感来源、参数身份、资源与时延协议。
