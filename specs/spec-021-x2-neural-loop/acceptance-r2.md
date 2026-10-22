# Spec021 独立验收 — r2 FAIL

日期：2026-10-23。验收者独立于实施者。只读冻结代码、实施报告、原始 JSON/JSONL、资源 CSV 与环境生命周期；未重跑 40 项已通过的定向检查，未启动脑、ROS、VM 或动作。

## 结论与限定返工

**FAIL：尚未满足至少两轮完成决策与动作反馈的工程闭环验收。** 第一次神经决策驱动实际前进并停稳；第二次推理完成后，原观测与当前完整姿态差超过预登记的 0.03 rad，桥正确拒绝输出并归零结束。错误未被改写成成功。

具体返工限定为新调度层的**观测采集就绪条件**：Spec018 的 STOPPED 证明水平低速与官方站立，不保证姿态在数秒脑计算期间足够稳定。当前 `LoopController.stationary()` 在曾前进后仅依赖 core.STOPPED，`finish_round()` 随即采集下一观测，因此在仍有明显角速度时开始推理。请新实施者在采集前增加有界的姿态稳定等待，逐样本检查角速度和完整四元数变化，保留新反馈与重置边界；记录就绪证据并覆盖对应定向场景。

不得放宽响应时的 0.03 rad/0.02 m 漂移门限、重标旧观测时间、改固定神经门限/模型、绕过最终 ACK 或修改已通过的运动核心。等待超时仍应保留失败并归零。新增门槛与等待预算须先登记，然后由新独立验收者复核限定修复和必要联调。

## 原始证据与实际行为

- r1 `data/x2_sim/agibot/data/var/x2-neural-loop-20261023-r1/`：宿主命令误用无 Torch 的 `.venv/bin/python`，在 runtime 元数据阶段报 `PackageNotFoundError: torch`。worker、VM 与动作尚未启动。r2 改用既有 `.venv-train/bin/python`；未改代码、模型或门限，不需要重跑已通过的软件检查。
- r2 `data/x2_sim/agibot/data/var/x2-neural-loop-20261023-r2/`，run ID `8b1a60278e7c433da40e42217aa06a6f`。`bridge_events.jsonl` SHA256：`9cc6a95c4a63240f185a009f1fd71d3b02b60b70eaf1ea1aeb21fb001d2b12a4`。
- 初次空源观测使用独立站稳观察器，未伪称已有控制权。第一帧完整脑原始 `(vx,vz,wz)=(0.2241348177,0.1069552675,0.2135996372)`，按固定 `vx>0.03` 产生 FORWARD；仅纵向轴执行。50 次非零发布，世界 +x 实际位移 **0.4228134454 m**，随后 `ack-001.json` 记录本源 STAND/RUNNING 和 core.STOPPED。
- 第二观测确实来自运动后的新 odom/MC，目标世界坐标保持不变，输入像素 hash 与第一帧不同。第二帧推理 **3.581242083 秒**，原始 vx 为 **0.2807895243**，但其输出**未被接纳为第二轮动作**；第二观测之后非零发布数为 **0**。
- 第二观测 odom 接收时刻 `65.123469071`，角速度约 `(-0.00135013,0.00036119,0.01468439) rad/s`。拒绝前最后 odom 时刻 `68.792735448`，相对原观测水平位移仅 **0.0009039334 m**，完整姿态差 **0.0485267429 rad**。因此此次失败明确来自姿态门限，不是位置门限；不是跨主机 monotonic 相减得出的结果。
- 桥以 `ValueError:Observation drift exceeds bound` 失败，`final_stopped=true`、`active_forward=false`、`cleanup_completed=true`。宿主与 worker 同样失败；无 `ack-002.json`、`host_ack.json` 或最终 `parameters.json`。不能把完成两次神经 forward 写成完成两轮闭环，也不能宣称此次失败运行已完成最终参数前后封存核验。

## 资源与关闭

既有 0.1 秒父监督采样记录最低系统 available **2,018,869,248 bytes**（高于 1 GiB），进程树 RSS 峰值 **1,650,900,992 bytes**。失败原因是桥拒绝姿态漂移，不是资源越界。宿主约 77.40 秒退出，桥约 16.28 秒退出。

`logs/x2_docker/20261023_neural_loop_r2/lifecycle.json` 记录 GPU VM 从启动到关闭约 **71.56 秒**（小于 120 秒），GPU 容器 exited、lighter stopped、CPU 容器 exited、默认 Docker context 仍为 desktop-linux。无需为审查或截图重开仿真。

软件协议的默认无 IO、输出绑定、持续零速和失败终止检查已有证据；本次集成发现的是观测前姿态稳定条件不足。通过上述局部返工及新验收前，Spec021 不得标记完成。
