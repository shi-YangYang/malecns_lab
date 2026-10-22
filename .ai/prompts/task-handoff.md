# 当前交接 — 完整MaleCNS到X2纵向MVP与冻结优化完成

更新：2026-10-23。工作目录 `/Users/yang/project/malecns_lab`。

## 当前结论

用户授权的“跑完MVP并完成优化，每阶段可本地commit、自行决策”已达到约定的纵向工程范围；不push、不训练、不接真机。本轮[Spec021独立验收](../../specs/spec-021-x2-neural-loop/acceptance.md) PASS，必要检查与实际验证完成后停止追加运行。源代码和报告本地提交，data/logs/模型/来源快照不入Git。

完整166700节点、25582938非零边、124177617连接计数的冻结脑，通过实际里程计驱动的虚拟标记输入与官方Sim + MC + AimDK完成两轮反馈。原始vx按固定>0.03门限触发每轮1秒MC目标0.2，实际前进0.41884m和0.42901m，两轮均确认水平STOPPED。第二输入变化368像素；参数与图计数不变，完整状态600/1100/1600步封存，worker/监督/host_ack/桥清理关联齐全。

## 已完成阶段

- Spec016官方动作PASS，`2c14c31`；Spec017速度与停车契约PASS，`65aae55`；Spec018动作适配器PASS，`2865d5d`。固定意图动作只算这些阶段，不冒充脑控制。
- [Spec019](../../specs/spec-019-x2-neural-inference/acceptance.md)完整脑冻结基线PASS，`5c8be07`；[Spec020](../../specs/spec-020-x2-frozen-optimization/acceptance.md)精确等价优化PASS，`a2108a4`。完整状态与输出精确一致，计算25.7747→9.7080秒，下降62.3351%；单独运行单帧约2.30秒。相关已通过检查不重复。
- Spec021最终54项定向检查一次通过，新独立验收r4 PASS。联调与VM共驻留时单帧3.6775/3.6744秒；第二输入前另等27.5958秒姿态稳定，不能用其计算耗时或GUI60FPS冒充闭环频率。
- 外围GPU兼容层 `5a26a7c`；官方SDK、MC、Sim及模型未修改。r1入口环境错误、r2姿态漂移拒绝、r3姿态等待超时均保留，详见本Spec实际报告与失败验收。

## 最终协议与限制

最终执行两轮，初始和下一输入均要求既有站立/水平低速条件外，再有1秒角速度范数<=0.005rad/s、姿态变化<=0.005rad的新双流窗口。等待姿态最多40秒，下一观测45秒，脑计算30秒、停车12秒，宿主240秒、含准备的VM120秒。输出消费前仍检查更新反馈及0.02m/0.03rad漂移。全程等待/思考发零，失败不续走。vz/wz原样记录但未执行。

这是虚拟标记、纵向离散动作、单次有界工程闭环；两轮均FORWARD，未证明反馈改变决策、学会目标跟随、连续实时导航或真机能力。初始站姿仍需官方SD及GUI Pause→Reset→Run；最终STOPPED是水平停稳，并非角速度全零。HAL持续角漂移内部根因未定位，但输入前与响应接纳门槛均未放宽。

## 环境与证据

- 当前GPU容器、私有VM、CPU仿真均已停止，完整脑进程已退出；默认Docker context仍desktop-linux，未重启Docker或操作Noria。不用仿真时继续保持关闭，避免发热。
- 最终原始目录 `data/x2_sim/agibot/data/var/x2-neural-loop-20261023-r4/`；编排、生命周期、分析、独立验收关联和截图 `logs/x2_docker/20261023_neural_loop_r4/`。VM开启至关闭95.553秒，ROS桥51.303秒，宿主101.204秒。
- RSS峰值1.53GiB、最低系统available1.99GiB，保留至少1GiB；MPS driver峰值不与统一内存RSS相加。共享锁、单完整脑、0.1秒监督继续适用。
- GPU runtime `data/x2_sim/gpu_probe/runtime-rebuild/`，独立LIGHTER_HOME/docker.sock；容器 `malecns-x2-gpu-trial`，6081，VM8CPU/6GiB、容器4GiB。CPU容器 `malecns-x2-x2-1` 保持停止。
- 宿主完整脑入口必须用 `PYTHONPATH="$PWD/src" .venv-train/bin/python -m malecns_lab.experiments.x2_neural`；普通 `.venv`用于轻量检查，缺少Torch。默认仅预览，执行需显式sim-only与全新data输出。容器ROS环境需source后prepend transport_src到PYTHONPATH，不能覆盖ROS路径。

没有未处理的本阶段验收问题或活动运行。后续真实相机、连续控制、动作轴扩展、训练或真机工作需要另定范围与验收，当前不自动启动。
