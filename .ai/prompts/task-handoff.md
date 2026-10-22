# 当前交接 — X2动作适配器通过，准备完整脑推理

更新：2026-10-23。工作目录 `/Users/yang/project/malecns_lab`。

## 用户目标与授权

持续目标：跑完MVP并完成优化；每阶段完成后可本地commit，自行决策，不逐项询问。不push、不接真机、不训练。完整MaleCNS使用真实参考资产，固定意图、重放和人工规则不能冒充脑控制。完整脑保留至少1GiB系统available，单完整脑进程，先测资源再制定仿真联调边界。不操作Noria或重启Docker。

仿真仅在需要的有界验证窗口开启，准备、编码、分析和结束后关闭，减少发热；CPU仿真保持停止。

## 已完成

- [Spec016](../../specs/spec-016-x2-motion-mvp/acceptance.md)：官方站立、前进与最终停止PASS，提交 `2c14c31`。
- [Spec017](../../specs/spec-017-x2-motion-contract/acceptance.md)：速度语义与停车核对PASS，提交 `65aae55`。0.2为MC目标，不能当实测速率或通用校准。
- [Spec018](../../specs/spec-018-x2-motion-adapter/acceptance.md)：STOP/FORWARD核心、显式ROS仿真入口及第四轮实测独立PASS，106项定向检查通过。第四轮前向净位移0.90069m；归零后5.09292秒确认停稳，最终连续1.00677秒最大水平速度0.01723m/s。相关通过检查不再重复。
- 外围GPU兼容层提交 `5a26a7c`；官方SDK、MC、Sim及模型未修改。历史60FPS是GUI渲染读数。

## 当前阶段

[Spec019](../../specs/spec-019-x2-neural-inference/spec.md) 已Ready，实施冻结完整脑前向基线。固定GPU训练最终checkpoint，仅继承theta/readout参数；使用校验后的旧来源快照复现原policy，新场景600步空白预热，三个预登记标记输入。单进程、240秒上限、6GiB申请和1GiB系统预留；由协调者显式运行，实施者不启动完整脑。无训练或ROS。

后续闭环计划采用实际odom驱动的虚拟标记，因为当前官方配置没有图像发布；必须明确传感来源，不能称真实相机导航。脑原始三轴输出保留，纵向离散意图门限事先固定；若模型不驱动任务，记录行为失败，不能用人工距离控制替代后宣称脑成功。完整图推理优化另登记后比较完整状态与动作，不裁图。

## 环境与证据

- GPU runtime `data/x2_sim/gpu_probe/runtime-rebuild/`，独立LIGHTER_HOME/docker.sock；容器 `malecns-x2-gpu-trial`，6081，VM8CPU/6GiB、容器4GiB。当前已停止。
- 最新关闭证据 `logs/x2_docker/20261023_motion_adapter/shutdown-stop-window.txt`：GPU容器exited、VM Stopped、CPU容器exited、默认context仍desktop-linux。
- Spec018最终原始目录 `data/x2_sim/agibot/data/var/x2-adapter-20261023-stop-window/`；前三轮失败/问题证据各自保留。源码散列、命令、分析、截图在 `logs/x2_docker/20261023_motion_adapter/`，不纳入Git。
- 主动关闭后Sim有Publish type unregistered日志，不称优雅退出。站立准备仍需官方SD与GUI Pause→Reset→Run，非自主起身。noVNC点击需100ms按下并确认状态。
- 已授权的动作阶段通过；完整脑感觉/动作闭环、实时性和现场版本匹配尚未完成。不得把阶段PASS写成整个持续目标完成。
