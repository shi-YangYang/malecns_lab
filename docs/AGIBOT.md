# X2 官方仿真与现场接入

当前选择 **智元官方 X2 Sim + MC + AimDK**。Mac 继续承担开发和 MaleCNS 计算，下一步尝试在独立 `linux/amd64` Docker 环境运行官方组件。尚未安装或跑通整套 X2；不把容器启动成功、机器人模型显示或单轴实验当作机器人控制成功。

## 官方组件

| 组件 | 作用 |
|---|---|
| `sim_mujoco` | 机器人物理仿真、关节与惯性状态反馈 |
| `mc` | 官方运动控制，将高层运动目标转换为关节指令 |
| AimDK | 应用侧感知与运动接口；具体可用接口以对应版本为准 |

官方仿真文档要求 **x86_64 Ubuntu 22.04**。Apple Silicon Mac 的容器默认架构为 ARM，尝试官方路线时需要 `linux/amd64` 转译；图形显示、MC 启动、通信和整体性能均需实测。当前没有验证 Mac 与官方 Linux 部署等价，也没有更换第三方步态控制器。

- [官方开发站](https://x2-aimdk.agibot.com/zh-cn/latest/)。
- [官方仿真部署](https://x2-aimdk.agibot.com/zh-cn/latest/sim_contents.html)。
- [SDK 与组件下载](https://x2-aimdk.agibot.com/zh-cn/latest/get_sdk/index.html)：`mc-x86`、`sim_mujoco-x86` 在 Extras 区。
- [运动与关节接口](https://x2-aimdk.agibot.com/zh-cn/latest/Interface/control_mod/index.html)。
- [Docker Desktop 设置](https://docs.docker.com/desktop/settings-and-maintenance/settings/)：转译能力取决于宿主、虚拟化后端及设置；不能据此推断 X2 已兼容。

## 接入顺序

1. 在独立容器验证 x86 程序及官方组件可启动，不停止或改动正在使用的 Noria 服务。
2. 先通过官方指令验证仿真 X2 的站立、前进、转弯和停止，并记录身体反馈。
3. 接入感知 → MaleCNS → 动作解码 → AimDK → MC → X2 的反馈闭环；明确相机、坐标、控制周期、时延和失效停止。
4. 与上海现场的具体 X2 型号、固件、MC/AimDK 版本及接口权限对齐，再进行现场适配。

既有 `BodyVelocity` 是 FLU 坐标的 SI 速度目标，不能不经核对直接当作 AimDK 参数。平衡和步态由官方 MC 负责；MaleCNS 的任务输出需要独立验证。单轴稳向实验已停止作为接入前置，结果见[历史实验](EXPERIMENT_RESULTS.md)。当前仓库没有 X2 SDK 驱动、仿真接入或真机验证。
