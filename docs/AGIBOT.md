# 智元 G2／X2 开发资料

资料核对日期：2026-10-20。G2／X2 是后续设备接入候选，**当前仓库尚无对应 SDK 驱动、仿真接入或真机验证**。文档可访问不代表软件包已获得，或其版本与现场机器人兼容。

## X2

- [官方中文开发站](https://x2-aimdk.agibot.com/zh-cn/latest/)。
- [SDK 获取](https://x2-aimdk.agibot.com/zh-cn/latest/get_sdk/index.html)：页面说明 SDK 通过右下角浮动工具栏下载；仿真所需的 `mc-x86`、`sim_mujoco-x86` 在 Extras 区单独下载。
- [运动与关节控制](https://x2-aimdk.agibot.com/zh-cn/latest/Interface/control_mod/index.html)。
- [MuJoCo 运控仿真](https://x2-aimdk.agibot.com/zh-cn/latest/sim_contents.html)：要求匹配的 x86_64 Ubuntu 22.04 环境和官方运控／仿真包。当前 Mac 不能直接视为该官方部署环境。

## G2

- [Genie 02 GDK v2.1.0 开发手册 PDF](https://cdn.robotshop.com/media/A/Agb/RB-Agb-09/pdf/Genie-02-GDK-v2.1.0.pdf)：92 页厂商手册，由 **RobotShop 第三方托管**，包含 Python、C++、ROS2 与设备控制开发内容；未确认是最新版本。
- [智元官网 G2 用户手册入口](https://www.agibot.com.cn/filepage/362.html)：第 14 章介绍开发者工具，从机器人无线终端的“开发者工具”App 查看本地文档，在线文档入口需要连接机器人并登录。
- [官方 Genie Sim](https://github.com/AgibotTech/genie_sim)：可作为仿真资料补充，不能替代现场 GDK 文档或证明仿真与真机 API 一致。

## 接入前需要确认

先向设备提供方核对现场型号、固件版本、可用 SDK／仿真包、开发机器要求，以及摄像头和高层运动接口权限。A2、A3、A2W 的 AimDK 文档不能自动当作 G2／X2 的接口说明。

候选链路为：感觉输入 → MaleCNS 神经计算 → 动作读出 → 设备能力与坐标适配 → 厂商高层运动接口。既有 `BodyVelocity` 是 SI 机身速度目标；无人机、轮式底盘和人形机器人的支持轴不同，不能只替换 SDK 函数名。平衡、步态和关节控制是否交给厂商控制器，应在对应设备 Spec 中明确。

正式接入需单独定义输入反馈、控制周期、时延、停止和人工接管条件，再分别验收仿真与现场实机。当前训练指标与既有 Tello 实验不能作为智元机器人控制效果的证据。
