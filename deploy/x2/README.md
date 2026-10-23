# X2 GPU 部署

这是 X2 仿真的唯一正式部署配置。它直接基于固定版本的官方 ROS Humble 镜像，安装 Mesa Venus/zink 与 VirtualGL，不依赖已退役的 CPU 镜像。MuJoCo 物理和官方 MC 的计算方式没有因此改变。

## 目录职责

- `Dockerfile`、`compose.yaml`：GPU 容器镜像和运行配置。
- `manage.sh`：主机上的部署入口，只连接项目私有 Lighter socket。
- `scripts/`：容器启动脚本，以及独立的官方预置动作兼容性工具。
- `keys/kisak-turtle.asc`：公开的 PPA 验签公钥，Dockerfile 校验指纹后启用软件源。

官方 SDK、MC、Sim 留在 `data/x2_sim/packages`，运行输出绑定 `data/x2_sim/agibot`；本目录不包含模型、依赖副本或本地 VM。现有私有 Lighter 保留在 `data/x2_sim/gpu_probe/runtime-rebuild`，本轮不迁移它。

## 日常运行

通过桌面应用管理 GPU 仿真生命周期，见[桌面使用说明](../../apps/desktop/README.md)。应用使用现有容器 `malecns-x2-gpu-trial`；无需为了这次目录整理重建容器。使用后退出实验，释放 GPU 资源。

noVNC 的主机入口是 `http://127.0.0.1:6081/vnc.html`，容器内部端口仍为 6080。容器默认命令只启动显示桌面；Sim 由应用通过 VirtualGL 启动，MC 与脑进程也由应用分别管理。

## 部署维护

在项目根目录运行以下离线检查，不需要启动 VM：

```bash
bash deploy/x2/manage.sh config
```

只有需要重新部署时才使用下面的维护命令。它们要求私有 GPU VM 已处于可维护状态；脚本不会启动 VM，不修改默认 Docker context。不要在桌面实验占用 VM 时构建或创建容器。

```bash
bash deploy/x2/manage.sh ps
bash deploy/x2/manage.sh logs
bash deploy/x2/manage.sh build
bash deploy/x2/manage.sh create
```

`build` 构建 `malecns-lab-x2-gpu:local`；`create` 只创建停止状态的容器，镜像必须已存在，且拒绝覆盖同名容器。已有安装不需要执行这两步。所有命令固定使用私有 socket，不使用裸 `docker compose up`，也不操作 Docker Desktop 的其他服务。

绑定目录必须已存在。Compose 使用相对本配置文件的 `../../data/x2_sim/packages` 和 `../../data/x2_sim/agibot`，并禁用自动创建空目录。CDI 设备 `lighter.sh/gpu=all` 由私有 Lighter 提供。

独立兼容性工具入口已移动到 `deploy/x2/scripts/preset-compat.py`；其准备、编译及验收属于相应任务，不随部署或应用启动自动执行。

## 验证边界

部署整理阶段仅解析 Compose、检查 shell 语法及部署入口的隔离行为，未重新构建镜像或启动 GPU VM。现有 GPU 实测属于已有安装；新 Dockerfile 从头构建、固定软件包在远端源的可用性和新建容器运行尚未验证。若固定版本不可获取，应保留失败信息并单独评估，不自动改用新版本或 CPU 渲染。

历史 GPU 图形兼容修复与 CPU 部署退役见 [history.md](../../history.md#spec-015)，官方软件、模型及实验资产不随文档清理删除。
