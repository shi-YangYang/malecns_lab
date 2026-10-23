# 当前交接

更新：2026-10-23。工作目录 `/Users/yang/project/malecns_lab`。

## 当前状态

- 文档已精简：全部旧 Spec 与决策汇总到根 `history.md`，`specs/` 已删除；重复手册按主题合并。以后每任务最多一份 Markdown，原地更新，只留有用信息，规则见 [AGENTS](../../AGENTS.md)。
- CPU 仿真容器/专用镜像已删除，GPU 部署及运行目录已整理；目录与文档整理随本次本地提交保存；此前阶段快照为 `414d3f5`，没有 push。原始数据、模型、日志、sources 和官方软件保留。
- **当前打招呼、左右、向前按钮无动作，原因待定位。** 训练、roadmap 和实测保持暂停，不能借文档清理恢复。后续主线应先查清按钮到动作的实际链路，不把模型能力当成已确定原因。阶段证据与其他限制统一见 [history.md](../../history.md#spec-024)。

## 运行入口

- 桌面：`cd apps/desktop && npm start`，仅打开界面。后端为 `malecns_lab.runtime.x2.behavior`；使用说明见 [桌面 README](../../apps/desktop/README.md)。
- 本地 `data/x2_behavior_desktop.json` 仍指向 `data/x2_behavior_readout_20261023_r2/checkpoints/0149`，仅 qualification 候选，不自行更换、训练或部署。
- GPU 安装在 `data/x2_sim/gpu_probe/runtime-rebuild`，容器 `malecns-x2-gpu-trial`，端口 6081；维护见 [部署 README](../../deploy/x2/README.md)。不用仿真时关闭，不动 Noria、默认 Docker context 或其他服务。
- 真机暂缺，之后上海接入。旧工程检查与纵向实测不代表当前多行为按钮可用，新 GPU Dockerfile 未重建验证。
