# 当前交接

更新：2026-10-28。当前请求为目录与文档整理，不恢复长期训练或仿真实验。使用入口见 [README](../../README.md)，运行手册见 [桌面说明](../../apps/desktop/README.md)，阶段证据统一见 [history.md](../../history.md)。

## 当前实现

- `npm --prefix apps/desktop start` 打开已有构建；点击一次“开始”自动准备脑与仿真，顶部“停止”负责独立停车/清理。主界面不再有回放选择器或手动 Pause/Reset/Run 步骤。
- 本机候选登记在 `data/x2_behavior_desktop.json`，指向 `data/x2_model_redesign_20261024_r1/b_current_history_v1`。后端固定 `.venv-train/bin/python`；桌面为 continuous 会话，默认执行仍为 measured / baseline / official，不能用显式 v11 / noslip1 成绩替代默认版本验收。
- 全 166,700 细胞窗口状态已采集；674,411 个真实来源点覆盖全部细胞，21,952 条抽样连接为直线关系示意。紫色节点按 ΔV，金色连接按突触前 Hz；零基线与显示平滑不冒充神经测量。
- 收尾时允许预选一个下一动作，实际反馈消费和停稳完成后才提交；停止、结束互动、异常取消预选。源码与 UI 检查通过，实际接续仍有下述缺口。
- 固定白点已在 `2958420` 修复：GLSL 保留字导致神经元着色器编译失败；实际渲染确认消失，Electron 构建已更新。

## 关键证据与遗留项

- v11 / noslip1 的 `desktop-20261026-103028-ccb4411f` 中 GREET→LEFT→RIGHT→GREET 4/4 通过，88.294492° / 90.483790°；此前矩阵失败保留。全部 16 对及 60 分钟/100 请求未完成，真机未接入。见 [连续动作历史](../../history.md#x2-continuity-20261026)。
- `desktop-20261027-072331-1146db3a` 单次前进 1.213814 m，52 批全脑状态完整；该场没有触发过期反馈重试，只证明正常链路回归。见 [活动与反馈](../../history.md#desktop-activity-20261027)。
- `desktop-20261027-081750-f57cd686` 收尾按钮可预选已实际观察，随后第二请求触发 120 秒预算；自动接续派发的 UI 证据使用真实收尾状态加模拟新请求应答，不是机器人接续成功。`logs/next-action-ui-evidence.json` 保留该边界。
- Electron 显示层恢复的软件检查及普通后台启动通过；原生故障后的自动恢复分支尚无真实触发证据。既有 preview 导入 NumPy 检查失败仍未处理。见 [稳定性历史](../../history.md#desktop-stability-20261026)。

## 后续工作边界

目录整理不启动训练、脑推理、仿真或依赖安装，不改变模型、目标、评分与运行参数。后续若继续动作验收，先核对当前用户授权、实际进程与资源，不沿用旧“正在运行/已经停止”的快照。

保留数据、模型、录像、日志、环境及旧 run 的 sources；私有 GPU 在 `data/x2_sim/gpu_probe/runtime-rebuild`，禁止为本项目停止 Noria 或重启默认 Docker。文档与原始历史身份分开，旧 schema/hash 不机械改名。一个任务最多一份过程文档，整理记录直接见 [本次历史条目](../../history.md#cleanup-20261028)。
