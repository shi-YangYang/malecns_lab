# 代码清理实施记录

日期：2026-10-21。风险：S2。范围：图谱前后端和已退出当前路线的单轴稳向入口。

- 删除 19 个源码/测试文件：`learning/yaw_deployment/` 的 5 个 Python 文件；两个 `diagnostics/yaw_signal_*` 文件；两个 `visualization/yaw_*` 文件；`control/yaw_readout.py`；四个网页闭环/学习回放模块；五个关联测试文件。
- 网页保留胞体图、群组放电率、刺激播放和可调刺激实验；删除单轴姿态、学习结果及稳向表单，相关类型、CSS、任务分发和产物校验同步移除。
- 历史 yaw 任务仅在加载时过滤，不恢复进程、不改写旧记录。现有测试文件新增一项归档保护回归检查，同时确认当前 API 拒绝旧 yaw 请求。
- 保留 `learning/yaw/`、其数值依赖和两个研究测试，作为历史离线研究；完整图、核心 SNN/GPU、Tello、数据、日志及运行环境均未改动。

## 验证

- 审阅相关 diff 与剩余导入，保留交互刺激所需的 `generated` 预览格式和目录切换。
- `.venv/bin/python -m pytest tests/test_atlas_experiments.py tests/test_atlas_export.py -q`：39 passed，3.03 秒。
- `apps/atlas` 内 `npm run build`：TypeScript 和 Vite 构建通过。现有 Three.js 打包体积提示仍存在，不属于本轮清理问题。
- 未运行完整脑、训练、全量测试、Docker 或硬件，也未提交。必要检查通过后停止验证，交独立验收。
