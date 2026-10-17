# Implementation Report — 清理旧房间仿真器

## Summary

完成 `cleanup-sim.md` 的实施范围，等待独立验收。删除仅由旧感知测试调用的 `Drone` 抽象和 `SimDrone` 房间仿真器，保留现行目标跟随环境。没有操作硬件、训练、修改 Retina 或创建兼容别名。

## Files Changed

- 删除 `src/cnsbridge/drones/base.py`、`src/cnsbridge/drones/sim.py`。
- `src/cnsbridge/drones/__init__.py`：只保留现行仿真职责说明，移除旧导出及导入副作用。
- `tests/test_senses.py`：删除旧仿真器及 `FlightCommand` 依赖；保留左右／上下方向检查及旋转与逼近的光流回归意图，用测试内纹理平面针孔投影生成刺激。
- `docs/ARCHITECTURE.md`：唯一活动旧路径说明更新为 `follow_env.py` / `follow_scenes.py`，明确其简化速度响应边界。
- 本实施报告。

## Important Decisions

调用检索确认旧类没有当前命令、Tello 或新学习链的调用；仅 `tests/test_senses.py` 使用 `SimDrone`。历史 Spec / 审计记录没有批量改写。

没有直接使用 FollowEnvironment 生成该回归的刺激：其背景固定在图像坐标，标记物是小块纯色区域，不能产生原测试所需的全视场纹理光流。替代夹具对同一静止纹理平面进行针孔投影：纯偏航仅旋转光线，逼近仅降低相机到平面的轴向距离。采样前断言光线在平面前方且纹理索引在边界内，不通过边缘夹紧制造光流。

原断言阈值 `max(rot) < 0.2` 和末 15 帧 `max(app) > 0.5` 均保留，Retina 算法不变。新夹具只验证此数值光流性质，不再声称覆盖房间几何、碰撞、飞行动力学或实飞。

## Validation

- Risk Level: **S2**，删除旧包导出并迁移感知检查来源。
- `PYTHONPATH=src .venv/bin/python -m pytest tests/test_senses.py tests/test_follow_environment.py::test_actions_change_next_camera_observation_with_flu_sign -q`：**6 passed in 0.14s**。其中 3 项为感知检查，3 项是现行目标环境的三轴导入／运动路径检查。
- `git diff --check -- src/cnsbridge/drones tests/test_senses.py docs/ARCHITECTURE.md`：通过。
- 在 `src tests docs README.md examples` 搜索旧类名、旧模块路径及旧包导入：没有活动引用。
- 未执行全量 P2a、项目全套、脑模型或硬件检查。上述定向检查通过后停止验证。

## Known Limitations

本批仅清理不再使用的房间仿真器。旧 CLI、过滤构图和既有 Tello 控制链不在本批范围，等待对应新实现替换；没有把整体学习任务标为完成。

## Remaining Questions

无实施阻塞问题。独立验收与阶段 commit 由协调者负责。
