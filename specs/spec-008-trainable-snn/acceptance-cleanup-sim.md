# Acceptance — 清理旧房间仿真器

## Result

PASS

2026-10-17，由未参与本轮实施的独立验收 Agent 检查。结论仅覆盖 [清理子规格](cleanup-sim.md)，不代表完整图训练、神经控制或实飞通过。

## Spec Coverage

- `drones/base.py`、`drones/sim.py` 已删除；`drones/__init__.py` 仅保留职责说明，没有旧导出、兼容别名或转发包装。
- 独立检索 `src`、`tests`、`docs`、`README.md`、`examples`，旧类名与旧模块引用没有活动命中。删除前旧仿真器的调用仅存在于本轮迁移的感知测试；新目标环境直接导入现行模块，不依赖已删除导出。
- 左右运动与向上运动的感知检查未改。旋转与逼近检查仍分别生成 30 / 45 帧，保留 `max(rot) < 0.2` 与末 15 帧 `max(app) > 0.5` 的原断言，没有修改 Retina 算法或放宽阈值。
- 替代刺激局限说明充分：FollowEnvironment 的背景固定在图像坐标，纯色标记无法提供该回归需要的全视场纹理。测试内针孔投影从同一静止纹理平面采样；偏航旋转相机光线，逼近降低轴向距离，投影公式符合这两种运动。显式检查正深度和纹理边界，未用边界夹紧制造变化。该 helper 仅在测试内，不是新增运行时仿真器。
- `docs/ARCHITECTURE.md` 将现行仿真职责指向 `follow_env.py` / `follow_scenes.py`，保留简化速度响应、非 Tello 标定动力学的边界。新目标环境、V2、神经核心及设备链不属于本子批改动。

## Validation

- Risk Level: S2。
- 阅读项目规则、使命、技术栈、任务交接、独立验收流程、Decision 009、总 Spec/Plan、清理子规格和实施报告；审查范围内删除文件的 diff、新测试全文和相关 Retina / FollowEnvironment 实现。
- 采纳实施者已有定向检查记录：`PYTHONPATH=src .venv/bin/python -m pytest tests/test_senses.py tests/test_follow_environment.py::test_actions_change_next_camera_observation_with_flu_sign -q`，6 passed；范围内 `git diff --check` 已通过。
- 本轮独立执行只读引用检索和差异审查。没有重复已通过测试，没有运行训练、完整图探针或硬件；没有修改业务文件或其他并行工作。

## Issues

未发现阻塞本子批验收的问题。

## Regression Risks

- 外部未跟踪的使用者若仍导入旧仿真类，将失去这些入口；这符合已授权移除无用实现、不保留兼容层的范围。仓库内没有剩余调用。
- 新刺激检验数值光流区分性质，不覆盖旧房间几何、碰撞或飞行动力学；实施报告与测试注释均已明确这一范围。
- 本批不会替换仍在使用的旧 CLI、过滤构图或 Tello 控制链，其迁移仍需相应替代实现与独立验收。

## Required Rework

无。协调 Agent 可将本清理子批标记完成并单独提交；整体学习目标仍未完成。
