# Implementation Report — P2a 视觉目标跟随环境

## Summary

实现独立于神经核心的三轴标记目标跟随环境、像素检测/空间 mask、分区场景注册、视觉基线与特权教师、闭环执行和证据日志。没有加载 MaleCNS、训练模型或连接设备。本报告是实施记录，不替代独立验收。

## Files Changed

- `src/cnsbridge/drones/follow_env.py`：相机四角投影、一阶速度响应、位姿更新、动作限幅和终止条件；真值与 frame 分开访问。
- `src/cnsbridge/drones/follow_scenes.py`：不可变场景、独立目标运动、显式 train/val/test 族与固定种子。
- `src/cnsbridge/senses/target.py`：BGR 像素检测、不可变有效性观察、保持图像空间的占用率 mask。
- `src/cnsbridge/learning/follow_task.py`：不含真值的 policy API、分离的真值教师、视觉/零动作基线、保护/评分/JSONL 和模块 CLI。
- `tests/test_follow_environment.py`：本子系统定向软件检查。
- `docs/FOLLOW_TASK.md`：契约、命令、坐标、协议、schema、首次运行数据及限制。
- 本报告。

## Important Decisions

遵循 Decision 009 和 P2a Ready Spec。只复用 `BodyVelocity`；未修改旧 sim、核心后端、共享动作或治理入口。相机 x_error 向右为正，y_error 向上为正；FLU yaw 正值向左，因此视觉基线 yaw 与 x_error 符号相反。教师原始动作、策略原始动作和限幅执行分别记录。无效观察停止目标速度且单独记录，不记作神经网络行为。

## Validation

- Risk Level: S2。
- 首次定向检查：`PYTHONPATH=src .venv/bin/python -m pytest tests/test_follow_environment.py -q`，10 passed / 1 failed。失败为前进 0.5 s 后标记高度仍处于同一离散像素尺寸；位姿与图像已有变化，测试窗口不足以越过量化。只将三轴符号测试窗口延长到 1 s。
- 变更后仅重跑受影响参数化检查：`PYTHONPATH=src .venv/bin/python -m pytest tests/test_follow_environment.py::test_actions_change_next_camera_observation_with_flu_sign -q`，3 passed。其它 8 个通过项不重复执行；当前 11 个检查均有通过证据。
- 随后加强失去目标检查，用持续非零策略验证保护确实覆盖原动作，而非只观察零动作；仅运行该改动项 `test_occlusion_and_out_of_view_are_invalid_and_do_not_score_success`，1 passed。
- 检查涵盖像素格式/静态/移动/空白/遮挡/裁切/出画、多候选、相机闭环三轴符号、种子重放、动作独立目标轨迹、限幅/不支持轴/响应滞后/终止、回调真值边界、日志保护、场景分区与视觉基线对照。
- 实际模块 CLI 各执行一次 `train-sine-combined` 的 visual / zero，8 s、80 步；输出保存在 `data/follow_task_smoke_20261017/visual/` 和 `zero/`，本地忽略资产。视觉基线横/纵/距离 MAE 为 0.06855 / 0.09631 / 0.17853 m；零动作为 0.14232 / 0.15100 / 0.51529 m；各轴非零步数 80/80/77 对 0/0/0。
- 两者无碰撞；视觉基线合格比例 75%，低于登记的 80%，所以任务成功为 false；没有据此调阈值或使用测试集选参。此结果只验证软件闭环与评分行为。
- 未运行全套测试、训练后端、完整脑、测试分区模型评分或任何硬件。

## Known Limitations

运动目标层仿真未经 Tello 动力学标定；目标碰撞是距离代理，没有障碍物几何。红色标记检测有像素量化/斜视误差，只能排除可见的缺损与裁切，不能保证辨别所有遮挡。不含姿态/电机控制，也没有目标搜索策略。主模型接入、学习效果和真实飞行均待后续批次。

## Remaining Questions

后续训练直接复用 `rollout(scene, policy, policy_name, record_frames)` 和只含图像/相机/时间的 `FollowPolicyInput`。通过 `record_frames` 获取输入，通过日志独立提取教师标签；不得把 truth 或观察误差直连到主动作读出。完整 SNN 空间感觉映射与训练预算由后续核心/探针结果决定。
