# 独立验收：仓库清理与官方 X2 路线

日期：2026-10-21。验收者未参与本轮代码、文档或治理实施，仅静态审阅与记录结论。

**结论：PASS（清理结果与提交前验收）。** 未发现本轮直接阻断问题；允许协调者更新任务状态并执行用户授权的本地 commit。写入本报告时 commit 尚未执行，本报告不声称已提交，也不要求推送。

## 验收结果

| 标准 | 结果 | 证据与边界 |
|---|---|---|
| AC1：当前方向明确 | PASS | README、AGENTS、使命、技术栈、交接、`docs/AGIBOT.md` 与 Decision 021 一致采用官方 X2 Sim + MC + AimDK，并把 Mac `linux/amd64` Docker 写为待实测方案。没有宣称已兼容或已接通；单轴稳向明确为历史研究，非机器人接入前置。 |
| AC2：删除成组、保留入口可用 | PASS | 已移除单轴网页、部署包装与未执行信号诊断；保留网页只导入刺激实验面板和原图谱。静态搜索保留源码/测试没有指向已删模块的导入。`learning/yaw/` 所需 `yaw_env/yaw_task`、`yaw_drive/yaw_vision` 仍在；不以删除任务入口为由破坏历史数值来源。复用实施记录中的 39 项定向测试通过与一次前端构建通过，不重复执行。 |
| AC3：文档精简与来源协议 | PASS | 文档实施记录列明删除 213 份重复/过期 Markdown，合并 checkpoint/迁移说明并建立结果索引。协调者报告全仓 148 份现存 Markdown 的链接/锚点检查为 0 错误，本验收不重复该扫描。直接查看 `mps_contract.NUMERIC_PRECEDENT` 的两个受保护文件，均未进入 Git diff；原路径与原文保留。 |
| AC4：资产、许可和失败证据 | PASS | LICENSE、THIRD_PARTY、核心 brain/learning/Tello 的已跟踪数值源未改；原始 data/logs/环境不在删除范围。`docs/EXPERIMENT_RESULTS.md` 与保留的固定模型验证报告明确记录联合稳向控制 FAIL、两个方向退化和锁定测试未开启，区分流程通过与控制成功。文档没有声称所有未提交的删除报告能从 Git 恢复。 |
| AC5：必要检查、独立审查与提交 | PASS（提交前部分） | 本次按 S2 复用已经完成的定向检查，加上独立差异/依赖/证据审阅，范围与改动相称。实际本地 commit 由协调者在此 PASS 后执行并报告；未提交前不得将本项的提交动作标为已完成。 |

## 关键代码观察

- `visualization/jobs.py::JobManager.__init__` 先调用 `validate_parameters`，再进入旧进程恢复和状态写回；历史 yaw 参数不属于当前刺激契约，因异常而被跳过。旧任务不会被恢复、重写成失败或暴露为当前 API 作业。
- `tests/test_atlas_experiments.py::test_retired_experiments_stay_unmodified_and_outside_current_api` 明确比较旧 `job.json` 前后字节，并检查旧请求返回 400、旧 ID 返回 404。该检查已包含在实施者通过的 39 项中。
- `visualization/worker.py::run_experiment` 仍沿 `NeuralReplayV2` 执行完整参考图，保留输入帧、神经统计、来源身份和真实仿真时间；没有新增设备控制、合成脑回退或 yaw 调度。
- `apps/atlas/src/main.ts` 保留胞体图、群组响应、刺激预览与静态果蝇，来源说明继续区分固定 LIF 仿真、群组均值和真实设备行为。

## 检查范围与限制

独立验收实际执行了只读 Git diff/status、相关文件阅读和删除模块引用搜索；没有修改业务实现，没有运行测试/build/训练、完整脑、浏览器、Docker 或硬件。测试与构建结果来自 `implementation-code.md`；文档删减与来源保护来自 `implementation-docs.md`，全仓引用检查由协调者提供。没有重新校验大型数据文件的全量哈希，也没有将此清理验收当作旧实验的重新复现。

没有发现需要扩大验证或返工的本轮问题；到此停止审查，后续仅由协调者完成状态收口和本地提交。
