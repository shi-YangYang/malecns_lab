# Implementation Report

## Summary

新增地面录像的离线神经活动诊断入口。现有视频逐帧进入 Retina、InputEncoder 和冻结 Brain，不读取动作标签、不实例化 MotorDecoder，也不连接设备。输出编码驱动、群组真实发放率、有限采样神经元窗口计数/发放率、刺激区间比较和静态图表。

## Files Changed

- `src/cnsbridge/diagnostics/collector.py`：逐帧重放、唯一有界采样、三种时基、来源哈希与有效配置。
- `src/cnsbridge/diagnostics/stimuli.py`：独立刺激 CSV 校验、完整窗口筛选、暖机排除、逐区间及等权刺激汇总。
- `src/cnsbridge/diagnostics/reporting.py`：无 pickle NPZ、JSON/CSV、Matplotlib Agg 曲线/热图与本地 HTML。
- `src/cnsbridge/diagnostics/activity.py` 与 `__init__.py`：延迟依赖 CLI、新目录独占与失败清理。
- `tests/test_neural_activity.py`：数据正确性与端到端检查。
- `pyproject.toml`：新增 analysis extra，all 包含 Matplotlib；既有核心依赖不变。
- 用户文档由独立文档实施者更新 README、NEURAL_ACTIVITY、DATASET、ARCHITECTURE；治理由协调者更新。

## Important Decisions

- 固定 50 ms 窗口；要求正 dt 精确整除，至少一个仿真步。保留全部帧原始活动，包括暖机和无标记片段。
- 群组均值覆盖组内全部神经元；单神经元按排序群组轮流、组内升序采样，去重，默认 128、硬上限 4096。禁用不需要的 raster 采集。
- 空群组以 NaN / N/A 表示，真实静默才是 0 Hz；输入驱动与神经元发放分别保存、绘制。JSON 用 null 表达无有效观察，不输出非标准 NaN。
- 刺激标记使用 AVI 播放时间及左闭右开区间。比较先排除暖机和跨界窗口，再算区间均值，按刺激等权汇总区间。只有显式 baseline 的有效区间参与基线差值，不做显著性推断。
- 输入会话、帧表、视频和可选刺激文件均记录 SHA-256；图内容哈希、相关源码哈希、配置/bias、seed、库版本和神经元索引映射写入元数据。
- 曲线每面板最多六组，原始曲线不平滑。热图明确为 50 ms 窗口率，不是逐脉冲 raster。
- 新输出目录已存在即拒绝；途中出错删除本轮独占新目录，避免留下成功外观报告。

## Validation

- Risk Level：S3（新增持久化格式与可选依赖），范围仅新诊断子系统。
- Checks Performed：`.venv/bin/python -m pytest tests/test_neural_activity.py -q -s`
- Result：**18 passed in 2.14s**，退出码 0；只执行一次，没有追加完整测试、build 或 lint。
- 覆盖已知脉冲计数/Hz、缺失与静默、索引/精确 body ID、采样上限、dt、刺激校验/边界/暖机/重复区间/基线、短/长视频清理、全空群组报告、阻断可选依赖的帮助，以及 MiniFly 端到端与覆盖保护。
- 合成端到端保留用于一次目检：`data/neural_activity_demo/synthetic_32459262ba/report/`。该目录是忽略的本地资产，包含 60 帧/3 秒合成棋盘格输入，模型为 MiniFly；不是用户真实视频或真实 MaleCNS 实验。
- 协调者已用 `view_image` 一次目检三张合成图：轴、单位、图例清楚且无截断；输入驱动与放电分图，时间与热图采样说明可读。没有需修改的图像问题，未重复生成。
- diff 检查及最终独立验收交新的验收 Agent，本报告不自行填写验收 PASS。

## Known Limitations

只支持当前 ground_recording 会话，不导入任意裸视频；固定相机、零 IMU yaw 和现有视觉编码/bias 均是建模条件。采样神经元不是全脑代表，组间可有成员重叠。处理耗时不是生物反应延迟。未运行全量 MaleCNS、真实录像、设备或训练；不据软件检查推断动作有效性。

## Remaining Questions

无实施阻塞问题。待协调者处理目检与独立验收后提交。
