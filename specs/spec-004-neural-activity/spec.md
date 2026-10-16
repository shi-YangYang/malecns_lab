# Spec — 离线神经活动记录与刺激分析

状态：Done。用户已选择先分析录好的视频；实施、目检和独立验收已完成，AC1–AC5 全部 PASS。提交由本次交付保存。

## 背景

用户希望先观察不同刺激下的神经信号，再研究通用解码器。已有 shadow 输出经过人工 MotorDecoder，不能替代神经活动诊断；地面录像和离线 Brain 重放已可复用。

## 目标

把现有地面录像顺序送入冻结网络，记录视觉编码驱动与神经放电，生成可重复检查的曲线、热图和刺激比较；不需要动作标签或连接设备。

## 非目标

不训练或接入动作解码器、不实飞、不改变神经/视觉算法、不自动推断刺激或认知功能，不建立完整网页应用；不采集或提交真实实验资产。

## 当前行为

record 保存 session.json/video.avi/frames.csv 与待填 labels.csv；learning.prepare 能顺序重放但需要期望动作标签。Brain.last_counts 是每个 tick 的全脑脉冲计数；空群组的 last_rates 当前为零，诊断必须另行区分。

## 目标行为

`python -m cnsbridge.diagnostics.activity SESSION --out-dir NEW_DIR [--stimuli CSV]`。默认使用 MaleCNS 配置；`--brain minifly` 仅检查软件流程。无 stimuli 时也能观察完整信号；传入 CSV 后按刺激区间比较。

## 功能需求

1. 只读现有地面录制会话；沿用 read_recording 校验，不读取动作标签。逐帧重放全部视频，每帧固定 50 ms，独立初始化网络，零 IMU yaw。
2. 独立 stimuli.csv 列为 start_s,end_s,stimulus；使用 AVI 播放时间，区间左闭右开，允许自定义名称、重复刺激和无标记空隙。baseline 为明确标注的静止基线。缺失基线时不给差值。
3. 保存每输入群组的编码驱动均值 Hz、每输入/输出群组的实际平均放电率 Hz、最多 --max-neurons（默认 128）的唯一采样神经元逐窗口计数与率。按群组轮流确定性选样，记录抽样方式，不声称代表全脑。
4. 元数据包含有效配置（含人工 bias）、网络来源与哈希、源码身份、seed、群组角色/成员/数量/缺失、采样全局索引/type/side/body ID，视频/主机/仿真窗口时间与处理耗时。空群组为缺失（NaN/N/A），非零数量但无放电才为 0 Hz。
5. 输出 activity.npz（无 pickle）、report.json、group_rates.csv、stimulus_summary.csv、输入驱动/群组放电曲线和单神经元热图，以及本地 index.html 汇总。原始活动不平滑；如有显示平滑须明确标记。默认暖机 2 秒保留原始记录，但比较排除暖机和跨刺激边界的窗口。比较先逐区间，再按刺激汇总，不给伪显著性。
6. 新输出目录独占，拒绝覆盖。帧数不足/过多、坏标注、无法整除 50 ms 的 dt 和缺依赖等应清楚失败，不留下成功外观的报告。CLI 帮助不需要 SDK、OpenCV 或 Matplotlib。
7. 文档给出 30 秒静态相机刺激安排、按实际视频时间标注、命令和读图步骤，明确记录的是模拟活动，不是活体测量、训练结果或避障证据。

## 边界条件

拒绝非有限/越界/重叠区间和空名称。未标记不等于 baseline，视频时间不等于主机时间。全空群组仍生成缺失诊断。采样有硬上限，不保存全脑逐时刻矩阵或完整 raster。处理耗时不是生物反应延迟。

## 技术约束

新增 diagnostics/ 业务职责，使用现有 Brain/Retina/InputEncoder 和 NumPy；Matplotlib Agg 生成科研静态图，新增 analysis 可选依赖。四个训练来源哈希文件 brain/brain.py、brain/lif.py、senses/retina.py、senses/encoder.py 保持字节不变。遵循 Decision 004。

## 兼容性要求

旧数据和 flydrones.* 标识不变；新格式使用 cnsbridge.neural_activity version 1。不恢复旧 tools/兼容层，不改实验控制和学习格式。输入视频、模型、日志保持只读。

## 验收标准

- [x] AC1：离线入口无需动作标签和 SDK，正确顺序重放，错误输入清楚失败，拒绝覆盖。
- [x] AC2：已知计数验证单位、缺失与静默、唯一采样上限和元数据索引；区分输入驱动与实际发放。
- [x] AC3：刺激时基、边界、暖机、无基线与重复刺激比较正确，所有帧都推进神经状态。
- [x] AC4：短 MiniFly 合成视频端到端生成可读数据、报告和静态图；一次图像目检。明确合成软件验证，不宣称真实 MaleCNS 有效性。
- [x] AC5：当前文档与依赖同步；核心四文件、旧资产/格式/控制不变；独立验收后按用户请求提交。

## 待确认问题

无阻塞问题。用户已确认离线；第一版支持现有地面录制会话，原始普通视频导入留待另立需求。
