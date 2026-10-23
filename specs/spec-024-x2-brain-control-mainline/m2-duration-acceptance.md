# M2 连续技能时长透传独立验收

日期：2026-10-23。验收者：新的独立只读验收代理。范围为 `m2-pulse-duration-implementation.md`、`m2-actual.md` 第五次登记、两层 CLI 的时长透传及其定向检查。风险等级 S2；没有修改业务代码或历史产物。

## 结论

**工程验收 PASS；准入第五次登记的一次有界仿真运行。** 只请求 `TURN_LEFT → TURN_RIGHT → WAVE_RIGHT`：左转固定两秒通过后才运行右转两秒，右转通过后才观察原官方挥手自然完成；任一失败停止后项，不复跑已通过的 HOLD/ADVANCE。此结论不表示两秒转向已成功、不表示挥手或其取消能力已通过，也不表示 M2/P2 整体完成。

## 差异与链路核对

以实际第四轮 `data/x2_sim/agibot/data/var/skills-20261023-151800-93a1e497/host_source.py` 及该 run 的 `transport_src` 为基线比较：

- `experiments/x2_skill_host.py` 的新增差异仅为引入时长校验、默认 1 秒的新 keyword/CLI 参数、向远端命令传递以及相应记录。`experiments/x2_skills.py` 的新增差异仅为同一参数的解析、校验、传入核心与记录。`control/x2_skills.py` 与运行快照逐字节相同。
- 两层 CLI 在 preview/执行前校验有限值且只能精确为 1 或 2；host 直接构造和 runner 的 `execute` 入口也校验。默认仍为 1 秒。实际链路是 host CLI → `SkillRuntime.command_seconds` → 远端 `--command-seconds` → runner args → 真正的 `SkillExecutor(command_seconds=...)`。
- host preview、运行事件和 summary 保存选择值；runner preview、configuration 与 summary 保存选择值，核心已构造时后两者读取 `core.command_seconds`。源快照/哈希机制保留，可核对本次条件而不改写旧 run。
- M4 `BehaviorRuntime` 仍以四个位置参数调用 `super().__init__`，新增 keyword 有默认值，不改变该入口的 1 秒默认条件。host 引入的校验模块在顶层不导入 ROS 或启动服务。

## 未变化的执行边界

- 核心仍以 `0.2 × intensity` 产生目标；isolated runner 的 intensity 为 1。TURN_LEFT 为 `+0.2 rad/s`，TURN_RIGHT 为 `−0.2 rad/s`，不翻转符号、不提高速度。时长同时进入连续技能截止时间与原速度命令 TTL，既有核心上限仍为 2 秒。
- 两个转向最终净角分别必须 `> 0.02 rad` / `< −0.02 rad`，且真实运动、保护状态、水平停稳、角速度和角漂移判据均保持原实现。第二秒增量仅作诊断，不能替代最终通过判据。
- WAVE_RIGHT 分支仍请求 `motion=1002, area=2, interrupt=False`，观察期限仍为 12 秒，与 `command_seconds` 无关；原归属、关节运动、返回、自然完成和取消未知的处理均未改。
- host 默认 UI/总预算仍为 80/240 秒，第五次显式选择已获准的 180/360 秒；清理保留 30 秒、单 probe 40 秒、下项至少 41 秒余量及前项失败停止后项均未变化。资源清理、私有运行环境、互斥和默认 Docker context 检查保留。

## 验证依据与停止

实施者已报告唯一运行的定向检查 `tests/test_x2_skill_duration.py` 为 **1 passed in 0.03s**；本验收阅读该检查并静态追踪真实构造和命令路径，没有机械重复。该检查通过隔离替身把 host CLI 贯通到真实 runner/核心，覆盖默认 1、显式 2、非法/非有限值、两层记录以及 M4 默认构造；动作源注册被立即拒绝，不能将其作为实际动作证据。

本验收只做源快照差异、相关代码和报告核对；未运行测试、build、lint，未启动 VM、ROS、完整脑或硬件。未发现本次改动直接相关的未解决问题，停止验证。由协调者执行已登记的新 run，并保留实际非零发布区间、脉冲第 1/2 秒及归零后 yaw、最终停止结果和关闭资源证据；不得自动追加时长、加速或放宽阈值。
