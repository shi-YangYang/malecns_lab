# M2 UI 准备预算修复独立验收

日期：2026-10-23。结论：**PASS，仅准入登记中的下一次五技能实际 probe；不代表 P2 或任何技能通过。**

## 范围与依据

独立只读审查 `experiments/x2_skill_host.py`、直接检查 `tests/test_x2_skill_budget.py`，以及 M4 的继承构造与 deadline 接缝。以第三轮 `data/x2_sim/agibot/data/var/skills-20261023-143217-57b4dc21/host_source.py` 为已验收旧基线，核对完整 diff，并对照 [第四次准备登记](m2-actual.md) 与 [实施报告](m2-ui-budget-implementation.md)。

本次为 **S1** 准备预算修改。未发现阻止此次实际 probe 的具体问题。

## 审查结果

- CLI 与 `SkillRuntime` 构造器均仅接受 UI/总墙钟 `(80, 240)` 或 `(180, 360)` 秒。默认保持 80/240；180/360 必须显式传入；混配、其他数值和非有限值被拒绝。
- 工作截止仍为总截止减 30 秒。UI 截止取工作截止减 60 秒和本次 UI 上限之间较早者，因此延长准备窗口没有取消清理保留，也没有让启动耗时游离于总墙钟。
- `ready.json` 及等待状态包含实际 UI 单调截止；状态和 `host_summary.json` 的预算从生效的 `deadline()` 推导，保存 UI 上限、有效总预算、清理保留及工作/总截止。UI 可能受总预算收紧，报告已明确这一限制。
- 旧四参数构造调用兼容。M4 `BehaviorRuntime` 仍使用自己的 `deadline()` 覆盖，因此新增登记不会把其有效总预算误写为独立技能 runner 的 240 秒；默认 UI 上限保持 80 秒。
- 基线 diff 未修改动作参数、技能结果解释、单技能运行器、每技能 40 秒上限、启动下一技能前 41 秒余量检查、失败即停止或资源清理逻辑。未触及官方 SDK/MC/Sim。
- 定向检查覆盖两组预览/构造、无效预算拒绝、UI 受总截止收紧、状态预算以及继承覆盖。实施者记录的唯一检查为 `1 passed in 0.02s`；此次未重复运行，也未追加 build、lint、typecheck 或全量测试。

## 准入与限制

允许协调者按登记显式使用 `--ui-timeout 180 --wall-timeout 360`，在新 run 目录内执行原顺序 `HOLD ADVANCE TURN_LEFT TURN_RIGHT WAVE_RIGHT`；继续保留 `--sim-only`、`--wave-probe`、失败即停和实际关闭确认。总预算仍可能不足以覆盖全部技能，不能把未执行项算作通过。

本次没有启动 VM、容器、ROS、浏览器、完整脑、训练或真机；没有更改业务代码，没有 commit/push。前三轮证据保留，第四轮实际结果须另记；挥手取消资格仍未建立。
