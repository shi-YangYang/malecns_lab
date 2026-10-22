# 空输入源启动限定返工

日期：2026-10-23。实施者：x2_adapter_bootstrap。风险等级 S2：控制状态机、ROS 启动门控和事件证据之间的局部接口修改。代码已冻结，交协调者做一次有依据的集成复验，再交新独立验收者审查。本记录不声称仿真复验已通过。

## 原因与范围

协调者首轮实际集成确认 ADD 成功，但持续零指令没有令 MC 当前输入源从空值变为本源，未发布非零。随包官方仲裁说明要求有效非零才能切换输入源。按 Spec 018 新增的“集成后限定修正”，仅增加一次、至多 0.5 秒的空源 CLAIMING；保留首轮失败证据和原源码快照，不改官方 SDK/MC/Sim。

仅修改三个授权文件：`src/malecns_lab/control/x2.py`、`src/malecns_lab/experiments/x2.py`、`tests/test_x2_motion.py`，并新增本报告。没有改已有动作契约、依赖、其他业务文件或历史报告。

## 实现

- 纯核心默认 `allow_empty_source_claim=False`，原先必须匹配本源的门控保持有效。启用此选项后仍须由传输层在注册成功后明确调用 `confirm_source_registration()`，还须已 arm、有本轮显式前进请求、当前源为空、STAND/RUNNING、直立和两路新鲜反馈，才可进入 CLAIMING。注册与就绪本身不会触发前进。
- CLAIMING 使用同一固定目标 `(0.2, 0, 0)`，期限从请求时起 0.5 秒。请求时间与 MC 序列均留存，只有之后的新 MC 样本报告本源且其余门控通过，才转 FORWARD。已有本源可以沿原流程直接 FORWARD；非空其他源始终拒绝。
- CLAIMING 的时间计入原意图 TTL 和从首个请求起的 5 秒上限，续期不能延长 0.5 秒期限，也不能在意图已经过期后将其复活。核心实例只允许一次空源 CLAIMING；失败后，即使反馈恢复或后来停止确认，也不再次自动申请空源。
- 竞争源出现、空源时离开 STAND、姿态异常、反馈间隙/过期、无效数据、过期或超时，均清掉意图并在持续 tick 时发送零；瞬时异常在接收回调中锁存，不能靠同 tick 前恢复掩盖。原有发布异常、停止证据、信号退出、证据写入故障处理保留。
- runner 仅在注册成功后设置注册标记，启动等待使用明确允许空源候选的检查，前进循环接受 CLAIMING 和 FORWARD，并沿用同一个意图截止时刻。新 `ownership_ack` 事件记录本源 ACK；MC 事件包含接收前后状态，tick 包含 claim_deadline 和 ownership_confirmed_at，summary 包含确认时间。CLAIMING 实际非零发布仍记为 forward_started，最终失败与停止确认独立记录。

## 已执行检查

仅执行一次：

```text
.venv/bin/python -m pytest -q tests/test_x2_motion.py
85 passed in 0.13s
```

新增定向检查覆盖：默认禁止空源、显式注册/arm/反馈/状态要求、注册本身不动作、请求后新 ACK、旧样本拒绝、超时、竞争源、姿态和间隙故障、恢复不重试、续期不得延长 claim 期限、过期 TTL 不可复活及 CLAIMING 计入 5 秒上限。原先默认 CLI/help 不加载 ROS、停止确认和证据故障检查一并通过。本文件的检查已包含 CLI 默认与隔离条件，未另外重复运行。

未运行全仓测试/build/lint/typecheck，未启动 VM、Docker、ROS、浏览器或硬件，未执行训练、commit 或 push。没有重新运行已通过的检查。

## 集成命令与限制

CLI 参数保持不变。复制冻结源码到新的 run/src，不覆盖旧 run；沿用完整 control 目录和必要包入口。**先加载现有 ROS/AimDK 环境，再 prepend 源码路径，保留 ROS 原有 PYTHONPATH**。此前 implementation-code.md 的命令直接替换 PYTHONPATH，不能照搬；历史记录保留，本次正确调用为：

```sh
source /opt/ros/humble/setup.bash
source /workspace/aimdk/install/local_setup.bash
PYTHONPATH="/agibot/data/var/<new-run>/src${PYTHONPATH:+:${PYTHONPATH}}" \
ROS_DOMAIN_ID=47 ROS_LOCALHOST_ONLY=1 \
python3 -m malecns_lab.experiments.x2 \
  --execute --sim-only --forward-seconds 2 \
  --startup-timeout 12 --stop-timeout 12 \
  --output /agibot/data/var/<new-run>/evidence
```

该代码的本源 ACK、真实位移、直立与最终停止仍待协调者实测；离线 85 项检查不替代集成证据。新的逻辑使用 Python 3.10 兼容语法，没有安装依赖或新增 ROS 宿主要求。本轮没有改变原有接收时间新鲜度、DDS/文件永久阻塞和仿真阈值不等于硬件保证等限制。
