# 代码实施记录

日期：2026-10-23。实施者：x2_adapter_implementation。风险等级 S2（局部控制状态机、ROS 反馈/发送与退出流程耦合）。代码已冻结交协调者开展一次相关仿真集成；本记录只证明离线检查，不证明仿真位移或真机表现。

## 修改

- `src/malecns_lab/control/x2.py`：标准库控制核心，注入 monotonic 时钟与发布回调。构造、未 arm 的 tick 不发布。只接受 STOP 或固定 `(0.2,0,0)`；BodyVelocity 六轴严格检查，没有改旧契约。意图 TTL 和从首次前进开始的 5 秒上限独立生效；停止未确认前不能重新申请绕过上限。
- 前进门控为新鲜 odom（0.25 秒）/MC（0.5 秒）、MC RUNNING、STAND/LOCOMOTION、body_status=1、自定义输入源匹配，以及有效四元数和倾斜不超过 0.5 rad。未来、相同、倒序、非有限接收时间和非法数值拒绝。反馈断流、短暂失控或倾斜在接收时或 tick 时清掉意图，随后反馈恢复不会自动续走。
- 停止持续发布零。只用停止请求后的 odom/MC 独立序列推进计算连续证据；要求至少 1 秒 STAND、直立、水平速度小于 0.02 m/s，位置逐段和累计变化作交叉核对（速度窗口容差 5 mm）。旧样本 tick、LOCOMOTION 暂时低速、数据间隙或位置异常不能直接确认完成。发送例外立即尝试零，并保留 failure，即使后来观测停止也不能把运行报为成功。
- `src/malecns_lab/experiments/x2.py`：默认预览和 help 不加载 ROS/控制模块、不注册、不动作。执行须明确 `--execute --sim-only`、domain47 和 localhost1。只注册 `malecns.x2_mvp`，priority40、timeout1000；ADD 有确定失败响应才尝试同名 ENABLE，不更改其他源。先发零等待本源反馈，不自动 SD、LD、GUI reset 或抢占其他源的配置。
- 入口每 20 ms 周期内最多 8 ms/64 次 spin，订阅 depth=1 防止积压旧样本；使用核心同一意图截止时刻，在正常终点先请求 STOP。SIGINT/TERM 设置退出意图后进入有界零速/反馈阶段，不立即销毁 ROS。默认启动等待 12 秒、前进 2 秒、零速停止等待 12 秒。
- 证据写入失败不再传播打断停止流程：锁存失败并继续发送零，最终 success=false；summary 写失败与资源清理分别处理。
- `events.jsonl` 保存所有本进程收到的原始 odom/MC、接收 monotonic/wall 时间、接收有效性、注册响应、每次 tick 的实际发布尝试/结果、意图与状态/原因。QoS depth=1 表示这些是收到的样本，不保证保存模拟器发出的每个样本。`summary.json` 区分意图接受、成功发布前进、观测停止和失败原因，净位移仅作摘要。

## 已执行检查

只运行本次新文件：

```text
.venv/bin/python -m pytest -q tests/test_x2_motion.py
```

首次：54 passed in 0.11s。协调者静态审查指出回调排队与意图截止竞争；实施者另发现同 tick 前恢复可能掩盖瞬时失控/长间隙，限定修复并增加对应用例。修改后同一检查：59 passed in 0.13s。随后协调者要求修复明确的日志写失败退出路径：增加证据失败锁存、继续归零、summary/清理失败隔离和两项最小故障注入检查；最终同一检查：**61 passed in 0.13s**。

该文件含无 ROS 的默认 CLI/help 与执行隔离条件检查，因此未再单独重复 CLI 检查。没有执行全仓测试、build、lint 或 typecheck；没有启动 VM、Docker、ROS、UI、真机、训练。没有改官方包、依赖、已有用户修改，也没有 commit/push。

## 协调者集成调用

在本轮忽略目录内复制现有 `malecns_lab/__init__.py`、整个 `control/`、`experiments/__init__.py` 与新的 `experiments/x2.py`，使目录结构为 `<run>/src/malecns_lab/...`。`control/__init__.py` 仍沿用现有 NumPy/legacy 导入，容器需保留其已有 NumPy；未安装整个科研包。新增模块没有 Python 3.11 专属语法，ROS Humble Python 3.10 的实际运行由协调者集成验证。

先由协调者按已有官方过程准备站立，再运行（`evidence` 目录必须不存在，避免覆盖证据）：

```sh
PYTHONPATH=/agibot/data/var/<run>/src ROS_DOMAIN_ID=47 ROS_LOCALHOST_ONLY=1 \
python3 -m malecns_lab.experiments.x2 \
  --execute --sim-only --forward-seconds 2 \
  --startup-timeout 12 --stop-timeout 12 \
  --output /agibot/data/var/<run>/evidence
```

成功条件是已经实际成功发布前进、停止观测确认且没有任何保留失败；否则非零 exit。是否达到 0.2 米前进/直立等仿真验收，还必须由协调者从原始轨迹和画面独立核对，不能仅据 exit/summary 的 success 判断。

## 限制

- 反馈年龄是本进程接收 monotonic 年龄；ROS header 原样留存，不宣称同一物理时钟。未做断流、DDS 或外部输入源竞争的真实仿真注入试验，当前故障证据来自假时钟/假发送定向检查。
- 阈值、四元数倾斜与 MC 标志都是当前仿真监测判据，不是硬件保证。MC 目标仍不是实测速度；没有接入 MaleCNS。
- 循环 deadline 不构成底层 DDS/文件系统永久阻塞时的硬实时保证。日志仍同步 flush；明确抛出的写入/flush异常现在锁存 evidence_write_error，只提示一次 stderr，停止循环继续，运行结果为失败。summary 写失败不抛出跳过清理；每项节点销毁、ROS shutdown、信号恢复和日志关闭独立尝试。磁盘故障时可能无法保留完整原始证据，不能宣称验收成功。
- 本报告不替代协调者的仿真实施结果与新独立验收。
