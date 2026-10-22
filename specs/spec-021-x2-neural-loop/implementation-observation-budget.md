# Spec021 r3 限定返工 — 独立观测等待预算与两轮执行

日期：2026-10-23。状态：代码冻结，定向检查通过，等待新的独立审查与有界实测。风险 S2，范围限于协议预算、宿主/ROS 登记与轮次的一致性。实施者未启动完整脑、VM、ROS、训练或动作，未 commit。

## 失败依据与范围

读取独立 [r3 FAIL](acceptance-r3.md) 后，按协调者在 [Spec](spec.md) 冻结的预算修正实施。r3 第二次姿态等待期间双流未中断，25 秒内角速度范数约 0.022→0.0061 rad/s 且仍全部高于 0.005 rad/s，四元数同步变化；旧逻辑正确拒绝不合格输入。本改动依据实际衰减证据增加全零等待预算，不能证明根因已经查明，也不保证 40 秒必然达标。

只修改 `control/x2_protocol.py`、`learning/x2_session.py`、`experiments/x2_neural.py`、`experiments/x2_bridge.py`、`tests/test_x2_neural.py`。原运动核心、姿态窗口算法、完整脑后端、参数、相机、神经门限及官方组件均不改。

## 实际改动

- `MAX_ROUNDS=2`，每轮最多 1 秒非零脉冲。宿主循环、完整帧数检查、末轮事实 ACK、宿主最终封存 ACK、桥最终接收均沿用同一个常量，第二轮停止后直接进入 `brain_seal`，不生成第三次观测。
- `POSE_TIMEOUT=40`，仍以 `min(ROS总截止-12秒, 等待起点+40秒)` 设置局部截止。连续 1 秒、角速度范数 0.005 rad/s、完整四元数变化 0.005 rad 的真实窗口条件不变；总预算不够仍失败。
- 新增 `OBSERVATION_TIMEOUT=45`，只用于 worker 收到非最终 STOPPED ACK 后等待下一观测。首次等待仍为 120 秒（含协调者启动环境）；完整脑处理、每轮停车 ACK 等待和最终封存等待继续使用原 30 秒 `BRAIN_TIMEOUT`，没有用观测预算延长脑计算。
- registration 记录 `rounds`、`brain_timeout_seconds`、`observation_timeout_seconds`、`pose_timeout_seconds`，参与现有 registration hash；worker 启动验证这些字段与当前冻结协议相符。ready 新增 `rounds`，ROS 桥在导入 ROS、创建 claim 和连接之前核对 registration/ready 轮次及预算，拒绝混用旧三轮或旧等待值。
- 两个默认 preview 都显示两轮与三个独立预算，保持只用标准库，不导入 NumPy/Torch/ROS、不加载脑和执行动作。桥的实际 configuration 事件同样记录两轮和独立预算。
- 原响应接纳新反馈、0.02 m/0.03 rad 限值、vx>0.03 神经门限、12 秒停车、host failure/竞争源/核心 sticky failure、宿主 240 秒和 ROS 120 秒上限保持原逻辑。VM 启动至关闭的外层 120 秒守卫仍由协调者执行，本次未启动环境。

## 必要检查

唯一测试命令：

```sh
.venv/bin/python -m pytest -q tests/test_x2_neural.py
```

**54 passed in 0.22s**，一次通过后停止验证。未运行旧数值、原运动子系统、全量套件、build 或 lint。

调整既有姿态超时和两轮端到端假反馈场景，确认末轮停稳事实与宿主最终 ACK 仍分开，第三观测不存在，全部 STOP 不冒充成功；默认 preview 和登记夹具核对新的预算。新增 5 个针对消息不一致的场景，分别拒绝错误 registration 轮数、脑预算、观测预算、姿态预算和 ready 轮数，均在 ROS 连接前结束。现有脑超时与姿态/抢源/资源失败保护检查继续通过。

## 冻结 SHA256

| 文件 | SHA256 |
|---|---|
| `src/malecns_lab/control/x2_protocol.py` | `1f28f25349b24989009ef7ba10b70d8c1a82ed9f017ae77c4f99572cf2eb66fa` |
| `src/malecns_lab/learning/x2_session.py` | `11662fe3f3bd025e4e3d5819521c4fc7d138df2834b4e43ec0a0a7e4bf8e69f6` |
| `src/malecns_lab/experiments/x2_neural.py` | `5aebc9c244235e81ca5594f92c0f9824d6583c19c36adf927440cce8887a246d` |
| `src/malecns_lab/experiments/x2_bridge.py` | `4700cc0d99b3175fcdc83bb814fb6f729556f0176a28d8e741b00314cc36c99f` |
| `tests/test_x2_neural.py` | `c6e7ab017c9a7f0f692de246f4a0fbd4ab5f9f119c877b7e50c11c3cfeb827da` |
| `src/malecns_lab/control/x2.py` | `e0e25488248eeb8bc76d4d116bdfd268f593dd2b8db093ead701a177800dfd8b` |

`control/x2.py` 为未改的既有运动核心，仅记录其身份。限定改动的前后 diff 保存在 `/tmp/x2-observation-budget.diff` 供本次独立审查；源码与此实施报告是持久交付。以上是软件检查，MVP 仍待新有界实测通过完整神经反馈、每轮停稳、参数与来源封存、资源和环境关闭验收。
