# P3c2独立冻结脑首两窗实测 — 2026-10-19

## 范围与登记

按已Ready且软件验收的[P3c2连续训练协议](p3c-training-runner.md)，在新目录`data/neural_training_frozen_20261019_r1`从共同原始初值启动独立`frozen_brain`训练。保持完整MaleCNS脑动态，仅更新读出；没有复制joint训练后的参数或神经状态。这里是正式冻结脑训练的唯一前缀，后续从该前缀继续，不重跑首窗。

首次命令：

```bash
PYTHONPATH=src .venv-train/bin/python -m cnsbridge.learning.train_snn \
  --graph data/malecns_v2/full_reference.npz \
  --dataset data/neural_teacher_20261019_r1 \
  --out data/neural_training_frozen_20261019_r1 \
  --mode frozen_brain --windows 1
```

第一进程终态收取后，在独立新进程执行：

```bash
PYTHONPATH=src .venv-train/bin/python -m cnsbridge.learning.train_snn \
  --run data/neural_training_frozen_20261019_r1 --windows 1
```

session20537和99503均已收取exit0；两个段的父复核均为`completed / registered_windows_verified`，监督和worker均completed。两次之间串行执行了新研究工厂的小图软件检查，没有并行完整图或数值测试，也未操作Noria/Docker。

## 身份与完整图

- run ID：`6e32b92e5923c00552e9dfb33e0d51915b4cbc6c7bf313570c2ef124aae27938`。
- 与joint共用protocol ID：`bade34b291333fa86129ab081319a5798ffb2bad10625142db6cd5294eddf412`。
- dataset ID：`f0fe9735abbffff6135a8a5d7c8f3943d638da90bfe6a40eb21ba53583137076`；manifest SHA：`2418141cc7711c64cfc27f7bd8056150b8a170dd0ebed3a3079c25bffecb1104`。
- policy ID：`2e2945c5c9925e7d05698912a034bb34a815ae2e52e295317ad7732d1d910f09`；29个训练绑定源沿用实际登记。
- 完整母图文件SHA：`98f83158ed12f0c836fb085c2e286e54f9458630c2a158bee3164af52648f6f3`。166,700节点、25,582,938边、124,177,617突触计数；原始有序计数SHA在两个段前后均为`4ce30eec9ecc64c7bb2723f9dee084b40cea08793375f40ab71cacc2592b0565`。

## 两个真实更新与跨进程状态

| 窗口 | 帧 | 脑/感觉step | 更新前动作MSE | theta改变数 | 读出权重/偏置改变数 | 两帧前向s | 反向/裁剪s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | combined 0–1 | 600→1600 | 0.5026592612 | 0 | 7884 / 3 | 20.081258 | 0.015143 |
| 2 | combined 2–3 | 1600→2600 | 0.0639842227 | 0 | 7884 / 3 | 19.894198 | 0.014818 |

第一窗只在新场景开始时进行600步空白预处理。第二进程的恢复快照引用第一checkpoint，继续frame2和step1600，没有重复预处理。theta保持原始零值、gain保持1；优化器只含读出权重/偏置，Adam步1→2，没有theta优化器槽。父程序复核真实数组、梯度、裁剪、Adam和参数变化及状态/游标链，而不是仅检查日志文字。

与已存joint第一窗的事件元数据比较，第一窗起始和结束完整state manifest精确相同，动作范围、读出梯度裁剪和读出更新统计相同。这符合共同初值、相同输入和首次更新发生在前向之后的设计；实际数组独立复核另见验收报告。第二窗两种模式的脑内参数已不同，不能要求其动作相同或把单窗差异解释为学习收益。

## 资源实测

每段申请6GiB，额外预留1,073,741,824 bytes，有效RSS为min(6GiB,available−reserve)，墙钟600秒、采样0.1秒。

| 段 | 启动available bytes | 有效RSS上限 bytes | 采样峰值RSS bytes | 最低available bytes | 样本数 | 监督s | worker s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4,891,148,288 | 3,817,406,464 | 1,526,218,752 | 4,021,288,960 | 346 | 36.745026 | 35.541485 |
| 2 | 6,113,378,304 | 5,039,636,480 | 1,785,577,472 | 5,348,442,112 | 229 | 24.411172 | 23.422541 |

均未触发资源停止。冻结脑不反传内部参数，本次两窗成本低于joint；这些是实际观测，不能保证完整epoch成本或设备实时性。

## 可恢复前缀与结论

首档`5141e8a40af41127d2035a8b07558466f1e9b52821029c54b24e5dfe3b723edd`；末档`48d2a7c54810f606a3ccab3a44e3c970960c34bc0dfcd24f5fa19fb3e8fecd55`，路径`segments/000002/windows/000002/checkpoint`。前缀SHA为`d49c3340211cefbf0bae1fcad307b0b1b6ee9fd18a53dfbc4e459f98ca535b8b`，游标combined next_frame4、completed_windows=2、optimizer_steps=2、epoch_complete=false。

已完成4帧/2窗，冻结脑还余158窗。首两窗已获新的[独立实际PASS](acceptance-p3c-frozen-run-r1.md)，完整报告已读取处理，无返工项；可以显式续同run最多四窗；不以两帧损失变化声称学习成功、不将本对照替代joint内部训练。完整joint/frozen终档、闭环、拓扑比较与部署/实飞仍未完成，AC3–AC6保持开放。
