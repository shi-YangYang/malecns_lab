# P1a — 完整图探针工具的软件验收

状态：Done，返工后的新独立验收 PASS。从 [P1 完整协议](p1-fullgraph-probe.md)拆出可单独审查和提交的软件步骤，不降低 P1 的实际更新标准；原误报成功问题与 FAIL 记录保留。

## 当前证据

首轮 `data/trainable_probe_20261017_p1/` 已写登记和失败终态，启动前 available=3,674,931,200 bytes，低于4GiB预留，未启动完整图计算。用户表示先释放内存，完成后通知；此时不自动重试。

## 范围

审查 `learning/checkpoint.py`、`resource_guard.py`、`probe.py`、`tests/test_training_probe.py` 及 `docs/TRAINABLE_SNN.md` 的新协议部分。软件要求继承 P1：安全 JSON/NPZ、严格来源与参数身份、全阶段父监督、登记先于计算、完整失败终态、固定输入/目标/优化/恢复协议，无输出或输入可训练旁路。

验收者独立核对实现和已有16项定向检查证据；不重复已通过测试，不运行全图或硬件。发现具体覆盖缺口时只做必要定向检查并保留证据。

- [x] checkpoint 和探针工具符合已登记的软件契约，生产路径不能替换完整参考图。
- [x] 资源预检/超时/RSS/异常/不完整证据不会误报成功，运行覆盖载图至保存与恢复。
- [x] 首轮原始文件与报告一致；缺少实际更新的边界明确。

输出 `acceptance-p1-tools.md` PASS/FAIL。工具 PASS 可作为独立一步 commit，提交说明不得声称完整图已训练；P1 和总 Spec AC2 继续未完成，必须用后续实际全图结果另行验收。
