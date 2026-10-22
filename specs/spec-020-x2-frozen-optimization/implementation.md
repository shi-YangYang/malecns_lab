# 完整脑冻结推理优化实测

2026-10-23，风险S2。wrapper与入口实施见 [implementation-code.md](implementation-code.md)、[implementation-integration.md](implementation-integration.md)，独立结论见 [acceptance.md](acceptance.md)。

## 修改与检查

新增独立 `learning/x2_frozen.py`，原完整图加载器、训练源码、旧身份、参数和计算表达式保持不变。每帧预计算不变项，逐步数值错误在GPU sticky标志累计、每50步读回，跳过no_grad下的checkpoint包装。保留原稀疏算子、延迟队列和输出累加顺序；不裁节点或边，不删除反向矩阵，不建立optimizer。新后端执行身份与旧policy身份分开。

同一推理入口新增显式 `--backend optimized --baseline`，鉴别已完成的基线及其输入/状态/输出证据，使用新目录保存优化结果与误差报告。默认仍只预览；基线不会重跑或被覆盖。任一数值差异或降幅不足20%均失败退出。

实际检查仅两份受影响文件：`tests/test_x2_frozen.py` 一次27 passed in 0.67s，包含CPU/MPS小张量瞬态异常；`tests/test_x2_inference.py` 修改后一次30 passed in 0.36s，覆盖比较输入/状态/输出及速度门槛。没有全仓检查或重复已通过的基线。

## 唯一完整图对照

```bash
PYTHONPATH="$PWD/src" .venv-train/bin/python -m malecns_lab.experiments.x2_inference --execute --backend optimized --baseline data/x2_neural_inference_20261023_r1 --output data/x2_neural_inference_20261023_optimized_r1
```

退出0，`comparison.json`为PASS。三个输入文件及内容一致；预热后与三帧后的全部v/g/refractory_until/18槽pending/drive phase/event counts、元数据时钟，以及三组features/logits/actions均精确一致；所有比较的不同元素数、最大绝对误差、RMS误差为0。固定最终checkpoint、54项旧来源、完整图身份匹配，参数与原始边计数前后未变。

| 同步后计算墙钟 | 原基线 | 优化 |
|---|---:|---:|
| 600步预热 | 7.2759秒 | 2.8164秒 |
| 2.0m输入 | 6.5410秒 | 2.2967秒 |
| 1.5m输入 | 6.0110秒 | 2.2895秒 |
| 1.0m输入 | 5.9469秒 | 2.3055秒 |
| 上述计算合计 | 25.7747秒 | 9.7080秒 |

合计减少 **62.3351%**（约2.65倍计算速度），超过事前20%门槛。优化整次父进程14.1460秒，其中构造/恢复2.6970秒、比较0.0826秒；原基线整次30.4477秒。数据来自一次顺序对照，不当作重复统计基准或实时性保证。每帧仍约2.3秒，不能宣称10Hz或60FPS神经控制。

## 资源、来源与退出

优化登记ID `7d91f0f93c0611a9e2b6f04a5668940259d6647892e5d7b4e63d2252d024158e`，运行执行ID `82cbc1c194927dfdea2c8dbfccab863601da4628df25b5e7f67c33cad90c52b1`；wrapper SHA256 `3e291b4d1abe12c1a26deaac7daa3b41623910f235caa6f4b490472756f73ba4`。实际wrapper身份与关联旧policy/图/checkpoint记录在运行目录，不伪装旧训练续跑。

申请/有效RSS预算6GiB，0.1秒监督；采样峰值1657782272 bytes（约1.54GiB），最低系统available约4.90GiB，高于1GiB预留。MPS current峰值1070658304、driver峰值1149992960 bytes，138次采样、无错误；RSS与MPS口径重叠不可相加。总墙钟未超过240秒，worker退出0，最终资源样本树RSS=0。采样和有界终止不是操作系统硬隔离。

原始目录 `data/x2_neural_inference_20261023_optimized_r1/` 保存完整新来源、登记、执行身份、基线绑定、三帧/四边界、参数/图检查、timing、comparison与资源/退出记录；基线目录未覆盖。仅读取已有JSON/CSV撰写本报告，没有再次运行比较或前向。

GPU仿真VM与CPU容器预检均停止，本阶段未启动；没有训练、真机、操作Noria、重启Docker或push。此阶段只完成冻结推理优化，完整MaleCNS到X2的实际反馈闭环仍是下一阶段。
