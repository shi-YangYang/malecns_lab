# Spec020 比较入口实施

2026-10-23；实施者 `x2_inference_implementation`。S2，仅修改原推理入口与比较证据包装，未修改 `learning/x2_frozen.py`、旧训练核心或Spec019报告。

## 接入

- CLI新增显式 `--backend optimized --baseline <已完成基线>`；默认仍为baseline，预览/帮助不加载数值模块、不创建输出。优化必须新目录，并拒绝把输出放入基线目录。
- 父进程核对基线registration绑定、summary/worker/supervisor完成和退出状态、固定checkpoint、完整policy、冻结参数声明、54项旧来源与整包快照、旧wrapper归档自身散列，以及3帧和4状态边界的JSON/数据文件散列。新旧wrapper源码不同是正常执行身份差异，不要求两次wrapper相同。新登记保存独立execution_id、实际runtime、baseline证据绑定以及包括 `x2_frozen.py` 的新来源清单。
- 原GPUPolicy按既有来源构造并恢复参数后，优化分支包装为 `FrozenInference(policy, checkpoint_id=CHECKPOINT_ID)`；保存其独立execution_identity及源码摘要，并与登记的原policy/checkpoint/新wrapper绑定。不重跑原基线，不修改原序列化state中的policy_id。
- 使用基线保存的原3张BGR帧，并与预登记投影、像素散列和样本元数据核对；预热600步、每帧500步、跨帧保留状态不变。保存原始输入/特征/logits/actions与4份完整状态。
- `comparison.json` 逐项记录3输入、4状态边界与3输出：shape、dtype、有限值、不同元素数、最大绝对误差、RMS；所有原始数组要求精确array_equal，元数据精确相同。任何不一致均FAIL，不调整容差。报告保留最早不一致状态/输出边界和数组名。
- 分别保存预热、逐帧、构造、执行包装、状态保存、比较与总耗时。计算时间只取预热+3帧；须不高于既有基线的80%才通过性能门槛。数值或性能失败先写完整报告再非零退出。单次比较不称统计性能保证或实时性。
- 共享锁、6GiB申请/1GiB系统预留、0.1秒监督、240秒总上限、MPS采样和仿真关闭预检沿用。基线哈希读取与来源准备也计入总预算；不创建optimizer或后台服务。

## 已做检查

只运行受影响入口测试文件一次：

```sh
.venv/bin/python -m pytest -q tests/test_x2_inference.py
```

**30 passed in 0.36s**。包括原guard检查、新显式CLI限制、基线目录保护、精确比较拒绝dtype/shape/微小浮点差异/NaN、完整4边界/3输出清单、最早差异、基线文件更改拒绝、20%边界及性能不达标失败。以现存Spec019真实完成记录检查元数据和文件散列绑定，确认计算时间为25.774687707889825秒，且不要求已归档旧wrapper等于当前新代码。

检查没有构造完整图、运行MPS或启动仿真。比较器小数组仅测试归档/误差汇报，不是替代网络或研究结果。通过后代码冻结，不追加检查。

## 协调者的一次真实比较

```sh
PYTHONPATH="$PWD/src" .venv-train/bin/python -m malecns_lab.experiments.x2_inference \
  --execute --backend optimized \
  --baseline data/x2_neural_inference_20261023_r1 \
  --output data/x2_neural_inference_20261023_optimized_r1
```

基线保持原样，新输出必须不存在。检查 `comparison.json`、`execution_identity.json`、`summary.json`、worker结果和资源日志。实施阶段没有真实优化速度或数值一致性结论；实际接受仍需协调者一次受监督运行与新独立验收。
