# Spec019 代码实施

2026-10-23；实施者 `x2_inference_implementation`。风险S2；仅完成包装、输入投影与无完整脑检查，真实基线由协调者执行。

## 修改

- `experiments/x2_inference.py`：默认预览和帮助只用stdlib，不加载NumPy/Torch/ROS，不创建目录。`--execute --output` 才申请新目录、共享数值锁及监督。无后台常驻、无重试、无模型回退。
- `learning/x2_inference.py`：固定最终run/checkpoint、CPU policy身份、窗口seal、完整参考图规模和600/500步时钟；从Git `732ddc0` 导出完整旧 `src/cnsbridge` 包，包含init、依赖及配置，核对全部54项登记源码散列，快照设为只读。保存当前wrapper来源与历史元数据，不改旧路径或历史身份；worker只允许新的快照命名空间，并记录实际runtime与旧runtime差异。构造后的policy必须与原完整身份相等。
- 同一包装只读取NPZ中的theta/readout_weight/readout_bias，核对整个NPZ的SHA256、条目清单、三个参数的shape/dtype/有限值；旧state及Adam不加载。原校验要求保留参数requires_grad=True；通过 `torch.no_grad()`、不构造optimizer、前后参数逐字节摘要相同且无grad验证冻结。
- 新场景600步预热后连续处理三个输入；保存预热及每帧完整脑/感觉状态共4份，原始BGR帧、logits、三轴动作、features、状态时钟与散列。各段计时在MPS同步后读单调时钟，神经时钟与墙钟分开。
- 父进程持有 `data/.continuous-training.lock`，worker验证父PID、锁占用及独占claim后才导入Torch；复用resource_guard，申请6GiB、实际min(6GiB,available−1GiB)，0.1秒检查，240秒总预算扣除准备耗时。另用既有MPSMemorySampler设分配限制并采样，不把MPS与RSS相加。失败保留阶段、日志、资源/结果，非零退出。
- 仿真预检仅调用独立runtime的 `lighter status`、检查GPU socket不存在，以及显式 `desktop-linux` context下CPU仿真容器为exited；不启停服务、不改变默认context。
- `senses/x2_marker.py`：世界y/z平面红色方标按4个角投影到160×120 BGR；接受有限位置及单位xyzw四元数，不包含运动模型。2/1.5/1m固定离线探针明确标记来源，不能称真实相机或X2反馈。

## 实际检查

只运行新文件的无完整脑检查：

```sh
.venv/bin/python -m pytest -q tests/test_x2_inference.py
```

第一轮20通过、1失败：真实 `git archive` 包含合法顶层 `src/` 目录条目，严格包前缀检查误拒。仅允许该父目录条目后，同一文件第二轮 **21 passed in 0.19s**，随后停止检查。

覆盖默认入口/帮助无重模块、显式新目录、世界姿态到像素的因果变化、输入拒绝、参数文件校验与只读3参数、单进程锁、伪独立worker拒绝、预检只读及失败保留、实际资源预算传入监督、源码路径/更改拒绝。另用本地真实JSON验证固定checkpoint和窗口身份，并从真实Git导出核对54项来源；未构造图、模型、MPS或ROS。测试小数组仅为序列化检查，不是替代网络或研究结果。

## 协调者执行命令

保持GPU VM和CPU仿真停止，输出目录必须尚不存在：

```sh
PYTHONPATH="$PWD/src" .venv-train/bin/python -m malecns_lab.experiments.x2_inference \
  --execute --output data/x2_neural_inference_20261023_r1
```

入口会自行做只读预检、锁与资源判定；无需另开完整脑。结果看 `summary.json`、`result.json`、`worker_result.json`、`timing.json`、`resources.csv`、`mps_memory.csv`、`states/` 与 `frames/`。若失败，保留目录且不自动重跑。

## 限制

本实施没有运行真实完整脑，因而没有性能、输出行为或MPS兼容性结论。父进程限额是0.1秒采样监督而非操作系统硬隔离；终止有既有监督器的退出宽限。预检为启动时状态观测，不锁定外部仿真管理工具。当前仍为静态输入基线，无X2动作或闭环；不证明实时性、目标跟随或跨设备泛化。
