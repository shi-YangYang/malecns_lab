# Implementation Report — 完整图伴随内核与线程基准

## Summary

按 Ready [协议](backward-kernel-benchmark.md)实现新的隔离离线入口，尚未运行完整图基准。轻量父进程登记完整资产 SHA、固定协议、运行环境、本地源码导入闭包及源码快照；受监督 worker 才导入 Torch、加载完整图并构建原 `TrainableLIF.base_weights_adjoint`。旧训练路径、线程配置和既有实验来源均未修改。

## Files Changed

- `examples/fullgraph_adjoint_benchmark.py`：父进程独占登记、同一工作区锁、固定资源监督、来源校验、完整图身份/模型构造、worker 终态与全目录证据清单。
- `examples/adjoint_benchmark_kernel.py`：worker 专用数值职责，构造一次只读共享伴随 CSR、固定输入与 180 次调用、全部 warmup/重复输出、wall/process CPU 原始时间、逐输出数值比较和独立核级候选判据。
- `tests/test_adjoint_benchmark.py`：轻量入口/来源闭包、登记与覆盖拒绝/锁释放、真实文件仅流式 SHA 登记与篡改拒绝、显式 200 维 CSR 代数 fixture 的固定调用顺序/结果/坏 warmup 检出。
- 本报告。

## Important Decisions

- 复用旧 `fullgraph_matvec_benchmark.py` 的纯 `canonical_hash` / `file_sha256`，没有改其字节、全局协议或登记函数。新数值 helper 与父入口分离，父入口导入不触发 NumPy/Torch。
- 现有 `generate_rewiring.workspace_lock` 的模块依赖会经 `rewiring_steps` 导入 NumPy，因此经协调者确认，在新入口保留短 `fcntl` contextmanager，使用完全相同 `data/.continuous-training.lock`、非阻塞排他及异常释放语义，不更改原生成器。锁覆盖目录创建、登记、受监督 worker 与父终态/manifest。
- 本地来源通过标准库 AST 递归解析静态及延迟导入，包含实际执行的 `__init__.py` 和旧 hash 工具；worker 另核实所有实际已导入的本地源码都在登记闭包中，外部库版本及线程环境另入 runtime。登记中还绑定真实完整母图文件 SHA 和完整逻辑图/原始计数摘要。
- 目录由父进程先 `mkdir(exist_ok=False)` 原子声明，再写登记；该声明在失败捕获外，避免失败处理覆盖既存目录。worker 需要登记父 PID、当前已持有的同路径锁和新的 `worker.claim`，重复运行不会覆盖证据。
- 资源固定申请 6 GiB、额外预留 1 GiB、有效上限 `min(6 GiB, available − 1 GiB)`、600 秒、0.1 秒采样。没有启动硬件或操作 Noria/Docker。
- 所有 5 个线程块和 3 个输入固定执行，单预热加 5 轮，各方法共 180 次；每个 case 保存后释放。数值比较覆盖全部预热/重复，并同时比较 SciPy/Torch 和各线程相对 Torch 4 线程，容差未放宽。负结果仍保存；资源完成、数值通过和核级候选分开，程序不选择生产后端。

## Validation

- Risk Level: **S2**，仅隔离离线实验工具及直接相关边界。
- `PYTHONPATH=src .venv-train/bin/python -m pytest tests/test_adjoint_benchmark.py -q`：最初 3 项通过，0.80 秒。数值项使用明确标注的 200 维代数 fixture，不运行脑模型；保存固定 180 次调用并对照独立 dense 乘法，另验证坏 warmup 使数值及候选失败。Torch 原有 sparse invariant/CSR beta 提示共两条，无失败。
- 随后代码审阅发现目录预检查与创建之间的覆盖处理边界，改为父进程原子声明目录；增加正向登记/篡改拒绝覆盖。仅重跑直接受影响的登记检查：`PYTHONPATH=src .venv-train/bin/python -m pytest tests/test_adjoint_benchmark.py -q -k registration`：2 项通过、2 项 deselected，0.21 秒。真实文件只流式计算 SHA，没有解码完整图数组或加载模型。
- 总计 4 个不同检查，登记修改后只重跑受影响项；已通过且代码未变的内核与 CLI 检查未重复。没有全项目测试、build、lint 或旧实验重跑。
- 工作区检查只发现本批新文件；G29、R13、原正向实验脚本和既有数据 run 没有被修改。实施者没有 commit；独立验收由协调者另派。

## Known Limitations

没有运行完整图 benchmark，尚无本次真实伴随吞吐/线程候选结论。显式小型代数 fixture 只检查实验软件，不能替代完整 MaleCNS 数值证据。固定线程顺序和有限重复的背景负载限制沿用协议；完整 autograd、真实上游梯度和两帧训练窗口等价不属于本批。已有训练没有提速或迁移。

软件独立验收和提交后，协调者在无其他数值任务时仅运行一次：

```bash
PYTHONPATH=src .venv-train/bin/python examples/fullgraph_adjoint_benchmark.py \
  --graph data/malecns_v2/full_reference.npz \
  --out data/adjoint_kernel_20261019_r1
```

资源结果位于 `result.json`，数值及候选位于 `worker_result.json`，每次调用位于 `timings.json`；各 `threads_<线程>_<输入>.npz` 第 0 行是 warmup，其后依次为 5 次重复。`manifest.json` 绑定源快照、登记、worker/父结果、资源 CSV、输入输出和日志。需要新的独立实际审查，不以单个布尔值替代证据。

## Remaining Questions

无阻止本批独立软件验收的问题；完整图实测结果待后续步骤取得。
