# M3 梯度失败证据限定返工

状态：F1代码修复与定向检查完成，等待新的独立验收者审查；首次完整脑探针仍未运行。2026-10-23，返工实施者 `/root/roadmap_m3_repair`。

## 范围与实际修改

仅处理 [m3-acceptance.md](m3-acceptance.md) 的F1。所属M3为S3，本次为单训练模块内的局部审核/归档修复，未扩大到模型或数值实现。

- `learning/x2_behavior_training.py`：将更新步骤抽为同文件的 `_audited_gradient_update`，每次更新使用新的 `gradient-audits/0001/` 目录。反向完成后，在任何梯度准入判断和 `optimizer.step()` 前写入 `audit.json` 与 `gradients.npz`。JSON明确记录梯度是否存在、是否有限、非零数、有限非零数、NaN/正Inf/负Inf数及可表达的范数；非有限张量的整体范数为null，原始值保存在非pickle NPZ，文件SHA256绑定到审核记录。
- `gradients.npz` 同时保留更新前参数，`parameters-after.npz` 保存结束时实际参数。审核记录区分反向是否返回、优化器是否被调用/正常返回、调用前后已返回步数、失败所处阶段及每个参数的实际变化数。拒绝梯度、优化器中途报错、执行后零变化均先归档再向外传播失败；中途部分改变不会被报成完整成功更新。
- 固定GREET两窗保持参数头零梯度/不更新。probe要求theta、行为头、阶段头以及跨窗边界v梯度有限且非零，预期参数确有改变；参数头的非预期梯度或变化也保留并拒绝。没有更换输入以制造非零参数梯度。
- worker最终整体参数变化统计在后续准入判断前写入结果对象；失败时保留。每次更新审核路径通过 `worker_result.json` 的 `last_update_audit` 可定位。
- `tests/test_x2_behavior_training.py`：新增7种拒绝路径及3种更新结果的CPU检查。测试明确读取准入判断/step发生前已经存在的审核和hash；覆盖缺失、零、NaN/Inf、masked异常、零变化、部分改变后报错及正常更新。

未修改policy、脑内核、感觉编码、loss、固定输入、信用范围、Adam字段或预算；未改历史run或快照。策略和CLI源码hash与前次验收一致。运行登记的来源清单自动包含本次训练源码及当前控制源码的新hash，不手改旧身份。

## 实际验证

唯一运行的定向检查（该文件因本次修改而重新检查）：

```sh
PYTHONPATH="$PWD/src" .venv-train/bin/python -m pytest -q tests/test_x2_behavior_training.py
```

结果：**18 passed in 1.22s，exit0**。小CPU参数/边界张量和archive fixture；未构造脑图，未调用MPS、仿真、硬件或完整训练。未运行全项目测试/build/typecheck，检查通过后没有重复运行。已按本次补丁核对改动范围。

## 限制与交接

这些检查只证明失败路径归档逻辑，不证明完整图存在有效跨窗梯度、能够在600秒预算内更新或学会任务。进程被外部强制终止时，只能保留最近已写的阶段记录，不能将尚未返回的step计为完成。新独立验收者只需审查F1修复和上述证据，不机械重跑已通过检查；准入结果由验收者决定。

| 文件 | 本次SHA256 |
|---|---|
| `src/malecns_lab/learning/x2_behavior_training.py` | `9ed3fb3582d3c3c8c9acaced47c8ca9c63aba44e99edfcf018a626033d286ffd` |
| `tests/test_x2_behavior_training.py` | `77a001c0344fa0c8705d74e1e98e03348003e6728f5f46c9174d71dd4842027e` |

未commit/push；未启动首次探针。
