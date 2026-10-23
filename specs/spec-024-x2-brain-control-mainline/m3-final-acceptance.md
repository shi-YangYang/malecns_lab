# M3 首次完整脑探针最终准入

结论：**PASS，准入已登记的首次600秒完整脑探针**。2026-10-23，新的独立验收者 `/root/roadmap_m3_final_review`。[前次验收](m3-acceptance.md)的唯一阻断项F1已关闭。本结论仅允许协调者按[探针登记](m3-probe-contract.md)运行一次；不是完整脑数值、学习能力或M3产品验收通过。

## 限定复核

本轮只复核F1与[返工报告](m3-repair.md)，静态阅读训练模块、对应测试及JSON写入实现，并核对当前源码SHA256。前次8项轻量检查与返工报告的18项CPU检查已通过，未机械重跑；未启动脑图、MPS、VM、训练、仿真或硬件，未修改业务源码或commit/push。

- **拒绝前确实持久化。** `_audited_gradient_update` 在反向返回后先保存 `gradients.npz`，随后写入阶段为 `gradient_admission` 的 `audit.json`，再进行梯度准入判断；在 `optimizer.step()` 前再写入调用阶段。JSON使用 `allow_nan=False`，临时文件写完后替换正式文件。路径见 `learning/x2_behavior_training.py:331–369`、`learning/resource_guard.py:20–23`。
- **失败梯度可审计。** 缺失梯度明确标记 `present=false`；零值、NaN、正负Inf分别计数。非有限整体范数使用null，保留有限子集统计；存在的真实梯度、第一窗末电压梯度及更新前参数均保存在非pickle数值NPZ，并由审核记录绑定文件SHA256与数组shape/dtype。没有把缺失梯度伪造成零数组。路径见训练模块 `284–342`。
- **更新异常保留实际结果。** 审核记录区分优化器尝试、正常返回、返回步数与失败阶段。拒绝梯度、零变化及step中途部分改变后抛异常都经finally保存实际参数变化计数、更新后NPZ及hash；中途报错不会计为完整返回的一步。worker保存 `last_update_audit`，后续整体变化判断也先把统计写入结果对象。路径见训练模块 `344–389`、`497–503`、`664–692`。
- **固定探针语义不变。** GREET两窗的参数头仍要求存在且有限的零梯度、零变化；theta及行为/阶段头要求有限非零梯度和真实变化，跨窗边界梯度要求有限非零。没有为制造参数头更新而更换输入，loss、Adam配置、两窗信用范围与600秒预算保持登记值。
- **定向检查针对F1。** 新测试在梯度准入和step回调内直接读取已落盘审核与hash；覆盖缺失、零、NaN/Inf、参数mask异常、零变化、部分改变后报错及正常更新。返工报告记录 `18 passed in 1.22s`，本轮静态核对其覆盖与实现一致，未重复执行。

未发现本限定范围内仍阻断首次探针的问题。若进程被外部强制终止，只能保证读取最近已完成写入的阶段与产物，不能把未返回的反向或step视作完成；返工报告已明确此限制。

## 审查版本与结论边界

| 文件 | 当前SHA256 |
|---|---|
| `src/malecns_lab/learning/x2_behavior_training.py` | `9ed3fb3582d3c3c8c9acaced47c8ca9c63aba44e99edfcf018a626033d286ffd` |
| `tests/test_x2_behavior_training.py` | `77a001c0344fa0c8705d74e1e98e03348003e6728f5f46c9174d71dd4842027e` |
| `src/malecns_lab/learning/x2_behavior_policy.py` | `0fc9c201b07b380dab6670ca7cee66140fc79b68a91a5c7a80c3be092610eb89` |
| `src/malecns_lab/experiments/x2_behavior_training.py` | `81901d47504b3439c3692ee66182ef2b7e097fd4a9ae179848fbf715ae4e0045` |

训练模块与测试hash匹配返工报告；policy与CLI保持前次审查版本，未泛化重审已无阻断的全图/两窗checkpoint路径。首次探针仍未运行，实际梯度、参数更新、内存与耗时必须由该次运行报告；**P3/P4均未运行、未通过**，不得将本准入PASS替代正式训练、行为评估或功能干预证据。
