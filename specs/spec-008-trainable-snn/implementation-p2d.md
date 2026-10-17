# Implementation Report — P2d 配对可观测性工具

## Summary

实现完整图生产入口的预登记、资源监督、两动力学配对采集、原始产物、独立轻量校验和可读报告。本批只运行显式数值夹具，没有加载完整 MaleCNS、运行P1、训练模型或连接设备。候选实验时间仍等待P1成本测量后冻结。

## Files Changed

- `src/cnsbridge/diagnostics/observability.py`：CLI、预登记、父进程监督及完整性升级；保留独立 `supervisor.json`。
- `observability_protocol.py`：严格固定协议、可收紧资源/可预先调整整数窗口、七掩码、步骤账本与实现身份。
- `observability_collect.py`：完整图只加载一次；模型顺序构造；同一完整LIFState与SpatialDriveState分支；全部DN、L1–L5、R1与全节点计数/时序采集。
- `observability_evidence.py`：身份、原始数组、完整基态、事件递推、计数交叉核对、资源终态及汇总重算。
- `observability_summary.py`：全部配对的逐DN/桶差值、双向变化统计及Markdown解释。
- `tests/test_observability.py`：协议/预算/身份/数值/持久化/中止/摘要篡改的定向检查。
- `docs/OBSERVABILITY.md`：候选模板、运行/校验接口、产物语义与尚未完成实际实验的边界。
- 本实施报告。协调者维护的P2d规格、Plan与handoff改动保留，未替其改状态。

## Important Decisions

- 原 `brain` 核心、P1和 `resource_guard` 完全未修改；新工具有独立schema，不沿用P1内部学习成功含义。
- 生产入口只能通过 `load_training_graph` 和 `load_policy_neuron_map`。没有小图CLI、裁图或回退；数值夹具始终 `numerical_test_only`，生产身份验证明确拒绝。
- `source_identity.json` 在图/映射准备后立即持久化；每模式 `identity.json` 在长循环前保存。每产物完成即更新 `artifacts.json`，使后续进程资源终止仍可解释已保存分支。
- 完整预处理状态含v/g、不应期、18步延迟队列与时钟；分支实际起始摘要必须一致。撤除期间继续相位和神经时钟，预处理只计一次。
- 全部DN逐1ms保存整数发放计数及float64累加后的复位后v/g平均值；L群分类型、R1事件/发放和完整节点阶段计数同步保存。校验器独立执行输入相位递推，并交叉检查各群桶/阶段计数。
- `completed` 只是进程正常退出。独立核验所有条件、步数、来源/数组/摘要/资源后才产生 `protocol_complete`；零响应不是完整性失败。保留未阈值化差异及增加/减少细胞数，不宣称生理/动作成功。
- 实现hash覆盖新模块、trainable/graph/tonic/policy_groups、reference/connectome/metadata、spatial_drive、resource_guard。

## Validation

- Risk Level: **S3**（跨模型状态、持久化证据与资源失败语义）。只运行新增相关测试，无全套、无完整图、无硬件。
- 首轮：`PYTHONPATH=src .venv-train/bin/python -m pytest -q tests/test_observability.py` → **38 passed**, 1.33s。
- 增加逐分支持久hash后，仅运行 `-k completed_branch_hash` → **1 passed / 38 deselected**, 1.21s。
- 增加中止时身份持久化后，该直接受影响检查重跑 → **1 passed / 38 deselected**, 1.65s。
- 将模式产物与摘要验证提取为可直接检查的纯证据函数，并补缺失分支、错误基态摘要、参数变化、身份错配及摘要/差值/报告损坏检查后，仅运行 `-k 'artifact_contract or invalid_persisted_mode or saved_summary'` → **8 passed / 39 deselected**, 1.20s。
- 合计47个不同检查有通过记录；没有无理由重复首轮全文件。PyTorch仅发出已有CSR beta warning，没有失败。
- 数值夹具为明确199节点软件电路：R1驱动、5个L群细胞及两个断开的DN观察细胞；不是参考脑。独立解析预期覆盖8mV背景首次发放约41.6ms，独立输入递推覆盖全部条件，完整零DN差异的公共产物校验正常通过。
- 父入口/校验器不导入Torch、不加载训练图的子进程import检查包含在首轮测试中；模板非执行、新文件保护和预检不启动worker亦有覆盖。

## Known Limitations

尚无完整生产图运行或 `protocol_complete` 研究结果；通过的是软件与明确数值夹具。生产完整图的耗时、内存、视觉到DN响应仍未知，不能由夹具推广。实际完整图运行保持等待用户内存释放通知及P1成本测量。

来源/产物hash校验用于检查保存证据一致性，不承诺识别任意恶意伪造。当前源码不匹配时旧证据校验拒绝，需使用登记对应版本。资源控制为采样监督，不是OS硬内存隔离。

## Remaining Questions

本软件子批无待用户确认的实施问题。实际协议窗口/预算在P1成本证据后预登记；独立验收由协调者另行安排。实施者未commit、未更新规格为Done、未声称最终验收通过。
