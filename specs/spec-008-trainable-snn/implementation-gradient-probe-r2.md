# Implementation Report — 梯度诊断 r2 重放对照

## Summary

按 [r2 Spec](gradient-probe-r2.md) 扩展原诊断软件，固定旧159→160完整档及同两帧输入。单worker顺序执行两次原生产 `policy.forward_tensor` 无hook完整范围重放，再执行500/250/100步带观察器的短信用分支，最后保留固定特征logit纠错检查。无optimizer更新。r1失败产物、原46源的历史身份与提交a0dcb32均保留，r1不重标为PASS。

## Files Changed

- `src/cnsbridge/learning/gradient_probe_math.py`：显式 `observed=False` 时真正调用原生产 `policy.forward_tensor`，不创建观察器、不调用 `observed_frame`；报告保存路径和观察标记。
- `src/cnsbridge/learning/gradient_probe_evidence.py`：v2协议、五分支清单、固定r1文件SHA/原Git提交来源、分层theta比较、重放分类及对应保存证据复核。
- `src/cnsbridge/learning/gradient_probe_worker.py`：每支仍恢复同一完整父档，串行五支并保存全部证据，第二次参考重放后封存分类；失败立即保留现场退出。
- `src/cnsbridge/learning/gradient_probe.py`：父结果记录五个实际完成分支。
- `tests/test_gradient_probe_r2.py`：新增3个直接相关检查。
- 本报告。

原42份绑定训练源未改；GPU感觉/Adam模块、原r1证据、旧完整图与所有旧训练档均未修改。

## Important Decisions

### 原生产路径及相同目标

无hook两支的前向直接调用生产方法，不用空观察器冒充原路径。短分支继续使用已经验收的50步checkpoint闭包与状态detach实现，全部1000步前向、DN逐步累加和原两帧动作MSE＋gain正则保持。无hook分支不存在边界伴随记录，报告明确空清单；短分支仍必须有20个登记边界。

### r1历史绑定

固定 `data/mps_gradient_probe_20261019_r1` 的registration、1000步values/report、worker结果和父结果共5个文件SHA；r1注册仍须保持failed。其新诊断4源与Git提交 `a0dcb3283d371339abf344278ab41c81b1da4100` 的原文件字节比对；旧42训练源仍与历史注册一致。当前r2四源绑定新注册，不要求它们等于r1。旧159/160来源锚必须与r1锚相同。

### 比较和解释

动作、特征、读出梯度、全部末态数组和状态元数据仍精确比较。两次完整范围theta与旧160、r1及彼此比较时，要求全有限、相同零支持、相同非零符号，并满足既有 `tests/test_mps_continuation.py` 的 `rtol=1e-5, atol=1e-7`。报告分别保存exact、不同项数、maxabs、非零参考项maxrel、最大ULP/ULP差异项数、相对L2和支持/符号差异，数值通过不会称逐位相同。

短范围theta仍保存同样差异统计，但不以近似原theta作为通过要求。所有数值必须有限。分类按Spec优先顺序：两次无hook不同→普通重放末位变化已观察；两次相同但与r1不同→观察版本或运行环境相关；都匹配r1但与旧160不同→旧档差异未定位；全等则如实记录全等。任一强制数值比较失败都不作原因分类。所有分类明确不证明特定硬件归约是原因，r1严格失败保持。

## Validation

- Risk Level：**S3**，仅涉及诊断比较协议、恢复分支调用与保存证据。
- 已阅读改动diff。唯一执行的定向检查：

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src .venv-train/bin/python \
  -m pytest tests/test_gradient_probe_r2.py -q -s
```

- session **7627**，已收取exit0：**3 passed, 1 warning in2.68s**。警告来自原CPU小电路构造的PyTorch sparse CSR beta提示。
- 检查覆盖：float32单ULP可数值通过但exact=false；超tol、零支持变化、非零符号变化、非有限和错误dtype拒绝；动作/feature/读出/末态单ULP拒绝；短范围theta不强制近似；分类分支和失败不分类；实际小电路通过原MPS `forward_tensor` 两帧，观察器构造/观察前向均被故意禁用仍通过，初态/参数/Adam全部保持。
- GPU小电路检查持有 `workspace_lock()`，已结束并释放；没有完整图计算或optimizer更新。没有重复原5测试、原MPS核、全套测试、build或lint。测试后只删除一个因替换旧比较函数而闲置的import，未改变执行行为，未重复测试。

## Known Limitations

尚未执行r2完整图实际诊断，未定位历史theta末位差异，也未证明短范围可解决完整图大梯度。软件需新的独立验收；通过后由协调者在全新目录执行一次受监督诊断，资源6GiB申请、额外1GiB预留、600秒/.1秒监督及共享锁保持。独立实际验收将只读核对保存证据。实施者未提交、未操作硬件或Noria。

## Remaining Questions

本批软件范围无待决问题。r2实际结果和后续训练损失/反向范围决定属于协调者后续阶段。
