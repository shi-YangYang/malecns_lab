# M3 冻结脑读出基线 r1 实测

2026-10-23。**正式结果 FAIL：触及已登记5400秒上限，未完成200次更新及最终验证选择。** 原run为 `data/x2_behavior_readout_20261023_r1`，监督会话53078已退出code1；没有延时、续写或改为成功。

- 完整计算50条训练、50条验证，共416+402=818帧特征，918份真实状态存档。
- 实际完成168次参数更新，只有167条完整更新后验证history；168号audit证明optimizer.step已完成，不能只按phase推断。没有`worker_result.json`、`selection.json`、`validation-scores.json`。
- supervisor为wall_limit，5399.765719959秒、worker returncode −15；外层总5400.137580125秒。第一个头更新phase在5255.1596秒，原预算余量不足以完成全部200次更新。
- 168号已落盘参数全部有限，theta与r2 initial逐元素相同；12个已保存checkpoint的theta也一致。无正常终态，不宣称完成最终Adam状态/全程封存审计。
- 已落盘最低验证loss为第149步3.0369427204，仅诊断候选；没有补算评分或将其部署。锁定测试未打开，P3/P4尚无通过结论。
- peak RSS为1,619,968,000 bytes，MPS current峰值1,100,497,408、driver峰值1,525,497,856，分别记录，不相加。

后续一次只读CPU审计用24.60秒核对全部818帧、918状态、37份来源及初始化身份，通过；具体hash、字段、链与证据边界见[m3-cache-recovery-readiness.md](m3-cache-recovery-readiness.md)。原缓存完整不等于原训练完成。

按[独立缓存恢复契约](m3-cache-recovery-contract.md)实施新路径：同一initial、全新Adam、固定200次重做头部训练，600秒总预算，不重算脑前向、不从诊断候选接续。实际执行仍须新工程独立验收和新run登记，原失败及科研资产保持不变。
