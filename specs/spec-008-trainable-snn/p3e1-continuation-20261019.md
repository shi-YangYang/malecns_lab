# P3e1 正式完整重连图续段实测 — 2026-10-19

状态：首两段[独立实际PASS](acceptance-p3e1-run-r1.md)并提交5410687之后，按[冻结协议](p3e1-bounded-rewiring-draft.md)显式推进同一唯一前缀。本文是已有父批准结果的S0整理，不重复生成或软件检查；后续段只追加，不覆盖此前证据。

目录 `data/topology_generation_main_20261019_r1/`，run ID `b4eba3601105fe9ec09ed7ae272b8012cbf679303830ab7e30be4bd75e8e4d71`。完整166700节点/25582938边，seed1901、成功上限25582837/提案上限102331348、13个绑定源码与1GiB额外预留不变。每段显式最多1000000新提案，不与G训练或其他完整矩阵作业并行；Noria/Docker保持。

```bash
PYTHONPATH=src .venv-train/bin/python -m cnsbridge.learning.generate_rewiring \
  --run data/topology_generation_main_20261019_r1 --proposals 1000000
```

## Segment 3：首次1000000提案配额

session36212已收取exit0，父`completed / registered_proposals_and_terminal_evidence_verified`。累计提案1,200,000、累计成功交换624,613，本段新增520,783成功；实际恢复父CSC/RNG/计数后才推进，terminal仍false。原始resume/registration/worker/监督/assessment/数组全部保留。

| 项目 | 实测 |
|---|---:|
| 监督墙钟s | 28.065047 |
| worker墙钟s | 27.915626 |
| 启动available bytes | 5,064,654,848 |
| 有效RSS上限 bytes | 3,990,913,024 |
| 采样峰值RSS bytes | 1,620,082,688 |
| 最低available bytes | 4,867,473,408 |
| 采样数 | 264 |

段ID `f6347f9605fe3d6b4ff03737765e8eab3a8e15697efe634ff83e3efddc1a53d2`；末档 `1bee13185dae9950802a59ffeb01f8b7852401451c5545237d4eadc8becf79ea`，逻辑图SHA `4bb13996160b7ce716805035d977622cf7cbef70fd134c336acedc4d586a1696`，前缀 `feb26b1e3d34542ada14b732b2c3aaec92d99907775f895eeab2ce3f08228dbb`。累计拒绝计数 `{"different_source_sign": 567873, "duplicate": 7443, "new_self_edge": 26, "same_source": 29, "same_target": 5, "self_edge": 11}`。

本段仅测得1000000提案及中间封存的实际成本，没有执行最终路径/类型对/全节点入强度诊断或research_control资产重载，不能据此保证终段成本或宣称已生成正式最终R。生成总预算尚未结束，未计算神经任务结果；边交换数量不等于充分混合或拓扑收益。

## Segment 4：累计2200000提案

在G第15段退出后串行执行，session82164已收取exit0；父completed，实际CSC/RNG/计数沿同一前缀恢复。累计2,200,000提案/1,146,325成功，本段新增521,712次成功，terminal=false。监督29.564487s、worker29.370434s；采样峰值1,787,150,336 bytes、最低available3,475,210,240 bytes、273样本。启动available=4,182,114,304 bytes，有效RSS上限=3,108,372,480 bytes，未触发1GiB预留。

末档 `4bdf0ac51ef5a25ff33aa1fa6dd39f1dd3223893b5880a3c978850a7df4fbd07`，逻辑图SHA `8d2753f100877218cc7ff2dc5248c9c43f510374ceebcb0a284958130607e184`，前缀 `bccfb150dd9d9a29b4e55e54acf6b488713f8ebc0dcbefa5cc35112ee8aac8d0`。本段父验收与原始证据均保留，未执行最终诊断或神经任务，不变更科学预算或种子。

## Segment 5：累计3200000提案

同一run串行续段，session37731已收取exit0；父completed，累计3,200,000提案/1,666,368成功，本段新增520,043次成功，terminal=false。监督29.476552s、worker29.339227s；采样峰值1,580,777,472 bytes、最低available3,597,254,656 bytes、271样本。启动available=4,645,371,904 bytes，有效RSS上限=3,571,630,080 bytes，1GiB预留未触发。

末档 `3f1cb7730fbe4c6264af30521d1f84fc133c07c5e3c8e4018185bab45afa3332`，逻辑图SHA `b0f9634858e636dbe86319667e2a80f5c8201ed45c610b35f3493993fe01c72c`，前缀 `4655b22f42b1361542eae32bd9552024ecac014365ac412f589b91e977c6ff55`。完整原始/父批准证据保持；当前仍没有最终candidate或神经任务结果。累计拒绝计数 `{"different_source_sign": 1513137, "duplicate": 20326, "new_self_edge": 49, "same_source": 66, "same_target": 25, "self_edge": 29}`。下一段从此唯一前缀显式继续，不重跑首3200000提案。

## Segment 6：累计4200000提案

冻结脑首两窗和相关数值检查均退出后，串行续正式R，session80085已收取exit0；父completed。累计4,200,000提案/2,186,263成功，本段新增519,895成功，terminal=false。监督30.040843s、worker29.909362s；283个样本，采样峰值1,643,479,040 bytes、最低available2,723,774,464 bytes。启动available=2,784,215,040 bytes，额外预留1,073,741,824 bytes，有效RSS上限1,710,473,216 bytes；未触发停止，没有操作Noria/Docker。

末档`1ee5705737b9f46e4b78b2fed463b752c5dfca84bb52d13b0098e506871ff895`，逻辑图SHA`266e806c4bdb9b3912133e22ee1b183694eea99b589a2a27ac2ce17a3ec8e66e`，前缀`09a33f58538f4c259f8028a23ce4d58ba778268e914c7301fc8e955b19246a53`。父注册/恢复/封存链保持；没有最终candidate或神经任务结果。

## Segment 7：累计5200000提案

上段退出后显式再续1000000提案，session93764已收取exit0；父completed。累计5,200,000提案/2,705,909成功，本段新增519,646成功，terminal=false。监督30.458140s、worker30.317938s；287个样本，采样峰值1,638,006,784 bytes、最低available2,799,337,472 bytes。启动available=3,464,871,936 bytes，有效RSS上限2,391,130,112 bytes，1GiB预留未触发，Noria/Docker保持。

末档`8e5d5bb09761bf41c36dad41964905dc8c59e913ab3a186102494d46349120f7`，逻辑图SHA`50014e834b8e9dab5e5b2011607503b8221a285870b557ab671fea8ac4d0f092`，前缀`21509d18cdd7e05b09a89604344d007bdcebf776d6139d221ec3960f886b85a2`。累计拒绝计数`{"different_source_sign":2460339,"duplicate":33478,"new_self_edge":67,"same_source":110,"same_target":49,"self_edge":48}`。继续保留完整父批准证据和唯一前缀，不重跑、换种子或以中间图替代最终R；本次未执行最终诊断及神经训练。
