# Task Handoff — spec-008-trainable-snn / Implementation

## 当前断点：joint52窗、R320万提案、P3e2协议Ready（2026-10-19）

本目标轮有实际进展，不是连续阻塞。P3e1生成器软件/37项不同检查及独立软件PASS已bcb0afc提交；首两段实际门槛PASS及G48窗记录已5410687提交。所有报告全文读取处理，无返工项。1GiB额外预留、Noria/Docker限制及完整母图不变；未重复旧模型前缀。

G同joint run `data/neural_training_joint_20261019_r1` 的session10217/46895/89107均收取exit0、父registered_windows_verified，实际完成windows41–52，累计104帧/52更新；forward next_frame24，脑/感觉step12600。末档 `6bf2ba97d7c7cfa8a73325370c1bbbaefa7be5b2021850da0e11d27bbd0348c3`，前缀 `27e90dc86f28f40988198e51057787b0007bd48bfc0a30280e5e85bfc37dde0c`。第15段监督418.324132s、采样峰值2697019392 bytes、最低available2809659392 bytes；见[续训实测](../../specs/spec-008-trainable-snn/p3c-continuation-20261019.md)。前进仍约+0.3而末窗教师约+.127..+.139，不称已学会跟随。G29个绑定源码保持，joint余108窗，独立frozen尚未开始。

R正式run `data/topology_generation_main_20261019_r1` 的两段各100000提案获[独立实际PASS](../../specs/spec-008-trainable-snn/acceptance-p3e1-run-r1.md)后，已显式串行完成三个1000000提案段；session36212/82164/37731均exit0、父completed。累计3200000提案/1666368成功，末档 `3f1cb7730fbe4c6264af30521d1f84fc133c07c5e3c8e4018185bab45afa3332`，前缀 `4655b22f42b1361542eae32bd9552024ecac014365ac412f589b91e977c6ff55`；详见[续段实测](../../specs/spec-008-trainable-snn/p3e1-continuation-20261019.md)。terminal=false，没有最终candidate；R13绑定源码保持，不能将中间状态接入研究模型。

新的[P3e2研究工厂Spec](../../specs/spec-008-trainable-snn/p3e2-research-factory.md)及[Plan](../../specs/spec-008-trainable-snn/p3e2-factory-plan.md)已获新的[独立协议PASS](../../specs/spec-008-trainable-snn/acceptance-p3e2-factory-protocol.md)，全文读取处理并Ready，无返工项。[只读边界审计](../../specs/spec-008-trainable-snn/p3e2-boundary-audit.md)也已全文处理。下一步单一实施者仅新增严格终局准入、真实R counts/正向/伴随/SciPy共享缓存及graph/model/map/policy身份、共同数值契约和相关小图检查；不改29/13绑定源，不实现参数/Adam/训练档或runner，不运行完整R神经计算。软件后新独立验收与提交。既有NeuralController只检查类型，不能误报为研究角色隔离器；本批无硬件接线。

当前无运行中的完整图作业，全部本轮Agent已完成且报告均处理。可交替显式推进G下一四窗及R下一1000000提案；始终串行、不自动循环科学预算，遇具体失败保留证据后分析。编码/静态审查可与完整图作业并行，数值检查须等完整图退出。整体goal仍active，AC3–AC6及完整训练/独立冻结脑/未见场景/拓扑比较/时延/条件实飞仍未完成，不标记complete或blocked。

## 前置：joint40/160窗，首场景已结束（2026-10-19）

本目标轮有实际进展。session53960/1973/51425/19423均已收取exit0，同一冻结joint r1新增窗口27–40、累计80帧/40更新；最后脑/感觉step40600，checkpoint `e1c656b913d9c1f51d8cada6214342623afc0dd7de7e66f4b5f2240229154096`，prefix `973724345d79154e1bb33970f3fcef00c8ee7dc0e5971d24b8e9e18ff8321b20`。游标仍为combined、next_frame80，下一窗口才进入forward场景并重置神经/感觉状态，参数与Adam延续。1GiB额外预留保持、Noria/Docker未操作；最新两窗段监督206.878887s、采样峰值3217833984 bytes、最低available4022337536 bytes。各段父数值/来源复核为registered_windows_verified，记录见[续段实测](../../specs/spec-008-trainable-snn/p3c-continuation-20261019.md)。所有29个绑定训练源码保持，不重复已验证前缀。

首个80帧场景的[只读动作汇总](../../specs/spec-008-trainable-snn/p3c-first-scene-20261019.md)、PNG与新[独立验收PASS](../../specs/spec-008-trainable-snn/acceptance-p3c-first-scene.md)均已全文读取处理，无返工项。三轴归一化MSE为0.942383/0.028666/0.045410，SI RMSE为0.291229m/s、0.033862m/s、0.106548rad/s；这些是训练中变参数轨迹上的误差，不是固定策略闭环成绩。派生资产在data/neural_training_analysis_20261019_first_scene，脚本/输入/输出SHA和progress快照均保留。所有本批Agent已完成，无待处理报告或活跃作业。第26窗饱和/Adam动量的[只读诊断](../../specs/spec-008-trainable-snn/p3c-saturation-diagnosis-20261019.md)已保存，当前没有实现bug证据。首场景前进输出仍近+0.3且跟不上教师转负；参数变化不能称为学会控制，保持阴性结果与原协议，不暗改学习率或挑档。joint余120窗，独立frozen160窗未开始，实际闭环/拓扑比较与实飞均未完成。整体goal保持active，不因为阶段提交或当前输出不佳就标记complete/blocked。

## P3e：完整R成本探针已实际PASS，正式生成器协议Ready

[P3e0成本协议](../../specs/spec-008-trainable-snn/p3e-generation-probe.md)及独立协议PASS已9a6ee72提交。data/topology_generation_20261019_probe_r1仅运行一次，session24773已收取exit0，父assessment passed；完整166700/25582938/124177617/101自环保持，100000提案、51847交换、target_reached=false，约0.4044%边位置变化。生成16.017873s、重载3.310660s，监督23.072731s，采样RSS2075377664 bytes、getrusage2358689792 bytes，二者口径分别保留。[独立实际PASS](../../specs/spec-008-trainable-snn/acceptance-p3e-generation-run-r1.md)已全文读取处理、无返工项，随报告提交f905472。原始脚本/协议文件已绑定SHA，勿为更新状态而覆写历史字节；该图仅成本样本，不进入训练，不声称混合充分或拓扑收益。

[P3e1续段生成器协议](../../specs/spec-008-trainable-snn/p3e1-bounded-rewiring-draft.md)已获新的[独立协议PASS](../../specs/spec-008-trainable-snn/acceptance-p3e1-generator-protocol.md)，全文读取处理、标记Ready并提交d4b395a。仅生成器范围具备实施条件，尚未实施/运行。下一步由单一实施者按该协议提取原交换循环、封存完整CSC/PCG64/累计计数与父链，补足逐源权重多重集和实际输入/DN机械诊断，29个G训练绑定文件保持；新的独立软件验收后提交，再正式首两段各100000提案/新worker恢复实测，新独立实际验收后才可扩大到1000000提案/段。完整研究神经工厂/存档/训练比较另行冻结，不冒充本批已完成AC6。

全部完整图作业串行，不并行模型或模型数值测试，不停止Noria/Docker。后续显式推进joint剩余四窗段；P3d软件已ac2c3b9独立PASS，实际评估只能等待joint/frozen两个160窗最终父证据具备后运行。没有新的硬件许可或付费算力授权。

## 前置：首次完整图续训已通过（2026-10-19）

[P3c2首两窗实测](../../specs/spec-008-trainable-snn/p3c-run-20261019-r1.md)与新的[独立实际PASS](../../specs/spec-008-trainable-snn/acceptance-p3c-run-r1.md)已全文读取处理，无返工项。软件33162f7后，data/neural_training_joint_20261019_r1两次--windows 1实际完成；session18740/43643均收取exit0，没有存活模型作业。首窗20个r2比较字段全部exact；次窗恢复全部参数/Adam/23状态摘要及游标精确，独立NumPy复算两步全部矩/参数最大差0。神经step600→1600→2600，累计4帧/2窗/2更新，真实增益和读出均改变。

两段110.689845s/102.645659s、采样RSS2.794754/2.877975GiB；1GiB额外预留未触发，Noria/Docker保持。下一步在同一冻结run从frame4显式续段，可每次登记1–4窗，不重跑首两窗，不自动循环全部160窗；实际验收门槛已满足。原始失败/预算规则保持，AC3–AC6与整体目标仍未完成。所有本批Agent已完成，报告已处理。

## 前置：连续训练工具软件通过（2026-10-19）

[P3c2实施](../../specs/spec-008-trainable-snn/implementation-p3c-runner.md)及新的[独立软件PASS](../../specs/spec-008-trainable-snn/acceptance-p3c-runner.md)已全文读取处理，无返工项。27项定向检查通过，实施者已收取session15953的exit0；两个真实数值子进程恢复后与连续四窗全部数组精确一致。新独立验收另只读使用r2真实保存数组检查新NumPy复核器，exit0，没有重算模型或重复旧检查。旧绑定源码保持，全部本批Agent已完成。

随软件提交后，协调者按[命令](../../docs/TRAINING_RUNNER.md)在新目录data/neural_training_joint_20261019_r1先运行joint第一窗，再用新进程续第二窗，各--windows 1。不得先跑4窗或自动160窗；终态收取后新独立实际验收，再推进真实前缀。当前无活跃模型进程。启动预算以实际新登记为准，1GiB额外预留与Noria/Docker限制保持。整体AC3–AC6尚未完成。

只读闭环规划报告也已读取处理，候选保存到[p3d Draft](../../specs/spec-008-trainable-snn/p3d-closed-loop-evaluation-draft.md)，不属于P3c2软件验收或Ready协议。下一阶段可重放廉价环境、直接恢复神经状态；还需独立冻结评估格式/场景/实际门槛，不能以本草案提前运行。

## 前置：连续训练协议Ready（2026-10-19）

[P3c1实施](../../specs/spec-008-trainable-snn/implementation-p3c-state.md)及新的独立[软件PASS](../../specs/spec-008-trainable-snn/acceptance-p3c-state.md)已全文读取处理，无返工项。51项定向检查通过，实施者已收取session45861的exit0；两模式均在同进程销毁/重建后实现两次更新的全部参数/Adam/神经与感觉状态/动作/特征/游标精确一致。新增training_optimizer和training_checkpoint、使用说明与测试，已登记核心/策略/数据/旧checkpoint/P3b源码保持。真实跨进程完整图续跑未执行，不以小电路替代。实施与验收Agent均已完成，所有报告已处理，无待等待作业。

现已细化[连续训练runner协议](../../specs/spec-008-trainable-snn/p3c-training-runner.md)：run/segment/窗口封存、真实dataset与父链核对、一般步Adam浮点复算、四场景160窗一epoch的joint/独立frozen训练和首次跨进程门槛均已冻结。新的[独立协议PASS](../../specs/spec-008-trainable-snn/acceptance-p3c-runner-protocol.md)全文读取处理，无返工项；只读Adam分析也已读取并纳入协议。当前无活跃模型进程或未处理Agent报告，协议标记Ready并先提交。

下一步单一新实施者新增runner/worker/证据模块及定向检查，保持旧绑定源码；新独立软件验收并提交后，joint前两窗口分两个worker执行，实际验收确认继承完整状态后再推进登记的全场景。1GiB额外预留、Noria/Docker和完整图要求保持。P3b r2内部收益阴性保留，整体AC3–AC6仍未完成，不标记goal complete或blocked。

## 当前进行中：P3c1完整状态续训存档（2026-10-19）

[P3b r2实际报告](../../specs/spec-008-trainable-snn/p3b-run-20261019-r2.md)及新的独立[实际PASS](../../specs/spec-008-trainable-snn/acceptance-p3b-run-r2.md)已全文读取处理。软件`a088bbf`后新目录`data/joint_update_20261019_r2/`仅运行一次，终端session78116已收取exit0，没有存活模型进程。协议完整、联合数值更新、内部动作效应均true；4255个theta/453个gain和全部读出改变，完整恢复/重载精确相等，r1优化前22个字段精确一致。238.290s、RSS2.75699GiB，1GiB预留未触发；Noria/Docker保持。

关键阴性：baseline动作误差0.50265934→updated0.20494002，但acute_restore0.19793667更低，内部变化本步使误差恶化0.00700335。不得把总误差下降归功于内部收益；独立冻结脑训练和未见场景仍待执行。审查指出DN逐位变化5040、超容差4895的文案精度问题，报告已修正，无未处理返工；实际验收Agent已完成。

新的单一实施者`training_resume_impl`按Ready [P3c1续训状态规格](../../specs/spec-008-trainable-snn/p3c-resumable-state.md)新增完整神经/感觉状态、命名Adam与游标存档；只新learning模块/定向测试/说明，不改脑/策略/教师数据/旧checkpoint/P3b绑定来源，不运行完整图。实现后新独立软件验收、提交，再登记短的连续多窗口及跨进程恢复实测。当前软件工作进行中；总目标仍未完成，AC3–AC6保持未勾选。

## 当前进行中：P3b r2分组裁剪实施（2026-10-19）

稳定分组裁剪已完成[实施](../../specs/spec-008-trainable-snn/implementation-p3b-stable.md)，新的独立验收者[软件PASS](../../specs/spec-008-trainable-snn/acceptance-p3b-stable.md)已全文读取处理，无返工项。62项定向检查通过，独立核对9个历史锚点、6个原终态文件和17个保持源码相符；不重复检查。脑核心/策略/数据/参数存档源码保持，1GiB额外预留已实时核对。实施与验收Agent均已完成，无待处理报告。本状态随软件提交后，协调者在新目录`data/joint_update_20261019_r2/`运行一次完整图，再新独立实际验收；不重跑r1。Noria/Docker保持，完整目标AC3–AC6仍未完成。

## 当前断点：P3b r1数值失败，内存门槛未触发（2026-10-19）

软件`0373c87`后仅运行一次`data/joint_update_20261019_r1/`，终端session `11160`已收取退出1；没有运行中实验，不重跑。见[实际报告](../../specs/spec-008-trainable-snn/p3b-run-20261019-r1.md)：完整两帧前向和反向完成，clip_grad_norm_的float32总范数溢出，optimizer.step未执行；实际P3b为FAIL，数值更新/动作效应均未判定。墙钟110.389s、采样RSS2.94583GiB，低于本次4.15309GiB上限，系统最低可用3.82401GiB高于1GiB预留，非资源失败。全部theta梯度有限但float64范数7.78556e31，4259项非零；读出梯度范数约156.62，正则梯度零。新的[实际验收PASS](../../specs/spec-008-trainable-snn/acceptance-p3b-run-r1.md)只针对失败证据/报告完整性，已全文读取处理，无返工项；原失败保留。

只读梯度分析报告已全文读取处理：稳定全局clip系数约1.28443e−32，预测7884个读出权重均无法形成float32可表示更新，因此仅修范数不足。下一步直接实施已Ready的[稳定分组裁剪候选](../../specs/spec-008-trainable-snn/p3b-stable-group-clip.md)，新独立[协议PASS](../../specs/spec-008-trainable-snn/acceptance-p3b-stable-protocol.md)已全文读取处理。它保持100ms完整图路径和原模型，只将theta/读出分别稳定裁剪，并保存Adam矩、绑定r1基线一致性；是新的优化规则，不解释巨大梯度机制。单一新实施Agent→新软件验收→提交→新目录r2一次运行→新实际验收；不得重跑r1。全部本轮Agent已完成，无待收取报告。Noria/Docker、1GiB预留及完整目标不变，AC3–AC6未完成，不标记goal完成或blocked。

## 当前断点：P3b 软件通过，准备一次实际运行（2026-10-19）

已核实1GiB额外系统预留在代码与协议中一致，原调整已完成，不重复修改或重跑旧检查。[P3b工具](../../specs/spec-008-trainable-snn/implementation-p3b.md)由新实施者完成并经新的独立[软件PASS](../../specs/spec-008-trainable-snn/acceptance-p3b.md)，36项定向检查通过。两份报告均已全文读取处理，没有返工项；原绑定核心/策略/数据/存档源码未修改。软件随本状态提交后，协调者按[使用说明](../../docs/JOINT_PROBE.md)在新目录`data/joint_update_20261019_r1`登记一次完整图运行，再由新实际验收者复核。当前没有完整图作业，Noria/Docker保持不变，不用软件PASS替代动作学习证据。

只读规划报告已全文读取处理：下一阶段优先新增独立续训格式，保存命名Adam状态、完整PolicyState/驱动相位、下一帧游标和父存档身份；更新后detach再封存，场景内不reset。续评估可用已封存executed_action轻量重放环境，神经前缀直接恢复，避免改动现有数据绑定源码。按既有成本线性估计，80帧无梯度约780s、四场景一轮约3.8小时，均非实测，说明600s单段预算需要续跑。窗口数、训练规模和学习率须等P3b实际结果再冻结；该建议未实施，不关闭AC3–AC6。

## 当前断点：三轴联合更新的前置已齐（2026-10-19）

本轮有实际进展：持续输入完整图379.67s/RSS1.500GiB、新有状态策略a2320b0、实际响应/工具规格85716d3、P3b协议926bf34、P3a实现65b35ca。P3a首次FAIL两缺口由新实施/新验收关闭，71项定向检查通过；其[实际四场景数据](../../specs/spec-008-trainable-snn/p3a-data-run-20261019-r1.md)也已由新的[实际PASS](../../specs/spec-008-trainable-snn/acceptance-p3a-data.md)确认。全部报告已全文读取处理、所有Agent已完成、没有运行中的模型或未处理返工。

下一步直接实施已Ready的[P3b](../../specs/spec-008-trainable-snn/p3b-joint-update-pilot.md)，不要再停留在内存设置/旧诊断/软件数据检查。单一新实施Agent负责joint_probe及证据工具，软件独立验收提交后新目录登记一次实际完整图动作损失更新。数据目录`data/neural_teacher_20261019_r1/`，dataset ID `f0fe9735abbffff6135a8a5d7c8f3943d638da90bfe6a40eb21ba53583137076`，manifest SHA `2418141cc7711c64cfc27f7bd8056150b8a170dd0ebed3a3079c25bffecb1104`。四场景320帧全有效，P3b预选combined的0/1帧在0/.1s，三个标签均正；不改选。生产读取已核对登记环境来源，勿为增加信心重新采集或重跑旧实验。

P3b固定50ms/帧、两帧100ms反向、60ms无梯度空白预处理；约303s是估计，反向内存未知。保留1GiB额外预留、6GiB/600s监督和Noria/Docker；工具和实际分别验收，内部数值更新/内部动作影响/协议完整分开。完整目标仍需可恢复长场景联合训练、神经闭环、冻结脑/急性恢复/视觉/重连对照及部署/设备条件，AC3–AC6未完成。没有付费或新硬件动作，不标记goal complete/blocked。

## P3a软件完成，准备实际数据（2026-10-19）

[P3a](../../specs/spec-008-trainable-snn/p3a-data-checkpoint.md)首次验收FAIL：数据实际来源可被val轨迹重封存伪装，轻量checkpoint未校验固定gain语义。新实施者完成环境确定性重放和严格模型语义，新验收者[PASS](../../specs/spec-008-trainable-snn/acceptance-p3a-r1.md)已全文读取处理，71项受影响检查通过，原FAIL保留，无返工项。本状态随软件提交，下一步以新目录生成四个完整train场景，再独立核对实际数据；未加载完整图训练。P3b协议Ready并已提交926bf34，工具尚未实现，不跳过前置门槛。

## 下一次动作更新协议已冻结（2026-10-19）

[P3b](../../specs/spec-008-trainable-snn/p3b-joint-update-pilot.md)已获新的独立[协议PASS](../../specs/spec-008-trainable-snn/acceptance-p3b-protocol.md)，完整报告已读取处理：固定背景、50ms每帧、两帧连续100ms反向、60ms无梯度预处理；仅train-sine-combined的0/1帧同帧限幅教师目标，Adam一次联合更新，内部恢复/完整恢复/重载五分支。估计303s不是实测；内部数值更新与动作依赖性分开判定，阴性不重试。当前P3a仍实施中，P3b工具尚未实现；必须待数据/软件前置验收后再新目录登记实际运行，不把协议PASS当动作训练完成。

## 最新实际证据与进行中工作（2026-10-19）

[持续刺激实测](../../specs/spec-008-trainable-snn/sustained-observability-run-20261019-r1.md)新独立[PASS](../../specs/spec-008-trainable-snn/acceptance-sustained-actual.md)已读取处理，所有72组窗口统计和296字段前缀复核通过。379.671s/RSS1.500GiB，固定背景在持续输入21–25ms已有差异，50ms窗口覆盖方向/大小信号，零背景阴性；不再重跑。本轮P2e已提交a2320b0。按[P3a](../../specs/spec-008-trainable-snn/p3a-data-checkpoint.md)实施教师序列与完整策略参数存档，单一实施Agent仅新增文件，尚待验收；完整训练续跑状态/优化器不在该参数格式中。下一次实际动作损失更新需预登记窗口、成本和干预标准；全部目标未完成。

## 最新完成：P2e软件接口（2026-10-19）

[P2e](../../specs/spec-008-trainable-snn/p2e-neural-policy.md)实施报告和新独立[PASS](../../specs/spec-008-trainable-snn/acceptance-p2e.md)已读取处理，28项定向检查通过，无返工项，随本状态更新提交。全部DN两半窗电位、内部theta梯度、跨帧状态与controller已接好，只是软件证据，没有动作学习结论。持续刺激单次运行也已正常终止：exec56241退出0、19,400步、379.671s、RSS1.500GiB；固定背景持续期间有差异、零背景阴性，296前缀字段精确相同，实际独立验收正在进行，不重跑模型。后续要完成策略持久化及有界联合训练协议，整体AC3–AC6未完成。

## 当前子批：有状态策略接口与持续输入实测（2026-10-19）

上一轮前向优化有实际进展，所有实验已终止。本轮按[持续刺激协议](../../specs/spec-008-trainable-snn/sustained-observability-plan.md)冻结60/100/30ms、19,400步，独立[协议PASS](../../specs/spec-008-trainable-snn/acceptance-sustained-protocol.md)已读取处理；提交后新目录登记并受监督运行一次，当前不预报响应。并行实施[P2e神经策略接口](../../specs/spec-008-trainable-snn/p2e-neural-policy.md)，仅新增learning模块/测试/说明，不修改诊断绑定的脑或空间驱动源码。两者真正独立：实际持续响应决定后续训练窗口，接口要求显式窗口配置，不用未验证默认值。Noria/Docker及1GiB预留不变。

## 当前断点：前向优化已完成完整图验证（2026-10-19）

本轮有实际进展：内核基准2a3979a、实际依据与生产规格ef7a315、生产实现13716c6。软件首次验收发现证据校验仍期望旧backend，已由新实施者修复、新验收者通过；46项数值＋37项证据检查通过，原FAIL报告保留。P1 r5和配对r2随后均正常终止，新的[实际独立PASS](../../specs/spec-008-trainable-snn/acceptance-forward-csr-actual.md)已读取处理，无返工项。所有本轮Agent已完成，模型进程已退出，无需等待。

最新成本与证据见[实测报告](../../specs/spec-008-trainable-snn/forward-csr-run-20261019.md)：P1 32.391s、RSS1.753GiB，全部已保存响应/gradient/theta/gain与r4精确一致；配对108.558s、RSS1.506GiB，22个NPZ/465字段与r1精确一致。两次源身份更新正常，原始图/数据未改，1GiB额外预留与Noria/Docker限制保留。不得重跑这些已通过实验或测试来增加信心；当前前向约3.9s/20ms神经窗仍非实时。

下一步推进三轴神经策略集成，而非继续停留在数值探针：先明确跨帧保留LIFState及SpatialDriveState、场景边界reset、TBPTT detach保留队列/相位/时钟；读出选全部1314 DN的电位/时间信息，不能只用已证明各条件相同的总发放率。视觉仅进入SpatialDrive，不直连动作；BodyVelocity的float转换只能在部署边界，训练须tensor输出。当前60/20/10ms结果的DN差异均在撤除后出现，因此持续输入与控制窗口的关系须用预登记新证据确定，不能默认每帧reset20ms。可依据新前向成本执行有界持续刺激协议，同时准备可配置但身份绑定的状态/时间接口；具体实际协议需先冻结，不根据输出追改窗口。

完整图任务训练还需独立策略checkpoint（theta/读出/归一化/固定背景/图与时间身份，旧checkpoint有意拒绝tonic）、教师/场景划分/预算、冻结脑与急性恢复/直接视觉/研究重连对照、部署时延及Tello现场条件。AC3–AC6和整个goal未完成，不标记complete或blocked，也不以资源设置已完成结束路线。未启动新硬件或付费资源。

## 当前子批：完整图前向内核候选计时（2026-10-19）

生产缓存A/B已实现，46项数值检查通过；首轮独立验收发现两个证据入口仍只接受旧backend，未运行全图。新实施者同步严格新身份，37项相关证据检查通过，新验收者[PASS](../../specs/spec-008-trainable-snn/acceptance-forward-csr-cache-r1.md)已读取处理，无返工项。随状态更新提交后串行运行P1 r5与配对r2，数值协议保持；尚无此版本完整图性能结论。

软件已独立通过并提交2a3979a，实际`data/forward_kernel_20261019_r1/`正常终止5.49s、采样峰值1.443GiB；三输入全部预热/五轮输出逐元素一致，SciPy中位约15.3ms对Torch约74.6ms。新的[实际独立PASS](../../specs/spec-008-trainable-snn/acceptance-forward-kernel-run-r1.md)已读取处理，无返工项。按[生产前向子规格](../../specs/spec-008-trainable-snn/forward-csr-cache.md)实施共享SciPy前向与原Torch全细胞伴随，软件与完整P1/带背景动力学对照分别验收；当前未实现生产替换，不把内核收益当整体收益。

上一轮只确认既有1GiB设置，属于状态复核，没有新的实施进展。本轮继续已授权目标，按[前向内核实测协议](../../specs/spec-008-trainable-snn/forward-kernel-benchmark.md)先实现可重复的离线基准并独立验收，再提交和运行一次。生产模型暂不修改；数值或性能结论待原始结果。启动前只读快照可用5.513GiB、swap1.434GiB，无已有模型进程；实际预算以新登记时为准。Noria/Docker保留。

## 当前断点：完整图配对响应已有实际证据（2026-10-19）

上一轮属于实际进展：完整图内核缓存软件及r4实际对照均独立通过，首轮配对诊断也已完成。新运行 `data/observability_20261019_r1/` 正常终止，5400步、468.51s、采样峰值1.500GiB；[实测报告](../../specs/spec-008-trainable-snn/observability-run-20261019-r1.md)及[新独立PASS](../../specs/spec-008-trainable-snn/acceptance-observability-r1.md)已读取处理，无返工项。没有活跃的模型进程或未处理Agent报告，所有模型与证据保持完整。

科学结论：零背景本窗DN无差异；固定8mV背景全部六刺激的DN电位有差异，数量378/246/105/243/50/699，左右/上下/小大对比249/229/682。超容差差异首次出现在刺激开始后21–25ms桶，即撤除后1–5ms，不能区分延迟传播与撤除响应。所有条件每个DN的30ms总脉冲数相同（全体总计127），唯一脉冲桶变化为body11072的87→88ms。因此不能把当前总发放率当可区分刺激的读出，也不能每帧清零后只观察20ms。图、绘图脚本与派生时序分析在该本地目录；不重复本次已完成实验来增加信心。

下一步已授权且不受旧内存等待限制：

1. 前向成本仍约16.5s/200神经步。只读 `forward_kernel_cost_review` 已返回并处理：本机非MKL Torch路径对每条边调用长度1的axpy，建议先受监督计时同一完整权重的Torch CSR与SciPy `csr_array`乘向量（固定零/稀疏/稠密向量、包含桥接成本、全部dtype/索引/内存共享语义），不要提前改生产核或宣称加速。参考官方源码：Torch `5c4886908584029761b579af026dcfb627c84070` 的 `aten/src/ATen/native/sparse/SparseCsrTensorMath.cpp`，SciPy `v1.18.1/scipy/sparse/sparsetools/csr.h`。候选未实施/未计时；核级通过不能替代后续完整动力学与梯度对照。保留1GiB预留、Noria/Docker，不并行加载多个全图模型。
2. 只读 `snn_training_integration_readiness` 已返回并处理：复用FollowEnvironment/target_mask/SpatialDrive/全部1314 DN；新策略需 `forward_tensor` 保梯度，部署边界才转BodyVelocity；跨帧保存LIFState及SpatialDriveState、场景边界reset、TBPTT只detach。固定背景、电位或时间特征是由本次信号支持的候选，不自动解冻rho。后续协议须覆盖延迟/撤除效应并验证持续输入响应，再做有内部gain更新的三轴任务训练。
3. 策略checkpoint需独立保存theta、读出、归一化和背景/时间/图身份；旧gain-only checkpoint不支持tonic或新读出，不可绕过。研究重连图目前被训练loader及policy map拒绝，后续需要隔离的受验证研究入口，不能把research_control冒充full_reference。冻结脑、急性恢复、直接视觉和拓扑比较均保留；AC3–AC6未完成。

本轮不创建新的硬件许可、不停止在用服务、不宣布整个goal完成。旧段落中“待内存释放/尚未运行”只描述当时状态，以本节与实际报告为准。

## 最新进展：伴随优化及实际对照完成（2026-10-19）

软件提交 `3f68a09` 后，新登记 `data/trainable_probe_20261019_p1_r4/` 返回完整数值通过；[r4报告](../../specs/spec-008-trainable-snn/p1-run-20261019-r4.md)及[新独立PASS](../../specs/spec-008-trainable-snn/acceptance-p1-run-r4.md)均已读取处理，无返工项，缓存子批A–E完成。实际总墙钟83.69s、反向13.79s、采样峰值1.745GiB；与r3相比全部gain/神经响应精确相同，梯度和theta有已记录的浮点末位差。只解释同协议单次性能，不代表通路或任务能力。

下一步[首轮通路协议](../../specs/spec-008-trainable-snn/observability-plan.md)已按新前向成本冻结60/20/10ms、5400步，预存 `data/observability_20261019_r1.protocol.json`；预估574.58s并非实测。使用新目录保留全部两模型七条件，无梯度；未启动前不声称结果。现有1GiB预留及Noria/Docker保留约束不变。整体AC3–AC6仍未完成。

## 当前子批：复用完整图伴随矩阵（2026-10-19）

上一轮属于实际进展：`62818a2` 已记录完整图 P1/AC2 的新独立 PASS，短窗资源阻塞解除。按[p1-adjoint-cache](../../specs/spec-008-trainable-snn/p1-adjoint-cache.md)实施静态转置 CSR 缓存，保留所有梯度与原数值协议。软件验收及提交后仅用新目录进行一次相同 P1 对照，实际性能另报。没有新的产品决策或硬件操作，不需要再次请求已授权的本机研究权限。

伴随缓存的软件 A–D 已获新的[独立 PASS](../../specs/spec-008-trainable-snn/acceptance-adjoint-cache.md)，44 项相关检查通过，实施及验收报告均已读取处理，无返工项；E 实测仍待提交后运行。后续配对可观测性只读规划已返回并读取：完整七条件、两动力学的60/20/10ms候选为5400步，最终窗口依据优化后无梯度前向成本在运行前冻结；不提前声明视觉到 DN 接通。Noria/Docker 保留，系统预留 1 GiB。各子批提交不结束完整目标。

## 最新实测：完整图内部更新已跑通（2026-10-19）

1GiB预留的软件调整已独立PASS并提交 `28e1b55`。随后一次新登记 `data/trainable_probe_20261019_p1_r3/` 正常结束，父进程完整证据校验后返回 `passed / numeric_internal_update_verified`，见[P1运行报告](../../specs/spec-008-trainable-snn/p1-run-20261019-r3.md)。完整166,700节点、25,582,938边、124,177,617突触计数保持；本次只有R1–R6内部共享增益实际变化，809个L1响应改变，恢复/重载/原始图摘要均通过。

实测采样峰值1.830GiB、总墙钟362.64s，反向291.70s；启动可用3.699GiB、有效RSS预算2.699GiB，最低采样系统可用2.573GiB。没有停止Noria或Docker。此前内存阻塞已不适用于这次短窗运行，不能继续等待释放通知作为下一步条件；长窗训练需求仍未知。新的独立验收者已返回[PASS](../../specs/spec-008-trainable-snn/acceptance-p1-run-r3.md)，报告已读取处理，无返工项；P1/AC2已关闭，AC3–AC6及整体目标跟随、实飞仍未完成。

只读性能审查报告已读取：当前伴随路径存在每步CSR转置到CSC后由Torch转回CSR的转换；可候选预构造伴随CSR，静态额外数组约0.287GiB。尚未实施或计时归因，不把291.70秒全部归因于此。后续按数值等价与相同完整协议比较，再冻结配对神经响应实验；不得绕过完整图或先声称策略已学会。

## 已完成调整：按用户要求预留 1 GiB 并重新测量（2026-10-19）

用户明确要求“保留到1gb”。[Decision 012](../decisions/012-one-gib-system-reserve.md)和[资源调整子规格](../../specs/spec-008-trainable-snn/resource-reserve-1g.md)授权将共享监督、P1及配对诊断新协议默认预留改为1GiB；保持6GiB进程上限、600秒、持续监控及完整连接组。实施与独立验收通过后提交，再由协调者用新目录运行一次P1探针，实际结果另报。旧4GiB登记/失败证据不覆盖，Noria限制继续有效。下面blocked及4GiB限制属于此前阶段记录，不覆盖本次用户授权。

软件调整现已完成，41项定向检查通过并获新独立验收者[PASS](../../specs/spec-008-trainable-snn/acceptance-reserve-1g.md)；两份报告均已读取处理，没有未解决返工项。随后实际测量及其独立验收也已通过，见本文件顶部；软件检查与全图实测为不同证据。

## 当前资源限制：Noria 必须保持运行（2026-10-19）

用户要求检查资源并清理无用内容后，进程审计未发现遗留训练或测试进程，发现 Docker 的四个 Noria 容器仍在运行。用户明确拒绝停机：“不可以，Noria 还在使用”。按[Decision 011](../decisions/011-preserve-noria-services.md)，禁止为本项目释放内存而停止、暂停、删除 Noria 或其 Docker / 虚拟机。没有执行此类清理；完整图训练仍未完成，目标当前为 blocked。后续依据其他可用资源变化继续，不再将 Noria 作为待批准的清理对象。

## 最新进展：P1 实际加载触及 RSS 上限（2026-10-18）

用户重新要求“继续”，无遗留模型进程，内存已高于4GiB门槛，因此按原协议只运行一次新登记 `data/trainable_probe_20261018_p1_r2/`。启动前 available=5,190,090,752 bytes；扣除4GiB预留后 RSS 上限为895,123,456 bytes。约2.7075秒后在 `load_graph` 阶段因 `rss_limit` 停止，采样峰值918,536,192 bytes。没有进入模型构造、前向或更新，P1/AC2仍未完成。原始记录与静态内存分析见[P1第二次登记报告](../../specs/spec-008-trainable-snn/p1-run-20261018-r2.md)。

本轮无业务代码修改，不重试、不裁图、不降低预留。仅优化loader无法使当前未改动的模型构造适配约0.83GiB上限；更大预算下的完整训练成本仍待实测。用户回复“可以，释放后告诉你”，等待通知后重新读取资源并使用新目录继续。旧失败资产保持不变。P2d工具已在 `2f7b7db` 提交，实际配对实验仍未执行；下面此前内存测量与通知状态为历史记录，不代表本次运行结果。

本次失败记录已通过新独立验收者的[S0文档验收](../../specs/spec-008-trainable-snn/acceptance-p1-run-r2.md)，报告已读取处理；PASS仅确认记录准确，数值运行仍FAIL。内存分析者也已返回报告，本轮无未处理子任务。

## 活动目标：完整路线实施与逐步提交（2026-10-17）

用户已授权完成全部路线、删除无用内容、每步 commit。活动 goal 不因 P0/P1 或文档通过而结束，参见 [Decision 009](../decisions/009-fullgraph-training-execution.md)。此前“本轮只规划/不训练/不提交”的记录仅指历史阶段。

- 前置基线已复核并提交 `69f0711`：完整数据、V2 离线研究工具与规划。没有重复已通过的旧诊断。
- [P0 核心](../../specs/spec-008-trainable-snn/p0-core.md) 已独立[PASS](../../specs/spec-008-trainable-snn/acceptance-p0.md)并提交 `44aa5ac`，19项定向检查；尚无全图更新。独立 `.venv-train` 已安装 torch 2.14.1、NumPy 2.5.4、SciPy 1.18.1、pytest 9.1.1、psutil 7.2.2 等。旧 `.venv` 保留。
- [P2a 环境](../../specs/spec-008-trainable-snn/p2-environment.md) 已独立[PASS](../../specs/spec-008-trainable-snn/acceptance-p2a.md)并提交 `365fab0`；三轴环境、像素检测、分区与日志齐全。已有训练场景视觉基线合格率75% / 零动作0%，二者任务成功均为false；不是 SNN 结果。
- [P1a 工具](../../specs/spec-008-trainable-snn/p1-tools.md)经返工后独立[PASS](../../specs/spec-008-trainable-snn/acceptance-p1-tools-r1.md)并提交 `0d6e18a`，16项初始检查和30项返工检查；最终通过须验证完整产物，不只信任退出码。首次实际登记 `data/trainable_probe_20261017_p1/` 的 available=3,674,931,200 bytes，低于预留4GiB，明确未启动模型。用户回答会先释放内存、完成后通知；尚未收到完成通知，不自动重试。P1实际更新/AC2仍未完成。
- [V1 清理](../../specs/spec-008-trainable-snn/cleanup-v1.md) 及20项定向检查通过；两处文档问题经新实施者返工、新验收者确认 PASS，已提交 `4e4a67f`。V2 已替代该离线流程，旧设备链仍保留到新链完成。
- [旧仿真器清理](../../specs/spec-008-trainable-snn/cleanup-sim.md)已独立PASS并提交 `31b9761`；6项相关检查，感知回归改用明确的测试内针孔纹理刺激，未改Retina或新环境。
- [P2b 映射](../../specs/spec-008-trainable-snn/p2b-mapping.md)已独立[PASS](../../specs/spec-008-trainable-snn/acceptance-p2b.md)并提交 `4eada75`，29项定向检查。完整R1–R6输入、全部descending_neuron输出、元数据身份校验及工程空间事件编码已实现；未加载完整图或训练动作，见 NEURAL_INTERFACE 文档。
- 新发现：3,377个R1–R6全部histamine，当前P0只驱动它们、其余细胞静息无背景时无法产生第一批非输入脉冲。此为方程与注释的静态推论，不是全图实测。P1对L1膜电位的优化仍有效，但不能据此推断多跳输出已接通。细节与文献边界见[可观测性记录](../../specs/spec-008-trainable-snn/signal-observability.md)。[Decision 010](../decisions/010-lamina-tonic-variant.md)已单列可选L1–L5背景假设，不改变原P0/P1无背景协议。
- [P2c 数值变体](../../specs/spec-008-trainable-snn/p2c-tonic.md)已独立[PASS](../../specs/spec-008-trainable-snn/acceptance-p2c.md)并提交 `d9d79a5`，53项定向检查通过。仅精确L1–L5的有界平衡点偏移，fixed/learnable模式独立身份，旧checkpoint明确拒绝变体以免遗漏rho；没有完整图传播或任务收益证据。独立设计审查与实施/验收报告均已读取处理。该轮开始时可用内存约3.0GiB，未收到用户释放完成通知，没有重试P1；数值子批完成不代表整体目标完成。
- 2026-10-18的[P2d配对可观测性工具](../../specs/spec-008-trainable-snn/p2d-observability-tools.md)已独立[PASS](../../specs/spec-008-trainable-snn/acceptance-p2d.md)，47项不同定向检查有通过记录。实现预登记、完整状态/相位分支、全部DN与完整节点计数、增量身份/hash保存及独立证据校验；实际[候选实验协议](../../specs/spec-008-trainable-snn/observability-plan.md)仍为Draft，窗口/预算将在P1实测后冻结。没有加载完整图或运行诊断，没有收到内存释放完成通知；实施与新验收报告均已读取处理。
- 本轮再次只读复核：available=2.8643035888671875GiB、swap_used=7.70867919921875GiB，没有运行中的P1/配对诊断进程。仍低于4GiB系统预留，实际运行连续受相同条件阻塞；软件准备不能代替完整图证据。后续窗口/工作点/训练方案依赖P1成本和配对响应实测，不继续凭假设扩展训练实现。保留旧失败登记，等待用户内存释放完成通知后从新的输出目录继续。
- 各实施/验收报告已读取处理，root按阶段提交。尚未完成全图内部更新、三轴学习、对照研究或实飞；整体目标未完成，不以工具子批通过结束完整路线。

用户通知内存已释放后，先按现有P1协议在新的输出目录登记实际完整图更新，600s/6GiB上限与4GiB系统预留保持。遇到具体瓶颈保留原结果再定向解决；不重跑旧检查。随后完成视觉到下降神经元的可观测性诊断，必要的背景工作状态/分级电位假设单列数值契约与对照，再进入任务训练、内部干预/重连比较、部署时延及Tello实验。无实测不得写成完成；硬件动作需当前现场状态，自动化检查不触发飞行。

## 最新进展：Mac 执行路线（2026-10-17）

用户在查新后确认按完整 MaleCNS 内部训练方向推进，并询问结合当前项目和 Mac 的具体下一步。本轮只读核对代码、已有运行记录及本机环境，新增 [Mac 执行路线](../../specs/spec-008-trainable-snn/mac-execution-plan.md)。M5 / 10 核 / 16 GiB；当前 `.venv` Python 3.14.6，模块发现查询未找到 torch/mlx/jax。未安装、未运行训练、未重跑旧诊断、未操作硬件、未提交。

最小下一批：先完成 LIF 独立数值契约与稀疏反向算子检查，再用完整图完成一次内部共享增益更新并记录资源和干预证据。优先候选为按突触前类型共享有界正增益，固定基图乘完整状态；不能把旧事件索引筛选直接当作正确代理梯度。建议采用独立训练环境，先 CPU、单序列短窗；真实训练速度和 MPS 支持待测。目标标记、跟随容差与完整任务实现仍为后续阶段，整个 Spec 保持 Planning。

本轮文档风险 S0；独立审查 [PASS](../../specs/spec-008-trainable-snn/acceptance-mac-plan.md)，仅确认规划与事实一致，不覆盖前次规划验收，也不代表后端可用或训练成功。报告已读取处理，本轮停止；后续按路线细化实施子批。

## 当前任务（2026-10-17）

用户明确要求 MaleCNS 网络自身学习控制，而不只训练解码器；已确认第一项任务为指定目标跟随，学习前进、升降与转向，当前只有这台 Mac。本机只读查询为 Apple M5 / 10 个物理 CPU 核 / 16 GiB，未进行训练性能测量。

本轮整理 [Decision 008](../decisions/008-trainable-malecns-control.md)、[Spec 008](../../specs/spec-008-trainable-snn/spec.md)及 Plan，并更新使命和路线；只涉及文档，S0。新 Spec 为 Draft / Planning，未实现后端、未训练、未接设备、未付费或提交。此前冻结脑＋岭回归保留为对照，不能继续作为用户新目标的终点。

独立文档审查 [PASS](../../specs/spec-008-trainable-snn/acceptance.md)，仅确认规划一致性；AC1–AC6 未完成。本轮未运行测试或仿真。

下一步先把独立 LIF 数值语义和完整图内部参数更新的资源探针细化到可实施范围；目标指定、空间感觉映射、训练后端、资源门槛与任务容差按新 Spec 的待确认项处理。不得重跑此前已通过的诊断来代替新的问题；不得因 Mac 资源有限暗中删图。用户已经回答的目标轴与算力不再重复询问。

## 前一任务记录：spec-007-model-v2

以下保留已完成工具批次的范围、证据和停止记录，不代表新 SNN 训练已完成。

## Role / Scope

EXISTING。用户授权“开始做，自由决策”，并确认“还没录，先完成诊断和工具”。本次范围为 [执行计划](../../specs/spec-007-model-v2/execution-plan.md) 的 P0–P3 离线工具、完整图资产和标准刺激诊断；M4 真实动作确认研究待采集后继续，不制造标签或研究结果。保留前序审计/设计未提交文件、原始数据和历史 V1 资产；没有 commit/push 授权。

## Evidence

- P0：全部 11 份官方 v1.0 flat-connectome 数据为母本，按作者有出处的 neuron 定义核对 body ID。完整参考图为 166,700 节点、25,582,938 条内部有向边、124,177,617 突触计数，保留弱边、101 条原始自环、217 个内部孤立节点和缺失注释；片段边界另有逐行/逐 ID 账本。资产 `data/malecns_v2/full_reference.npz` 不覆盖旧图。见 [Decision 007](../decisions/007-full-dataset-baseline.md)、[P0 报告](../../specs/spec-007-model-v2/implementation-p0.md)及[角色校验返工 PASS](../../specs/spec-007-model-v2/acceptance-p0-r1.md)。
- P1：完整图传播、共同输入事件、132/98 维因果特征、失败/重置/覆盖及信号报告已 [PASS](../../specs/spec-007-model-v2/acceptance-p1.md)。六类标准刺激各 50 帧全部有效；合计 15 s 仿真耗时 96.83 s，峰值 RSS 1,102,233,600 bytes。活动有刺激差异、左右不对称和静默，不代表动作正确。操作和实测见 [MODEL_V2_USAGE](../../docs/MODEL_V2_USAGE.md)。
- P2：掩码训练、分区隔离、每图每输入种子独立拟合、val 选参、pilot 不算 test、同评分行和覆盖统计已通过。[原验收](../../specs/spec-007-model-v2/acceptance-p2.md)的 CI 三态问题已关闭，见 [P2-R1 PASS](../../specs/spec-007-model-v2/acceptance-p2-r1.md)。
- P3 图工具：R/W/P 机械不变量与离线资产隔离已 [PASS](../../specs/spec-007-model-v2/acceptance-graph-controls.md)。完整 W 实际重放 42 帧、2.1 s 仿真，耗时 27.205 s，峰值 RSS 1,762,639,872 bytes；无动作训练或 G/W 效果比较。

## Current Status

当前 P0–P3 工具与完整图诊断批次已完成，M1–M3 已勾选。P3 登记/执行集成原验收的总预算终态问题已由新实施/验收者关闭，见 [P3-R1 PASS](../../specs/spec-007-model-v2/acceptance-p3-r1.md)。所有子批报告已读取处理，当前无未解决返工项；M4 保持未完成。不得重复前序检查或完整脑运行。

## Entry Points / Assets

- [完整脑重放与报告](../../docs/MODEL_V2_USAGE.md)：`diagnostics.replay_v2` / `diagnostics.report_v2`。
- [掩码读出](../../docs/LEARNING_V2.md)：`learning.train_v2 --protocol ... --out ...`。
- [登记与对照](../../docs/EXPERIMENT_V2.md)：`learning.experiment_v2 template/register/validate/execute`；研究图仅通过显式母图/派生图入口。

运行证据在 `data/diagnostics_v2/standard_stimuli_20261016/`、`data/diagnostics_v2/standard_stimuli_report_20261016/` 和 `data/malecns_v2/research_replay_W201_static/`，属于本地忽略资产。当前没有真实录像/标签，不运行动作训练、不连接设备；首期只监督 vx/wz，其他轴不当作已学会。

## Validation / Stop

总实施按 S3 的具体跨模块风险做定向验证，局部返工按 S1；保留原 FAIL 与后续 PASS。已通过检查不重复；Ruff 未安装，未执行也不称通过。数据已校验，不无原因重下载、重扫 31 GB 或重建图。本批已满足停止条件。后续需真实录像、独立动作标签和任务容差，才能推进真实训练及 M4；P4 实体闭环与 B/C 参数优化另立任务。
