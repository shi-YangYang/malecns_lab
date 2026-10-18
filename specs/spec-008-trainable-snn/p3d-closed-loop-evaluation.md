# P3d — 有状态神经闭环评估

状态：软件已通过新的独立[返工验收PASS](acceptance-p3d-rework-r1.md)，无返工项；[原FAIL](acceptance-p3d.md)和[返工实施](implementation-p3d-rework-r1.md)保留。协议由[早期候选](p3d-closed-loop-evaluation-draft.md)收敛并已独立[协议PASS](acceptance-p3d-protocol.md)。实际评估须等 joint 与独立 frozen_brain 均完成冻结的160窗，再执行首次1+1帧实际验收。当前无正式评估结果，AC3–AC6仍未完成。

## 目标与边界

在独立完整场景让神经动作改变相机位姿和下一帧，比较内部训练、独立冻结脑、急性内部恢复与常规视觉控制。支持跨进程完整状态恢复，不重复神经前缀。教师录像损失不能替代闭环行为。

不改变已冻结脑核心、感觉映射、优化器、训练数据、评分阈值或现有29个绑定训练源码；不启动硬件、不挑最好中间checkpoint、不把拓扑贡献纳入本批结论。保留完整图及全部DN特征；所有场景仍使用同一种红色标记，不代表任意目标或真实房间泛化。

## 最终参数准入与条件

登记从真实唯一封存训练链核验，不能只信`progress.json`。复用`training_contract.validate_run`、`training_evidence.scan_run`、最终`training_archive.inspect_checkpoint`。两个独立run须模式正确、完整图/初始来源/预算/数据/运行环境相同，并实际达到epoch=0、scene_index=3、scene_id=train-sine-yaw、next_frame=80、completed_windows=160、optimizer_steps=160。frozen全链theta保持原始数组，不能共享joint训练读出；数值异常或坏链不得准入。

| 固定条件顺序 | 有效参数 |
| --- | --- |
| joint | joint第160窗最终theta及读出 |
| frozen_brain | 独立frozen第160窗读出，theta保持初值 |
| acute_restore | joint最终读出，theta恢复已封存r2原始数组，再初始化场景 |
| direct_visual | 现有`follow_task.visual_teacher`，手写图像控制基线 |
| zero | 现有零动作基线 |

绑定两个training_run ID、协议/数据身份、最终window ID、完整训练checkpoint ID、nested参数checkpoint ID及实际文件SHA。worker用`load_policy_checkpoint`加载参数，不加载训练末隐藏状态/Adam。acute须比较原始theta真实数组，不能仅假定“初值是零”。

`policy_id/model_id`不含学习后的数值，另定义`parameter_set_id=canonical_hash({policy_id, theta_digest, readout_weight_digest, readout_bias_digest})`。条件、段、状态档、恢复前、计算后均核对实际有效数组与该身份。相同policy ID但参数不同必须拒绝。条件目录保存effective parameters.npz和condition.json，不改原训练档；非神经基线用自身来源，不伪造参数/神经状态。

## 冻结场景与比较

- val顺序：val-triangle-forward、val-triangle-yaw、val-triangle-vertical、val-triangle-combined。
- test顺序：test-chirp-forward、test-chirp-yaw、test-chirp-vertical、test-chirp-combined、test-chirp-occluded、test-excursion-out-of-view。
- 每条件依次完成本split全部场景，再下个条件。全部5条件×val终局并封存验证汇总后才允许test；资源/数值/证据失败不算完成场景，真实物理失败是正式场景终局。
- 绑定`scene_registry()`真实dataclass定义、全部TargetCamera参数、环境/评分源码。保持160×120、72°水平视场、20cm标记、期望距离1.5m、环境dt=0.1s、tau=0.25s、8s/计划80帧及已有失败边界。神经每帧50ms/500步，初始空白60ms/600步。

各条件共享外生轨迹/初始位姿/监督，动作导致各自图像与轨迹；不能要求逐帧图像/监督标签相同。每场景固定参数重新initial_state一次，场景间不保留隐藏状态。只使用最终档；若看val后改模型或协议，另立候选并保留原记录，不用test调试或选档。

frozen比较同容量独立读出训练下的内部学习效果；acute检查已学习内部参数的动作/行为贡献，两者不可替代。direct_visual非同预算训练网络，不能据本任务宣称优于所有无人机模型。真实拓扑贡献需独立受控重连子批，本批不关闭该项。

## 输入、动作与监督

每帧只读BGR → `torch.no_grad()`下`policy.forward_tensor(frame, state)` → 保存全部2628 DN特征和raw动作 → 独立限幅/目标丢失监督 → 环境step → 新观察。神经接口只获图像与自身状态，不接检测误差、距离、教师动作、真值或环境对象；评分器与直接视觉基线在外层使用各自允许信息。评估不需要teacher_action。

BodyVelocity FLU vx/vz/wz限幅±0.3m/s、±0.2m/s、±0.5rad/s，其他轴为零。检测无效仍推进神经并保存raw输出，随后监督置零；单列lost_target/action_clipped，不把监督停住算作脑的能力。非有限状态、DN或raw动作、不支持的轴立即停止留诊断，不通过限幅掩盖。每段计算前后参数数组完全不变。

## 分段与独立状态格式

每次显式调用1–40帧；每10帧通常seal一个chunk，段末或物理终止可短块，支持1帧/奇数边界。段不跨场景；请求超过计划剩余帧在启动前拒绝，物理失败可提前结束。无自动矩阵循环/重试。

新增`cnsbridge.evaluation_checkpoint.v1`，purpose=`no_grad_closed_loop_continuation`。无Adam，保存完整brain.v/g/refractory_until/全部pending、drive.phase/event_counts，policy/model/drive/spatial身份和神经/感觉/帧时钟；context包括evaluation_run_id/protocol_id/condition_id/parameter_set_id、scene_id/scene_spec_id、next_frame/planned_frames、terminal/termination_reason、父checkpoint ID、真实前缀SHA、环境摘要/下一图像摘要，以及源码/runtime/实际数组声明和SHA。checkpoint.json最后写，新目录不覆盖。

支持frame0、奇数、80及物理提前终止，不能套用只接受正Adam步/偶数帧的训练档。可复用Torch-free `checkpoint_state_digest/validate_state_digest`；仍须校验真实数组清单/shape/dtype/有限性、phase∈[0,1)、非负event_counts/refractory、严格非布尔整数游标。摘要校验不能替代数组校验。

时钟固定`brain_step=drive_step=600+500*next_frame`、环境time=min(next_frame*0.1,8)。terminal必须对应真实done；终局可汇总、不得继续前向。段配额耗尽不等于场景完成。直接视觉/零基线用相同环境/动作链，但不虚构PolicyState。

## 环境恢复与父复核

从冻结scene/camera/dt/tau新建FollowEnvironment，顺序重放已封存executed actions，每帧核对像素、检测、truth/next_truth、时钟、监督和done，末尾核对truth全字段、已执行帧数、done及下一图像SHA。重放结果是权威环境状态，不直接赋值私有字段、不重演神经前缀，也不增加样本数。

恢复神经后、下一前向之前，worker保存实际参数、完整状态摘要、环境摘要及游标，与父档/重放结果比较。不得额外预处理；不能仅凭结构ID恢复到另一参数条件。

父不导入Torch/建图，独立NumPy复算保存DN→既有标准化→linear→tanh→动作尺度，rtol=1e−6/atol=1e−7，严格清单/shape/dtype；这只验证读出转换，不独立证明DN生物正确。父重算直接视觉/零raw动作、环境重放、检测/监督和所有指标，不信worker的success。全图来源、路径、状态证据分别保留。

## 持久化与失败

套件run.json登记完整协议/来源/runtime/最终参数；conditions/<condition>/保存有效参数与清单。episodes/<condition>/<scene>/有episode.json；segments/000001/有registration、phase、resources.csv、worker.log、resume.json/resume.npz、worker_result、原值supervisor_result、父result。

段内chunks/<全场景chunk号>/含evidence.npz（帧、全部DN、raw/executed动作、索引/时间）、trajectory.jsonl（检测/监督/truth/next_truth/误差）、event.json、prefix.jsonl、checkpoint/、最后chunk.json。基线无DN。证据→event→prefix→checkpoint→chunk seal避免循环哈希；规范前缀行绑定位置及event SHA。拒绝越界路径、重复/跳帧、分叉、错场景/条件/参数、坏摘要；chunk与segment编号分开。

父按唯一连续链复核，一次仅保留一份完整状态/当前块。未封存尾部保留但不计数；资源/IO失败仍failed，可单列资源采样范围内的有效sealed prefix，下一显式续段才使用。数值/协议失败不得自动恢复/回退/重试。保留supervisor原值，不把部分进展改为成功。物理失败是有效研究终局但行为失败，不补80帧。无有效记录的失败不调用假定非空轨迹的评分函数，不造success。

## 资源、入口与实现计划

生产worker复用`train_snn.workspace_lock()`，与训练/评估共享OS锁，不能并行加载图。每段1GiB额外预留、6GiB申请、有效min(6GiB,available−1GiB)、600s/0.1s监督，Noria/Docker保持。父只轻量准入/复核；时钟偏移沿用保守拒绝，不放宽采样范围。以下是待实现接口，不是当前可运行命令：

```text
python -m cnsbridge.learning.evaluate_snn register --joint-run <run> --frozen-run <run> --out <new-suite>
python -m cnsbridge.learning.evaluate_snn run --suite <suite> --condition <name> --scene <id> --frames <1..40>
python -m cnsbridge.learning.evaluate_snn summary --suite <suite>
```

register只登记，run核对冻结顺序/条件并持锁，summary从验证证据汇总，不完整时不冒称完整比较。源清单绑定所有实际复用helper，不修改29个已冻结训练源码。单一实施者新增evaluate_snn（入口/监督）、evaluation_contract（准入/身份）、evaluation_checkpoint（无Torch格式读取及worker内恢复）、evaluation_worker（生产循环）、evaluation_evidence（链/重放/评分）；清晰独立职责可再拆，不能堆成单文件。新增定向测试和使用文档。

## 输出与比较规则

复用`follow_task._errors/_metrics`语义：水平/垂直/距离MAE、可投影数、有效观察率、合格率、success、三轴非零样本/动作变化、监督数、遮挡数、目标丢失/恢复片段、失败。x/y容差各0.15、距离0.25m，计划80帧合格率至少0.8且完整无失败才success。比率分母固定80，提前失败不缩分母；有效片段MAE须并列实际覆盖/失败。

保存各条件各场景指标与配对差值，固定场景集合等权汇总，不只展示最好场景。缺失/资源失败/数值失败/物理失败分开，缺失不赋零；固定小场景集合不伪报统计显著性。并列实际帧数、神经时间、环境时间、worker/监督墙钟、峰值RSS/最低available、阶段耗时，不以环境dt冒充实时处理率。

## 验证与首次实际门槛

协议文档S0审事实/diff；持久化/监督实现S3，限新评估子系统与直接接口，不重复旧训练检查。新的独立软件验收后提交，再进行实际门槛。

软件至少检查：连续vs两个真实OS进程恢复的数值/环境一致；frame0/奇数/终局；同policy ID错参数拒绝；acute在initial_state前干预；神经无真值/检测泄漏；动作改变下一图像；丢目标仍推进状态；参数不变；父读出/环境/评分复核；全val门槛；坏链/缺档/假成功/失败有效前缀；父无Torch；基线不虚构神经证据。小数值fixture不是生产图回退。

两模式最终参数准入后，正式实际门槛固定joint/val-triangle-forward：第一worker只进1帧封存退出，第二新worker从frame1恢复再进1帧。两帧作为正式val前缀继续，不额外计算完整脑连续对照。新的独立实际验收核对奇数边界、前向前实际参数/完整状态/环境、无重复预处理、来源/资源；首帧若真实物理终止则保留正式结局，不伪造续段，按事实重定门槛。

门槛后显式每段最多40帧，全部val汇总后开放test。软件通过、有效实验与跟随成功分开；阴性结果也可通过证据验收，不能要求成绩变好才保留。拓扑及部署仍按各自子批，整体goal不随本批结束。

成本仅按既有两帧无梯度19.36–19.62s线性估计：40帧387–392s加加载/预处理/保存，80帧约13min，三个神经条件全部10场景约6.5h。非长场景实测，不据此放宽600s。
