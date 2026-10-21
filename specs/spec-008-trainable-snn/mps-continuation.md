# MPS 完整连接组状态迁移与有界续训

状态：Done（仅本批迁移与两窗续训）。2026-10-19；[实际报告](mps-continuation-run-20261019.md)与新独立实际PASS已处理。用户最新授权和旧数值门槛的处置见 [Decision014](../../.ai/decisions/014-resume-on-mps-with-recorded-numerics.md)。本批不是旧A数值FAIL的返写，也不把GPU训练称为CPU轨迹等价；Spec008整体训练和学习目标未完成。

## 目标和非目标

把现有完整 MaleCNS joint 训练从真实已完成边界迁到 MPS，保留全部训练状态；实际完成一个完整两帧训练窗口，再跨进程恢复完成下一个窗口。提供每次1–4窗的可续训入口和清楚的运行记录。当前joint已知60/160，启动前重新读取旧run终态链确认，不能仅信任聊天数字。

不重新初始化、不重复旧训练前缀、不剪节点/边、不改变float32、动力学、感觉编码、目标/损失、学习率、窗口长度或场景顺序。不把新能力接入实飞；不同时迁移frozen/R研究分支或宣称全程同后端公平比较。不默认启动无人值守无限续训。

## 模型与数值边界

- 保留 full_reference 166700节点、25582938非零有向边、124177617绝对突触计数及既有文件/逻辑图/原始有序计数身份。采用现有严格loader/map/config验证；真实输入全部R1–R6、全部既定DN，固定L1–L5背景8mV，rho=0。
- 新独立 `brain/mps_trainable.py` 与 `learning/mps_policy.py`，不得 monkeypatch、更改旧对象class或伪造CPU model/policy ID。复用原Config/State/Output、代理梯度、loader/映射/输入事件生成；局部设备适配保持原step积分/阈值/复位/不应期/18步pending和读出操作顺序。完整双向COO缓存一次构造并逐块核对坐标和值，固定权重不申请逐边梯度。所有N维输入梯度保留，静默细胞不删除。
- MPS float32神经/读出前后向，明确禁CPU fallback。感觉phase/events保持原NumPy/float64规则，输入增量按原float32生成后传GPU；不重写编码。CPU与MPS参数/状态的迁移返拷必须精确。GPU状态不得被隐藏的CPU神经计算替代。
- 单独CPU小参数主副本接收GPU raw gradient，使用原 `training_math.clipping` 的float64范数与float32系数乘法、原Adam配置/命名组/学习率；CPU step后把更新参数精确回拷GPU。原内部gain有界、rho固定，不额外学习阈值/背景。
- 每窗保存raw/postclip梯度、参数前后值、gain/rho、命名Adam矩/step、DN/动作、loss、完整末态、时钟和游标。复用原 `validate_window_numbers` 独立算术检查；不因允许设备差异跳过finite、裁剪、Adam和自身readout一致性。CPU/GPU训练轨迹差异不作为已证明无影响，旧kernel原始差异报告关联进新run。

## 迁移与持久化

新入口 `python -m cnsbridge.learning.train_mps --cpu-run <旧run> --out <新run> --windows 1`；后续 `--run <新run> --windows 1`，每次仅1–4窗且总预算仍160。新目录拒绝覆盖，worker单次claim，所有已封存来源保持只读。生产入口只支持joint/full_reference/fixed_lamina_8mv，不因缺数据回退小图。

1. 复用旧 `validate_run`/`scan_run` 校验CPU来源、数据和真实完整前缀，选择最后完成且可恢复checkpoint；旧链只读。旧训练checkpoint通过既有reader验证，不必创建第二个完整CPU脑。
2. 新run绑定parent run ID、checkpoint ID/文件SHA、prefix、CPU继承完成数与optimizer step、完整graph/map/drive/config身份、实际新源闭包、runtime与GPU/CPU optimizer后端说明。新policy/model/state ID必须与实际源码/设备对应，drive/map身份保持。
3. 精确保留theta、readout weight/bias、Adam m/v/step、v/g、全部pending、不应期、drive phase/events、scene/frame/时钟。保存迁移前后数组或摘要及逐项精确校验，不reset/precondition、不把继承60窗伪造为GPU记录。
4. 新checkpoint为非执行性NPZ+JSON，实际绑定图/源码/策略/运行身份及字段、shape/dtype/有限性/时钟/上下文，校验文件摘要后才恢复。GPU参数与CPU主副本精确一致；保存时无grad/计算图。严格拒绝缺失、额外、篡改、其他run或来源不兼容档。
5. 每窗最终seal之后才成为新进度；失败保留部分证据，不把未完成窗加到预算。数值/协议失败不静默续接；资源或中断可从最后完整合法seal恢复。run最新派生进度不覆盖CPU来源档。
6. 下一独立进程恢复所有数组并在前向前精确核对，再计算下一窗；不通过“重新训练一次结果差不多”冒充恢复。场景边界按原规则只重置神经/感觉状态，不清Adam/参数。

## 资源、测量与结果边界

轻量父进程不得加载完整图/Torch；新worker采用共享独占锁、6GiB申请、额外1GiB预留、min(6GiB,available−1GiB)RSS、600s/.1s监督，MPS分配预算不高于有效预算/推荐值；分别记录MPS current/driver周期及阶段用量，不与RSS相加。Noria/Docker不动。无并行模型/数值测试；完整图仅在受监督worker加载。

每窗分开记录构造/恢复、前向、反向、CPU裁剪/Adam、同步/传输、保存及整个段耗时。MPS计时使用同步。不同输入/状态的历史CPU耗时只能作为范围参考，不能声明严格配对训练加速；已有核级速度保留原结论。实际运行须有内部theta更新、完整非零且有限梯度证据和正确Adam步进；非零更新不等于学会控制。

若完整窗口OOM/超时/非有限或明确算术不一致，保留CPU父进度和失败数据，定位具体问题后新登记修正，不静默缩图/缩窗/放宽原独立算术检查。暂不预承诺可实时飞行或所有训练都能完成。

## Plan / 验证

风险S3：涉及核心动力学设备适配、优化器桥、状态持久化及续训链。一个实施Agent串行完成高度耦合模块，根Agent维护Spec和运行，新的独立验收Agent审查。只做相关子系统定向检查，不跑全项目；不得重复无变化已通过检查。

新增文件按脑、策略、优化器/窗口、checkpoint/证据、轻量runner/worker清楚分责，避免复制整个研究训练框架。可以复用旧纯helper及现有MPS探针只读代码；业务模块不得反向依赖examples，应提取为新增包内实现且保留历史源。不得改G29、R13、已封存MPS20源文件。最小测试覆盖：非对称有符号代数、实际MPS动力学/静默细胞theta梯度、状态/时钟/输入约束、CPU主参数桥及Adam精确恢复、篡改/模式/来源拒绝、scene边界、runner不可覆盖和故障seal边界。小图仅数值fixture，不替代真实脑。

软件PASS/提交后，协调者在新目录 `data/neural_training_mps_20261019_r1` 运行第一窗；若通过，再新进程续一窗。新的独立实际验收核对保存的实际数组、资源、来源、内部更新和恢复，不重跑训练。确认完成GPU切换和实际续训后交付当前进度与继续命令；未来正式科学比较需登记各分支迁移点。

## 验收标准

- [x] M1：隔离MPS核心/策略及CPU Adam桥定向检查和独立软件验收通过；后续内存修正亦分别验收，实际完整图证据见M2/M3。
- [x] M2：r3完整图真实CPU→MPS状态精确迁移，第61窗完成内部更新并保存合法新checkpoint，独立实际PASS。
- [x] M3：r3第二独立进程准确恢复所有参数/Adam/递归状态与游标，实际第62窗成功；保留旧源与预算，独立实际PASS。
- [x] M4：实测报告、GPU前后向/CPU优化器边界、实际速度/内存、62/160进度和续训命令经独立S0核对；随本状态更新创建阶段中文commit。

无待用户决定事项。重新开始不是本批方案；用户将来明确要求从零纯GPU研究对照时另建run，不删除现有进度。

## 实测后局部修正：COO构造峰值

软件提交542f50e后的r1首段在 `load_full_policy` 因RSS上限停止：启动available3329949696 bytes，有效上限2256207872 bytes，采样峰值2302181376 bytes，5.381370833s；GPU活跃分配仍仅2901248 bytes。没有训练更新，合法父边界仍60。初步代码定位是直接COO构造并coalesce同时保留多份全边坐标/排序缓冲；尚不能据此判断完整BPTT可用内存。

允许局部优化新MPS缓存构造：先使用已验证canonical CSR，再转换成已排序COO，避免对完整COO再次排序/聚合。必须保持完整坐标/值、原权重乘法、实际GPU返拷逐块核对、全节点梯度不变；不修改G29/R13/旧MPS20。风险S1，仅最相关已有MPS矩阵/梯度检查加缓存坐标值语义检查；通过后独立静态验收，不重复全子系统。保留r1全部失败记录及542f50e源版本，修正后提交并以新r2登记，不能继续旧r1冒充同来源。其后依旧先一窗再独立进程下一窗。

## 实测后内存修正：完整窗口分段重算

984dfc2后的r2已完成完整图构造（3.3063s）与全部状态精确恢复（0.6628s），但首帧前向保留反向张量时触发MPS allocator限制；实际限额2162245632 bytes、current采样峰值2055973888 bytes，worker11.2548s。合法边界仍60，没有完成窗口。保留r2失败、resume证据和984dfc2来源。不能降低1GiB预留或仅重试相同不足的预算。

允许在新GPU策略中引入固定50神经步块的activation checkpointing（尾块按真实剩余步数），风险S3，限于保存/重算与梯度连续性。每两帧仍1000步且跨块、跨帧不detach；块边界保留完整v/g/pending/不应期，theta与读出梯度须贯穿整个窗口。积分、代理梯度、阈值与reset、感觉输入和DN逐步累加顺序保持不变，不能用块和替换逐步累加以改变舍入顺序。

块函数必须可重复：每次从绑定的感觉phase/event/clock副本重建局部SpatialDrive，输入mask及脑初始clock不可受后续循环变量或外部drive副作用影响。反向重算不能推进外部感觉状态、重复预热或修改训练游标。不使用随机输入或逐边梯度，不隐藏CPU神经计算。采用PyTorch non-reentrant checkpoint，显式选择固定设置并记录在新身份/来源中；旧MPS直接前向保留为数值fixture比较路径，不提供生产绕过。

验收包括实际MPS直接与分段路径在同一小型数值fixture上的末态、动作、DN、raw梯度比较（包含跨块/跨帧/非零完整状态），CPU Adam桥及存档恢复的受影响检查；不重跑未改动的来源/CLI/资源测试。先独立软件验收/提交，再在新r3从CPU60重新登记首窗。后续仍只在首窗成功后新进程续下一窗，资源与自身算术门槛均不放宽。
