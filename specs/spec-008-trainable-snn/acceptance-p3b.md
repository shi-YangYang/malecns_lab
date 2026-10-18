# Acceptance — P3b 两帧动作联合更新工具

## Result

**PASS**（软件范围）。2026-10-19，由未参与本轮实现的新独立验收 Agent 审查。允许协调者处理本报告、提交软件后，按冻结协议在新目录登记一次受监督完整图实验。尚无 P3b 完整图实际运行结果；本 PASS 不表示网络已学会控制、资源预算必然足够或整体 AC3–AC6 完成。

## Spec Coverage

- **完整图、身份与输入：通过。** 生产 worker 只用 `NeuralPolicy.from_reference` 与生产 checkpoint loader；没有数值 fixture 回退。父进程先绑定图文件字节摘要/provenance、封存数据 manifest/ID、固定协议、源码及当前预算。固定 combined 的 0/1 两帧、同帧 `teacher_target`、相机/时钟/valid 校验均与协议一致。父复核重新读取封存数据并逐项比较输入/标签副本；全部输入和 DN 身份、原始计数摘要及 DN 顺序有对应检查。
- **连续计算和联合优化：通过。** baseline 从 60 ms 无梯度空白预处理开始，两次 50 ms `forward_tensor` 直接传递同一连续状态，帧间不 detach/reset。仅 theta 与 readout weight/bias 进入已登记 Adam 参数组，只有一次 `optimizer.step`；任务损失按两帧三轴等权归一化 MSE，固定 rho 不参与训练。初始 theta/gain、正则值和正则 theta 梯度分别检查；裁剪前后的完整梯度、全局范数、系数、前后参数和一次更新步号均落盘。非有限数值停止，已生成证据保留。
- **五分支与重载：通过。** baseline、updated、acute_restore、full_restore、checkpoint_reload 分支参数组合符合协议，后四次关闭梯度且每分支 fresh initial state。updated 后先保存完整策略参数；急性恢复保留新读出、恢复 theta，完整恢复再还原读出。分支输出为独立 NumPy 副本；更新函数退出前释放反传图与优化器，worker 删除旧 policy 并回收后才构造重载模型。存档未被冒称为优化器/隐藏状态续训档。
- **资源与进程边界：通过。** 普通父入口和证据模块不导入 Torch/worker；图 metadata 之外的完整模型加载、计算与重载在既有监督器启动的子进程内。预算明确为额外 1 GiB、申请 RSS 6 GiB、有效上限 `min(6 GiB, available−1 GiB)`、600 s、0.1 s 采样；没有服务、Docker 或 Noria 操作。新目录约束和注册失败保留行为明确。原始监督终态先保存为 `supervisor_result.json`，域结论另存；资源失败不能升级为协议完成。
- **独立证据复核：通过。** 父校验使用保存数组独立重算动作读出、分轴/总损失、梯度范数、裁剪、首次 Adam 更新和恢复/急性恢复/重载比较；同时检查源/策略/模型/图/时钟/存档身份、参数对应、文件摘要、五分支完整状态摘要清单以及资源 CSV/子进程终态。它不重新执行神经模型，也没有将自报 worker completed 直接提升为领域通过。状态证据准确表述为预处理与末状态摘要，不是全节点逐步轨迹。
- **阴性结果与研究边界：通过。** `protocol_complete`、`joint_numeric_update`、`internal_action_effect` 独立判定；零梯度、无动作影响或损失上升不被删掉或重试到阳性。比较使用全部两帧三轴和全部 DN 特征及登记容差。文档明确两帧离线模仿、工程背景/输入、无闭环或泛化证据，急性恢复包含预处理状态改变的影响，不据此定位唯一刺激通路。

## Validation

- Risk Level: **S3**（新训练、持久化证据及资源监督交互），本轮审查范围限 P3b 新工具。
- Evidence Reviewed / Checks Performed：AGENTS、使命/技术栈、项目规则、验收流程、Spec 008/Plan、当前交接、Decision 009/011/012、P3b 冻结协议与独立协议 PASS、实施报告；五个新增 learning 模块、完整定向测试文件、使用文档；相关既有 policy、checkpoint、trainable graph/tonic、空间驱动和 resource guard 接口。
- 独立阅读了未跟踪新增文件，并核对 Git 工作区/差异范围。既有核心、数据与 checkpoint 实现未改；交接文档为协调者修改，本报告为验收者唯一写入。
- 实施报告记录 `PYTHONPATH=src .venv-train/bin/python -m pytest tests/test_joint_probe.py -q` 明确结果 **36 passed, 1 warning in 1.14s**，warning 为既有 sparse CSR beta 提示。已独立核对测试断言覆盖真实小电路两帧状态/梯度、一次优化、五分支、裁剪/非有限、阴性分类、受损证据、固定数据选择、父进程导入和监督边界。没有发现需要新增执行检查的具体未覆盖阻断风险，因此未重复该通过检查或旧套件。
- 本次未加载完整图、运行训练、重新生成数据、访问硬件、操作服务或执行 commit。数值 fixture 通过没有被当作完整图实测。

## Issues

无阻断问题。P3b 实际产物尚未生成，完整图实际运行和独立实际验收必须作为后续单独阶段完成。

## Regression Risks

100 ms 反向的实际内存、梯度数值和总墙钟仍未知；采样监督不是硬隔离。恢复/重载的生产路径与完整 DN 维度尚待本次真实运行证据。当前两帧标签均正，一次更新不证明负方向控制、长时信用分配、闭环稳定性、未见场景表现或真实连接组的优越性。父复核能检验已保存证据一致性，不是重新计算神经轨迹或防止人为重造整套证据的密码学证明。

## Required Rework

无。协调者可提交本软件阶段后，严格按已登记数据、时间窗和资源预算运行一次；原样保留阳性、阴性或失败结果，由新的实际验收者复核，不修改冻结选择来取得成功。
