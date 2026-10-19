# Acceptance — MPS 完整窗口分段重算

## Result

**PASS**。本次由未参与实现的新验收 Agent 独立审查，范围限于 `mps-continuation.md` 追加的“完整窗口分段重算”软件子批。允许提交后在新 r3 登记完整图首窗；尚未证明完整窗口能够容纳在当前内存预算内，也未将第61窗或跨进程第62窗标记成功。

## Spec Coverage

- **完整梯度窗口**：生产 `forward_tensor` 始终进入固定50步、尾块按剩余长度的 non-reentrant checkpoint。`v/g/refractory`、全部 pending 和两个 DN 累加器是显式张量输入输出；跨块、跨帧没有 detach，优化器仍在原两帧窗口反向结束后才更新。没有改变每帧步数、图规模、动力学或代理梯度。
- **重算无外部副作用**：工厂参数绑定每块的神经 clock、帧内起点、长度与感觉快照，不引用后续循环变量。`SpatialDriveState` 的数组由原实现复制为不可变数据；每次重算新建私有驱动器并恢复该快照，`restore_state` 再复制 phase/events。mask 也被复制并置为只读。局部构造的 PCG64 不使用全局 RNG，恢复后使用原 phase；反向不改变外部感觉状态、预热流程或训练游标。
- **原累加次序**：DN 运行和作为块输入输出，每个神经步仍按原顺序累加，半帧归属使用绝对帧内步号。没有先对块求和再归约；尾块和跨半帧块均覆盖。
- **真实来源**：固定 checkpoint 配置在策略身份和协议中登记，新 helper 进入策略实现摘要及运行源闭包。CPU 父档与旧来源不被伪装成新后端；直接前向只允许 `numerical_test_only` 图，完整图无法通过该入口绕过生产重算策略。
- **范围与失败边界**：差异限于新 MPS 策略、身份/协议、helper、直接相关测试与文档。原 G29/R13/MPS20 业务文件没有出现在本轮 diff；原 Adam、状态存档、监督和资源门槛未改。Spec 保留 r1/r2 失败及各自源码版本，r3 仍须从合法 CPU60 边界另行登记。

## Validation

- Risk Level: **S3**，具体风险为反向重算的状态纯度、完整时间梯度连接、数值累加次序以及来源身份。
- 独立阅读 AGENTS、使命/技术栈、唯一项目规则、验收流程、当前 handoff、Decision014、Ready Spec 追加段、实施报告、前序软件/缓存修正验收。
- 独立审查全部本轮业务 diff 和未跟踪的 `mps_recompute.py`，对照原感觉导出/恢复/事件实现、MPS 神经 step、训练窗口反向与 Adam 顺序，以及新增测试。
- 采信实施者已完成的两项实际 MPS 定向检查：`test_checkpoint_matches_direct_multiple_blocks_two_frames_and_raw_gradients` 与 `test_actual_mps_window_cpu_adam_exact_archive_restore_and_scene_reset`，**2 passed / 8.89 s**。首项连续两帧、每帧120步（50+50+20），初态包含非零 v/g、全部18项 pending、活动不应期和非零感觉相位/计数；动作、DN 和完整末态精确一致，theta、读出和初始 v/g/pending 的 raw gradient 最大差记录为0。测试还核对反向前后外部状态不变、时钟只前进240步。第二项覆盖受影响的 CPU Adam、参数更新、命名矩、NPZ 恢复、再次更新和场景重置。
- 本验收未重复已经通过的测试，未运行完整图、数值基准或全项目检查，没有修改业务实现。

## Issues

未发现阻断本次软件子批的问题。

## Regression Risks

小型 fixture 支持重算与直接路径的数值语义一致性；它不证明完整图的实际峰值、速度或当前预算是否足够。分段重算会增加前向计算和感觉重放成本，实际端到端耗时仍待受监督 r3 测量。CPU/MPS 轨迹等价没有成立，本报告不增加该门槛，也不把用户接受差异解释为已证明等价。

## Required Rework

无软件返工项。协调者提交后执行新 r3 首窗，只有成功后才独立进程恢复下一窗，并另行审查真实保存数组、内部更新、恢复和资源证据。若实际仍 OOM、超时或自身算术失败，保留失败数据和 CPU60 合法进度；不能以本软件 PASS 宣称续训完成。
