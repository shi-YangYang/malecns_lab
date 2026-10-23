# Spec023 实施记录

2026-10-23，风险 S2；实施者仅完成新离线入口与轻量检查，未执行完整脑、VM、仿真、训练或设备动作，未提交。

## 实际修改

- `learning/x2_task_eval.py`：固定5场景15帧协议同时纳入 registration 与 execution 哈希；新目录封存54个原登记历史源码及完整历史包、checkpoint来源、当前包装源码。复用既有参数/完整图校验、独占锁、只读仿真预检、240秒父进程监督、1GiB预留与MPS采样预算。worker须具有监督父进程、已占用共享锁和一次性claim，之后才导入数值栈。
- 同一完整policy与FrozenInference加载一次；每场景调用原initial_state，执行600步空白预热，3帧各500步且连续保留神经/感觉状态。只传像素给forward_tensor，不传期望标签、距离或可见性。缺目标是160×120×3 uint8灰32图像。未修改旧算法、数值后端、动作门限或旧schema。
- 每场景保存完整初/末状态NPZ、4个完整状态摘要和15帧的输入NPY/输出NPZ/JSON；摘要使用历史validate_state，涵盖神经与感觉数组、时钟和身份，初态跨场景严格一致。每帧记录原action/features/logits、像素/数组/文件hash、前后状态、神经步和执行身份。非有限输出数组先保留，再判技术失败。
- 每场景独立真实SpikeTelemetry目录与run_id：检查42批完整性、每批50步、600步预热加3×500步、22群组/32神经元、所属身份、摘要与最新批一致；缺批/采样错误不作已完成。总期望210批。
- 评分调用原`control.x2_protocol.intent`，只由raw vx严格大于0.03判FORWARD。有效完整执行的行为FAIL仍为status=completed、behavior_pass=false；技术FAIL为status=failed、behavior_pass=null，不对缺帧计低分。独立报告6个FORWARD与9个STOP、逐场景/逐帧和恒策略6/15、9/15对照。
- `experiments/x2_task_eval.py`：无--execute时仅预览固定协议，帮助/默认导入无Torch/NumPy/SciPy/ROS/历史脑导入和结果目录副作用；执行必须明确全新data或logs子目录。

## 实际检查

唯一相关测试组：`.venv/bin/python -m pytest -q tests/test_x2_task_eval.py`。

首次为10通过、1失败：顶层导入intent经control包初始化连带导入NumPy，违反默认惰性。将导入延后到实际评分/worker；未改control包。因相关代码修复复跑同一组，最终 **11 passed in 0.15s**。检查覆盖默认/帮助惰性、固定样本/灰色缺目标、标签/顺序、严格门限/有限值/缺帧拒绝、恒策略评分、路径不可覆盖、父进程/锁/claim、预检失败保留、父监督预算与有效行为FAIL、协议hash绑定和遥测缺批/错时钟拒绝。测试仅用像素与序列化fixture，无构造神经网络。

未运行全仓测试、build、lint、完整脑数值配对或其他已通过检查。新增文件差异经静态检查，无空白错误。

## 实施交付时的准入限制

尚未执行Spec023真实完整图，故不宣称模型通过、资源实测通过、210批真实遥测完成或真实动作能力。新独立审查需确认协议/来源/监督/状态续接和失败分离，之后由协调者执行一次已授权有界离线评估。首次真实结果需核对所有状态、输出、参数/原图不变和MPS监督；行为FAIL不构成自动调参/训练/重试理由。

这是新拟定X2任务标签的一次固定相关样本筛查，历史模型源于离线模仿训练；不把本筛查FAIL扩大为原训练全部无效，也不以脉冲变化单独证明决策正确。

协调者随后完成唯一一次授权实测，最终状态见 [actual-evaluation.md](actual-evaluation.md) 与 [acceptance.md](acceptance.md)：工程与证据PASS，任务行为FAIL。上述未运行说明仅对应实施者交付时点，不是当前待办。
