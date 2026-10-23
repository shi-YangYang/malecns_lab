# M1 新行为契约、工程编码与整段数据

状态：实施完成，等待新的独立验收。2026-10-23。风险 **S3**：新增序列化契约、身份/新鲜度门禁与控制交互边界；验证限于新增子系统，不运行全仓测试或设备。

## 实际修改

- `src/malecns_lab/control/x2_behavior.py`：独立 `malecns_lab.x2_behavior.v1` schema；五任务、五行为、三阶段、九执行状态；Observation/Feedback/Perception/Decision/ExecutionEvent；严格 raw-logits argmax 与参数 sigmoid 一致性。HOLD 只接受 FINISH；其他行为 START/CONTINUE/FINISH 由单在途执行归属检查。仅使用 Python 3.10 兼容标准库代码，适配官方 ROS Python 3.10。
- `src/malecns_lab/senses/x2_behavior.py`：57个固定通道，任务/请求变化/能力/身体反馈有效性、实际行为与阶段、正负分开的身体量、视觉有效性及4×4实测占用。无相机默认全零且 invalid。输入不含教师、未来、时间/序号/身份；仅当前请求与反馈归属相等与否作为观测有效含义。
- `src/malecns_lab/learning/x2_behavior_data.py`：完整互动示范生成、教师与观测分离、完整 episode 和参数族拆分；train 200、validation 50、locked-test 100（每任务分别40/10/20）。锁定测试单独文件，训练读取其他 split 不会打开它。仅模拟执行反馈，不冒充实际仿真数据。
- `tests/test_x2_behavior.py`、`tests/test_x2_behavior_data.py`：消息往返、原始输出一致性、身份/顺序/过期与未来时刻、请求更换、在途互斥、ACK与完成分离、无视觉掩码、数据拆分/篡改/锁定文件隔离和保护不代答。

旧数值内核、原始连接组、旧schema与旧包 `__init__` 均未修改。现有 `control`/`senses` 包入口会连带加载 NumPy，这是已有行为；新实验 CLI 应在实际执行时再导入新协议。数据模块自身导入不加载 NumPy/Torch/SciPy/ROS。

## 冻结 API

```python
# control.x2_behavior
Task.GREET / FACE_LEFT / FACE_RIGHT / SHOW_FORWARD / END_INTERACTION
Behavior.HOLD / ADVANCE / TURN_LEFT / TURN_RIGHT / WAVE_RIGHT
Phase.START / CONTINUE / FINISH
ExecutionStatus.IDLE / ACCEPTED / STARTED / RUNNING / COMPLETED / CANCELLED / REJECTED / FAILED / UNKNOWN

Observation(run_id, session_id, observation_seq, request_id, task,
            received_monotonic, feedback=Feedback(), perception=Perception(),
            capabilities=(Behavior.HOLD,), request_changed=False)
Decision(run_id, session_id, decision_seq, request_id, observation_seq,
         state_before_id, state_after_id, behavior_logits, phase_logits,
         parameter_raw, behavior, phase, intensity, policy_id, checkpoint_id,
         input_map_id, neural_start_step, neural_end_step)
ExecutionEvent(run_id, session_id, event_seq, decision_seq, execution_id,
               request_id, backend, skill_pack_id, status, behavior,
               received_monotonic, mc_request={}, raw_feedback={},
               stop_evidence={}, protection_reason=None)
# 所有Message提供to_dict/from_dict，字段schema与kind严格核对。

gate = DecisionGate(run_id, session_id, policy_id, checkpoint_id,
                    input_map_id, max_age_s=30.)
gate.observe(observation)  # 收到后由主机赋单调时间，不能由模型续期。
admission = gate.admit(decision, now)  # accepted/reason/decision/active_execution_id
# rejected不改原始decision；observe新请求后旧request决策失效。
gate.execution_event(event)  # 串行归属/状态转移；ACK不得直接COMPLETED。

# senses.x2_behavior
encoder = BehaviorEncoder.from_policy_map(existing_validated_policy_map)
encoder.input_body_ids, encoder.input_indices, encoder.input_map_id, encoder.identity
occupancy = encoder.encode(observation)  # numpy.float32[3377], [0,1]
channel_values(observation)  # 57有序命名通道；无教师/动作选择。
input_contract()  # 码本算法/尺度/事件率/身份协议

# learning.x2_behavior_data
registry = generate_dataset(seed=2401)  # identity + splits{train,validation,locked_test}
manifest = write_dataset(new_output_directory, seed=2401)  # 拒绝覆盖
train = load_split(output_directory, "train")
# episode['frames'][i]['observation'] = Observation.to_dict()
# episode['frames'][i]['target'] = {'behavior': str, 'phase': str, 'intensity': .5}
score_predictions(episodes, {episode_id: raw_prediction_rows})
```

编码生产绑定必须是已有 `PolicyNeuronMap`，身份哈希与实际数组一致，角色完整参考，166700节点、3377个R1–R6、1314个DN，输入/输出不重叠。码本将输入按 `SHA256(seed:body_id)` 排序并按57通道轮分，全R1–R6每个恰好归属一个通道；保留原始body_id及图索引顺序。没有输入资产或用数值fixture映射均拒绝，不生成替代脑。DT 0.1ms、最大150Hz、事件电压68.75mV，分别绑定到input身份，具体神经事件状态由M3设备编码器维护。

数据覆盖未站稳、已朝向目标、同请求执行中重复反馈、慢反馈、拒绝、缺反馈、旧任务执行中收到新请求、正常完成，以及结束互动时空闲/前进/挥手。完成历史在教师的每请求记忆中，任务不是静态按钮到单动作表。完整片段内不重置；teacher只存在数据生成模块，运行时执行器和感觉编码不调用它。

`score_predictions` 仅给出原始行为/阶段宏F1与整段教师标签一致性，保护事件不计作正确。它**不是实际任务完成率**，模拟反馈中的拒绝/缺失单列outcome；P3/P5仍需正式登记评估和实际执行证据。没有将此轻量fixture宣称为训练通过或产品完成。

## 真实检查

唯一一次定向测试：

```text
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_x2_behavior.py tests/test_x2_behavior_data.py
16 passed in 0.31s
```

另外检查现有diff及新增代码差异；没有运行build、全量lint/typecheck、全仓pytest、完整脑、训练、GPU VM、仿真或硬件。没有commit/push，科研资产未修改。没有在本阶段构造任何生产脑或用合成图验证资产绑定，完整映射与设备刺激的实际绑定由M3生产入口核验。

## 限制与后续

- M1 admission不发布MC指令，不替代M2的技能能力验收、停止保护、现场控制权和动作归属。COMPLETED的实际反馈证据由M2承担；协议状态检查只禁止直接以ACK冒充完成。
- 模拟执行反馈是初始数据；M2真实反馈增补、M3资源探针与有限训练、锁定评估及脑功能对照尚未由M1执行。
- END_INTERACTION仅进入脑；紧急停止通过执行保护事件单列，不依赖脑决定。
- 真实R1码本生产映射的验收需使用已验证完整图的 PolicyNeuronMap；M1轻量检查没有启动该大图路径。
