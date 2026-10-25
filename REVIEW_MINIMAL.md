# 1. 运行身份与配置

范围：只审阅下列一个原始观察小模型 A 运行、一个 END 失败请求。Goal 保持暂停；本次仅读取、摘录与核对文件身份，未执行训练、编码、教师、模型、评分、仿真或测试。

- run：`/Users/yang/project/malecns_lab/data/x2_model_redesign_20261024_r1/a_context_factor_v1/execution-r1/A-seed-24102431`。
- 执行时间：2026-10-24T20:19:03.279641+08:00；结果 `DONE`、exit_code=0；worker记录耗时 3.551365041 秒，监督器记录进程耗时 3.830511666 秒。程序完成不代表行为合格。
- 模型：单向单层 `GRU(44,32) → Linear(32,9)`，7,785参数；固定种子 **24102431**，最终 epoch **300**，没有续训；CPU单线程，PyTorch 2.14.1、NumPy 2.5.4。
- 模型身份：`execution-r1/A-seed-24102431/checkpoint-300.pt`；记录的SHA256 `554760a7fbd042512d7e5fccbc8b3b180e57c8fbc58e4d22c5a2cfba5cc1f5e2`。本次未加载权重，也不在文档中包含权重。
- 数据schema：`malecns.x2.context_factor_labeled.v4.v1`；dataset_id：`75f216e8d5639f9a9ef5aaaa43dd233079cdad04e53366c045abe6f2435a5e9b`。
- 数据文件SHA256：`7eeb678cb7384740d6c4235cb98cbe0aa162887b3afba75990bf220b146b90ba`。
- 实验协议SHA256：`399a0e1fd66621c7e92b1e314b3f6bb2bf51260374b8ff8a0d26e0fb80bb3955`。
- 训练：22完整会话、1514帧；开发：12完整会话、722帧，分为旧331帧和新增391帧；整会话划分，没有相邻帧随机拆分。本文件的失败请求来自新增391帧。
- 历史：每步可利用本会话从第0帧至当前帧的前缀，没有滑窗截断；训练会话长度34–153帧，开发28–109帧。所选会话87帧；**失败发生在第0帧，实际前序历史0帧，含当前输入的因果前缀长度1**。
- 状态：每会话初始化；跨request.changed保持GRU状态。批量输入右侧补零，损失仅使用真实帧mask；预测记录只写真实帧。模型为单向GRU，后续帧不是第0帧可用历史。
- 标准化：语义编码含固定裁剪/缩放，随后直接转float32；无数据拟合均值/方差或z-score。
- 优化：Adam lr=.003、betas=(.9,.999)、eps=1e-8、weight_decay=0；clip范数1；每epoch一次全批更新，真实帧等权joint9交叉熵。
- 实际成绩见下表；属于开发验证，非独立盲测。

| 数据组 | 行为及阶段同时正确 | 合资格完整请求全帧正确 | END正确/总数 | 误启动数 |
|---|---:|---:|---:|---:|
| 训练 | 1514/1514 | 84/84 | 7/7 | 0 |
| 旧开发 | 315/331 | 17/19 | 2/2 | 5 |
| 新增开发 | 324/391 | 2/8 | 0/4 | 12 |

成绩来源：`/Users/yang/project/malecns_lab/data/x2_model_redesign_20261024_r1/a_context_factor_v1/execution-r1/A-seed-24102431/metrics.json` 的 `train`、`old_development`、`context_factor_development`；运行身份来源：`/Users/yang/project/malecns_lab/data/x2_model_redesign_20261024_r1/a_context_factor_v1/execution-r1/A-seed-24102431/result.json` 和 `/Users/yang/project/malecns_lab/data/x2_model_redesign_20261024_r1/a_context_factor_v1/execution-r1/supervisor.json` 中 kind=A、seed=24102431 的条目。

源码版本：从该运行登记的研究目录原文件读取，以下六份代码的当前字节SHA256均与运行协议或数据标注身份绑定一致；不是后来修改的v2诊断代码，也没有借用另一个种子的模型。它们不是每个run单独复制的sources快照，因此本文件以哈希一致性确认内容版本，不以mtime推断。路径中的旧实验名仅表示复用的评分函数来源，不表示使用旧实验的模型、数据或状态重置规则。

| 以本段研究目录为根的源码路径 | SHA256 |
|---|---|
| `contract/teaching_v4.py` | `55844e7a5ea52e31bda4b49b1343c5c83338dd15ef7e2f7314af6f239a3a9272` |
| `a_context_factor_v1/worker.py` | `b4efc3ed37c10d6ecc75b81f8e48937292ff7419cda52bf72974ab859dd54d41` |
| `a_context_factor_v1/shared.py` | `ce4d9470b3ad26f17992d8fd830c5121cb1138bd18a31fb1cfc0d1d5058a6dd0` |
| `paired_frozen_v4_preparation_r2/shared.py` | `c6f75e5077183e49d1f7d49d19011971ba87b31e9dcbd004556e0069cb7167ed` |
| `a_negative_coverage_v4_r1/experiment.py` | `f8ff0d79f85b5cac84c16df54db0e30df444fec7052a192cfe6d78ac221d3a87` |
| `context_factor_data_v1/pipeline.py` | `eb40910d3c314b66e08085dd66f512e7451df03305168066b8f8e5ffabf96f4e` |

# 2. 正确答案如何生成

动作含义：HOLD待命/不发运动，ADVANCE前进，TURN_LEFT/RIGHT左右转，WAVE_RIGHT右手挥手；阶段START发起、CONTINUE继续、FINISH结束或保持等待。END_INTERACTION是请求任务，不是动作类别。下面原代码的target默认返回HOLD/FINISH，强度固定0.5。教师memory和annotation不作为模型输入。

来源：`/Users/yang/project/malecns_lab/data/x2_model_redesign_20261024_r1/contract/teaching_v4.py`；函数/常量：`TASKS/BEHAVIORS/PHASES/STATUSES/ACTIVE/FAILURE/PLANS/FRESH_SECONDS`；原始行号 18–28。以下逐字摘录，保留原缩进。

```python
SCHEMA = "malecns.x2.observed_compound_teaching.v4.prototype"
TASKS = ("GREET", "FACE_LEFT", "FACE_RIGHT", "SHOW_FORWARD", "END_INTERACTION")
BEHAVIORS = ("HOLD", "ADVANCE", "TURN_LEFT", "TURN_RIGHT", "WAVE_RIGHT")
PHASES = ("START", "CONTINUE", "FINISH")
STATUSES = ("IDLE", "ACCEPTED", "STARTED", "RUNNING", "COMPLETED", "CANCELLED", "REJECTED", "FAILED", "UNKNOWN")
ACTIVE = ("ACCEPTED", "STARTED", "RUNNING")
FAILURE = ("REJECTED", "FAILED", "UNKNOWN", "CANCELLED")
PLANS = {"GREET": ("WAVE_RIGHT",), "FACE_LEFT": ("TURN_LEFT", "WAVE_RIGHT"),
         "FACE_RIGHT": ("TURN_RIGHT", "WAVE_RIGHT"), "SHOW_FORWARD": ("ADVANCE", "WAVE_RIGHT"),
         "END_INTERACTION": ()}
FRESH_SECONDS = .75
```

来源：`/Users/yang/project/malecns_lab/data/x2_model_redesign_20261024_r1/contract/teaching_v4.py`；函数/常量：`teacher_step（完整函数，含memory更新和嵌套target）`；原始行号 113–168。以下逐字摘录，保留原缩进。

```python
def teacher_step(o, memory):
    """Offline annotations from exactly the observable history, no executor IO."""
    x = observable(o)
    if not memory or x["request_changed"]:
        memory.clear()
        memory.update(task=x["task"], completed=[], satisfied_primary=[], request_failed=False)
    assert memory["task"] == x["task"], "Task changed without observable request edge"
    if x["feedback_fresh"] and x["same_request"] and x["event_new"]:
        if x["status"] == "COMPLETED" and x["behavior"] != "HOLD":
            if x["behavior"] not in memory["completed"]:
                # v4: credit only the next skill supported by the previously
                # consumed visible prefix. Current post-completion heading
                # must not retroactively establish a primary prerequisite.
                # Missing prerequisite evidence is a contract failure here,
                # not proof of an actual physical execution-order violation.
                prior_remaining = [b for b in PLANS[x["task"]]
                    if b not in memory["completed"] and b not in memory["satisfied_primary"]]
                if not prior_remaining or x["behavior"] != prior_remaining[0]:
                    memory["request_failed"] = True
                else:
                    memory["completed"].append(x["behavior"])
        elif x["status"] in FAILURE:
            memory["request_failed"] = True
    def target(b="HOLD", p="FINISH", reason="", outcome="pending"):
        return {"behavior": b, "phase": p, "intensity": .5}, {"reason": reason, "request_outcome": outcome}
    if x["task"] == "END_INTERACTION":
        return target(reason="ordinary_end", outcome="end_intent")
    if x["protection_state"] != "READY":
        return target(reason="protection_latched", outcome="blocked")
    if not x["feedback_fresh"]:
        return target(reason="feedback_missing_or_stale")
    if memory["request_failed"]:
        return target(reason="request_failed_or_cancelled", outcome="aborted")
    plan = list(PLANS[x["task"]])
    # Once a fresh ready snapshot establishes that the initial relative turn
    # goal is already satisfied, later heading drift must not restart that turn
    # after waving. This is observable history, not an invented completion.
    if (plan and plan[0] in ("TURN_LEFT", "TURN_RIGHT") and x["status"] not in ACTIVE
            and x["standing"] and abs(x["heading_error_rad"]) <= .05):
        if plan[0] not in memory["satisfied_primary"]:
            memory["satisfied_primary"].append(plan[0])
    remaining = [b for b in plan if b not in memory["completed"] and b not in memory["satisfied_primary"]]
    if x["status"] in ACTIVE:
        if not x["same_request"]:
            return target(reason="cancel_previous_request")
        if not remaining or x["behavior"] != remaining[0] or x["behavior"] not in x["capabilities"]:
            memory["request_failed"] = True
            return target(reason="unexpected_or_unavailable_active_skill", outcome="aborted")
        return target(x["behavior"], "CONTINUE", "attributed_active_execution")
    if not x["standing"]:
        return target(reason="await_actual_standing")
    if not remaining:
        return target(reason="observed_compound_complete", outcome="completed")
    if remaining[0] not in x["capabilities"]:
        return target(reason="required_capability_unavailable", outcome="unsupported")
    return target(remaining[0], "START", "next_observed_compound_skill")
```

teacher_step依赖的observable、validate_observation在第3部分原样给出。教师记忆由实际标注循环按会话新建；以下两段是同一个函数的不连续原始节选，段间省略数据一致性检查，不是把两段拼成新代码。

来源：`/Users/yang/project/malecns_lab/data/x2_model_redesign_20261024_r1/context_factor_data_v1/pipeline.py`；函数/常量：`label_and_audit（逐会话记忆初始化、逐帧遍历节选）`；原始行号 135–150。以下逐字摘录，保留原缩进。

```python
    for is_new, splitmap in ((False,base["splits"]),(True,raw["splits"])):
        for split, sessions in splitmap.items():
            for original in sessions:
                s = copy.deepcopy(original)
                sid, group = s["session_id"], sid_group[s["session_id"]]
                counts = group_counts[group]
                memory, previous_request = {}, None
                cursor = previous_sequence = 0
                previous_event = None
                previous_time = -1.
                hashes = {"session":actor.digest([]), "request":actor.digest([])}
                source_completed, source_failed = [], []
                for i, f in enumerate(s["frames"]):
                    loc = dict(group=group, session_id=sid, frame_index=i)
                    o, ep = f["observation"], f["event_provenance"]
                    fb = o["feedback"]
```

来源：`/Users/yang/project/malecns_lab/data/x2_model_redesign_20261024_r1/context_factor_data_v1/pipeline.py`；函数/常量：`label_and_audit（调用教师与写入标签节选）`；原始行号 186–197。以下逐字摘录，保留原缩进。

```python
                    target, annotation = teacher.teacher_step(copy.deepcopy(o), memory)
                    if is_new:
                        check("target" not in f and "annotation" not in f, "raw_had_labels", loc)
                        f["target"], f["annotation"] = target, annotation
                        if f["actor_provenance"]["point"] == "factor_probe":
                            probes[-1].update(target=target, annotation=annotation)
                    else:
                        same = target == f["target"] and annotation == f["annotation"]
                        check(same, "old_target_annotation_changed", loc, old_target=f["target"],new_target=target,
                              old_annotation=f["annotation"],new_annotation=annotation)
                        old_comparison.append(dict(**loc, exact=same))
                    check(target["intensity"] == .5, "intensity_changed", loc)
```

未展开的依赖：label_and_audit中的check只记录一致性错误，copy.deepcopy为标准库深拷贝；raw生成与文件封存工具不参与本帧target函数，未递归纳入。teacher_step的全部状态更新已包含。本次不重新生成答案。

# 3. 模型实际上看到了什么、如何输出

输入边界、字段选择、失效反馈替代值以及全部44通道编码如下。只有frame["observation"]进入vector；target用于监督，session/request ID和来源/评分元数据不进入前向。

来源：`/Users/yang/project/malecns_lab/data/x2_model_redesign_20261024_r1/contract/teaching_v4.py`；函数/常量：`validate_observation（完整函数）`；原始行号 53–67。以下逐字摘录，保留原缩进。

```python
def validate_observation(o):
    assert set(o) == {"task", "request_changed", "capabilities", "protection_state", "feedback"}
    assert o["task"] in TASKS and type(o["request_changed"]) is bool
    assert o["protection_state"] in ("READY", "LATCHED", "RECOVERING")
    assert "HOLD" in o["capabilities"] and set(o["capabilities"]) <= set(BEHAVIORS)
    assert len(set(o["capabilities"])) == len(o["capabilities"])
    f = o["feedback"]
    assert set(f) == {"valid", "age_s", "status", "behavior", "same_request", "event_new", "standing",
                      "heading_error_rad", "vx_m_s", "wz_rad_s", "mc_mode"}
    assert f["status"] in STATUSES and f["behavior"] in BEHAVIORS
    assert all(type(f[k]) is bool for k in ("valid", "same_request", "event_new", "standing"))
    assert all(type(f[k]) in (int, float) and math.isfinite(f[k]) for k in
               ("age_s", "heading_error_rad", "vx_m_s", "wz_rad_s")) and f["age_s"] >= 0
    assert f["mc_mode"] in ("STAND_DEFAULT", "WALK", "unknown")
    return True
```

来源：`/Users/yang/project/malecns_lab/data/x2_model_redesign_20261024_r1/contract/teaching_v4.py`；函数/常量：`observable（完整函数）`；原始行号 70–88。以下逐字摘录，保留原缩进。

```python
def observable(o):
    """One information boundary shared by teacher and proposed sensory adapter.

    Identity *relationships* are observable facts. Actual IDs, family numbers,
    teacher state, target, future execution and simulator success never enter.
    """
    validate_observation(o)
    f = o["feedback"]
    fresh = f["valid"] and f["age_s"] <= FRESH_SECONDS
    return {"task": o["task"], "request_changed": o["request_changed"],
            "capabilities": tuple(o["capabilities"]), "protection_state": o["protection_state"],
            "feedback_valid": f["valid"], "feedback_fresh": fresh, "feedback_age_s": f["age_s"],
            "same_request": bool(fresh and f["same_request"]),
            "event_new": bool(fresh and f["event_new"]),
            "status": f["status"] if fresh else "UNKNOWN", "behavior": f["behavior"] if fresh else "HOLD",
            "standing": bool(fresh and f["standing"]),
            "heading_error_rad": f["heading_error_rad"] if fresh else 0.,
            "vx_m_s": f["vx_m_s"] if fresh else 0., "wz_rad_s": f["wz_rad_s"] if fresh else 0.,
            "mc_mode": f["mc_mode"] if fresh else "unknown"}
```

来源：`/Users/yang/project/malecns_lab/data/x2_model_redesign_20261024_r1/contract/teaching_v4.py`；函数/常量：`encode_channels（完整函数）`；原始行号 91–110。以下逐字摘录，保留原缩进。

```python
def encode_channels(o):
    """Semantic input only: these channels still require a versioned brain map."""
    x = observable(o)
    c = {"task."+t: float(x["task"] == t) for t in TASKS}
    c["request.changed"] = float(x["request_changed"])
    c.update({"capability."+b: float(b in x["capabilities"]) for b in BEHAVIORS})
    c.update({"protection."+p: float(x["protection_state"] == p) for p in ("READY", "LATCHED", "RECOVERING")})
    for k in ("feedback_valid", "feedback_fresh", "same_request", "event_new", "standing"):
        c[k] = float(x[k])
    c["feedback_age"] = min(x["feedback_age_s"] / (2*FRESH_SECONDS), 1.)
    c.update({"status."+s: float(x["status"] == s) for s in STATUSES})
    c.update({"behavior."+b: float(x["behavior"] == b) for b in BEHAVIORS})
    c.update({"mc."+m: float(x["mc_mode"] == m) for m in ("STAND_DEFAULT", "WALK", "unknown")})
    for k, scale in (("heading_error_rad", math.pi), ("vx_m_s", .5), ("wz_rad_s", 1.)):
        c[k+".positive"] = max(0., min(1., x[k]/scale))
        c[k+".negative"] = max(0., min(1., -x[k]/scale))
    # The threshold has a declared meaning; avoid a teacher-only predicate at
    # clipping/quantization boundaries. It is not a correct-action feature.
    c["heading.aligned"] = float(abs(x["heading_error_rad"]) <= .05)
    return c
```

联合类别顺序来自这次实际导入的共享评分模块；共9个合法行为/阶段组合，HOLD只允许FINISH。

来源：`/Users/yang/project/malecns_lab/data/x2_model_redesign_20261024_r1/paired_frozen_v4_preparation_r2/shared.py`；函数/常量：`BEHAVIORS/PHASES/CLASSES`；原始行号 13–15。以下逐字摘录，保留原缩进。

```python
BEHAVIORS = ['HOLD', 'ADVANCE', 'TURN_LEFT', 'TURN_RIGHT', 'WAVE_RIGHT']
PHASES = ['START', 'CONTINUE', 'FINISH']
CLASSES = [['HOLD', 'FINISH']] + [[b, p] for b in BEHAVIORS[1:] for p in PHASES[:2]]
```

以下是当前worker的模型、输入向量及批量序列构造；没有额外标准化或历史窗口截断。np和torch分别是本次记录的NumPy/PyTorch，未抄入库实现。names取本次protocol.json的input_channel_names，其顺序受代码assert检查。

来源：`/Users/yang/project/malecns_lab/data/x2_model_redesign_20261024_r1/a_context_factor_v1/worker.py`；函数/常量：`run：编码器绑定、A.__init__/forward、vector、tensors`；原始行号 25–50。以下逐字摘录，保留原缩进。

```python
    names = protocol['input_channel_names']
    encoder = module(ROOT / 'contract/teaching_v4.py', 'unchanged_semantic_encoder_only')
    class A(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.gru = torch.nn.GRU(44, 32, batch_first=True); self.head = torch.nn.Linear(32, 9)
        def forward(self, x, hidden=None):
            state, hidden = self.gru(x, hidden)
            return self.head(state), hidden
    model = A().cpu()
    assert sum(p.numel() for p in model.parameters()) == 7785
    def vector(frame):
        channels = encoder.encode_channels(frame['observation'])
        assert list(channels) == names
        return np.asarray(list(channels.values()), dtype=np.float32)
    def tensors(sessions):
        maximum = max(len(s['frames']) for s in sessions)
        x = torch.zeros(len(sessions), maximum, 44)
        y = torch.zeros(len(sessions), maximum, dtype=torch.long)
        mask = torch.zeros(len(sessions), maximum, dtype=torch.bool)
        for i, session in enumerate(sessions):
            for j, frame in enumerate(session['frames']):
                x[i, j] = torch.from_numpy(vector(frame))
                y[i, j] = CLASSES.index([frame['target']['behavior'], frame['target']['phase']])
                mask[i, j] = True
        assert torch.isfinite(x).all()
        return x, y, mask
```

来源：`/Users/yang/project/malecns_lab/data/x2_model_redesign_20261024_r1/a_context_factor_v1/shared.py`；函数/常量：`load_data：训练会话排序及帧连续性检查节选`；原始行号 92–97。以下逐字摘录，保留原缩进。

```python
    groups['train'] = sorted(source['splits']['train'], key=lambda s: hashlib.sha256(s['session_id'].encode()).hexdigest())
    groups = {key: groups[key] for key in EVAL_GROUPS}
    for sessions in groups.values():
        assert sessions
        for session in sessions:
            assert [f['frame_index'] for f in session['frames']] == list(range(len(session['frames'])))
```

来源：`/Users/yang/project/malecns_lab/data/x2_model_redesign_20261024_r1/a_context_factor_v1/worker.py`；函数/常量：`run：全序列前向与mask损失节选`；原始行号 84–92。以下逐字摘录，保留原缩进。

```python
        for epoch in range(epoch_start + 1, 301):
            model.train(); optimizer.zero_grad(set_to_none=True)
            t = time.perf_counter(); logits, _ = model(x)
            loss = torch.nn.functional.cross_entropy(logits[mask], y[mask])
            assert torch.isfinite(loss); timings['forward_loss_s'] += time.perf_counter() - t
            t = time.perf_counter(); loss.backward(); grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
            assert np.isfinite(grad); largest_grad = max(largest_grad, grad); timings['backward_s'] += time.perf_counter() - t
            t = time.perf_counter(); optimizer.step(); timings['optimizer_s'] += time.perf_counter() - t
            assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
```

下面原评估代码同时显示：批量预测及argmax输出转换；逐会话hidden=None；每帧hidden续接；真实帧评分；没有按request_changed清零。保存的logits来自批量前向，流式前向只验证argmax一致，未保存逐帧流式logits。

来源：`/Users/yang/project/malecns_lab/data/x2_model_redesign_20261024_r1/a_context_factor_v1/worker.py`；函数/常量：`run：模型评估、历史续接、类别转行为/阶段及计分`；原始行号 111–147。以下逐字摘录，保留原缩进。

```python
    model.eval(); predictions, metrics, logits_groups = {}, {}, {}
    evaluation_start = time.perf_counter()
    archived = read(contained(protocol['baselines'][str(seed)]['predictions']['path'])) if kind == 'baseline' else None
    matched_rows = 0
    with torch.no_grad():
        for group in selected_groups:
            sessions = groups[group]; sx, sy, mask = batches[group]
            t = time.perf_counter(); logits, _ = model(sx); batch_seconds = time.perf_counter() - t
            predicted = logits.argmax(-1); rows, encode_ns, model_ns = [], [], []
            old_rows = {(r['session_id'], r['frame_index']): r for r in archived[OLD_DEV_MAP[group]]['frames']} if archived and group in OLD_DEV_MAP else None
            for i, session in enumerate(sessions):
                hidden = None
                for j, frame in enumerate(session['frames']):
                    target, pair = CLASSES[int(sy[i, j])], CLASSES[int(predicted[i, j])]
                    row = {'session_id': session['session_id'], 'family_id': session['family_id'],
                           'frame_index': frame['frame_index'], 'request_index': frame['request_index'],
                           'task': frame['observation']['task'], 'expected_behavior': target[0], 'expected_phase': target[1],
                           'predicted_behavior': pair[0], 'predicted_phase': pair[1], 'logits': logits[i, j].tolist(),
                           'joint_correct': target == pair, 'false_motion': target[0] == 'HOLD' and pair[0] != 'HOLD',
                           'false_start': pair[0] != 'HOLD' and pair[1] == 'START' and target != pair}
                    if old_rows is not None:
                        old = old_rows[(row['session_id'], row['frame_index'])]
                        assert row == old, 'Baseline old-development raw prediction/logits must reproduce exactly'
                        matched_rows += 1
                    rows.append(row)
                    t0 = time.perf_counter_ns(); one = torch.from_numpy(vector(frame)).reshape(1, 1, 44); t1 = time.perf_counter_ns()
                    single, hidden = model(one, hidden); index = int(single[0, 0].argmax()); t2 = time.perf_counter_ns()
                    assert index == int(predicted[i, j])
                    encode_ns.append(t1 - t0); model_ns.append(t2 - t1)
            if old_rows is not None: assert len(rows) == len(old_rows)
            metrics[group], predictions[group] = score(rows, sessions, qualifications)
            metrics[group]['timing'] = {'batch_forward_seconds': batch_seconds, 'streaming_frames': len(rows),
                'input_prepare_tensor_ms': {'p50': float(np.percentile(encode_ns, 50) / 1e6), 'p95': float(np.percentile(encode_ns, 95) / 1e6)},
                'model_argmax_ms': {'p50': float(np.percentile(model_ns, 50) / 1e6), 'p95': float(np.percentile(model_ns, 95) / 1e6)},
                'scope': 'CPU semantic encoding/tensor and continuous GRU/head only; no brain, cache, robot or VM. Batch includes padded tails.'}
            metrics[group]['streaming_predictions_equal_batch'] = True
            logits_groups[group] = logits[mask].numpy()
```

相关未展开依赖：load_data按已保存dataset的session_id分组；训练排序/连续性已贴出，评分资格加载在第5部分。read/module/dump/sha为JSON读取、模块加载、写出、哈希工具；不补齐运行编排或库依赖。

# 4. 一个失败请求的真实记录

选择规则：本次固定运行在context_factor_development保存顺序中的第一个END错误，没有跨运行挑选。

会话：`62a5bc5e2560eee443a39710b58572b88f4d773053285c952de1e987c761b333`；family：`context-v1/development/composition-0`；请求：`62a5bc5e2560eee443a39710b58572b88f4d773053285c952de1e987c761b333/0`。

请求边界为 **[0,1)**，即只有第0帧；request_index=0，request_changed=true。第1帧开始下一请求，不是本帧可用历史。所选会话的模型前序历史为空，因此下面一帧就是该请求全部实际可用记录，不是省略前序后只摘出错点。

数据源标明 `real_execution_evidence=false`、模拟执行反馈；它是离线输入，不是机器人实测。

来源：`/Users/yang/project/malecns_lab/data/x2_model_redesign_20261024_r1/context_factor_data_v1/labeled/dataset.json`；JSON位置：`splits.development[session_id=62a5bc5e2560eee443a39710b58572b88f4d773053285c952de1e987c761b333].requests[0]`。以下为已保存数据的原值，只有排版调整（不是重新计算）；函数、源码行号不适用于数据对象。

```json
{
  "request_id": "62a5bc5e2560eee443a39710b58572b88f4d773053285c952de1e987c761b333/0",
  "task": "END_INTERACTION",
  "first_frame": 0,
  "expected_source_outcome": "predeclared_actor_context_probe",
  "actual_execution_success": null,
  "protocol_context": {
    "kind": "end",
    "task": "END_INTERACTION",
    "actor_behavior": "HOLD",
    "actor_status": "IDLE",
    "history": "none",
    "terminal": "retained",
    "case_id": "context-v1/development/composition-0/case-000",
    "role": "ordinary_end"
  },
  "end_frame": 1,
  "source_lifecycle": {
    "close_kind": "end_intent",
    "source_frame_range": [
      0,
      1
    ],
    "observed_new_event_sequences": [],
    "complete_robot_task_claimed": false,
    "measured": false
  }
}
```

来源：`/Users/yang/project/malecns_lab/data/x2_model_redesign_20261024_r1/context_factor_data_v1/labeled/dataset.json`；JSON位置：`splits.development[session_id=62a5bc5e2560eee443a39710b58572b88f4d773053285c952de1e987c761b333].frames[0]`。以下为已保存数据的原值，只有排版调整（不是重新计算）；函数、源码行号不适用于数据对象。

```json
{
  "frame_index": 0,
  "request_index": 0,
  "observation_time_s": 0.0,
  "observation": {
    "task": "END_INTERACTION",
    "request_changed": true,
    "capabilities": [
      "HOLD",
      "ADVANCE",
      "TURN_LEFT",
      "TURN_RIGHT",
      "WAVE_RIGHT"
    ],
    "protection_state": "READY",
    "feedback": {
      "valid": true,
      "age_s": 0.05,
      "standing": true,
      "heading_error_rad": 0.0,
      "vx_m_s": 0.0,
      "wz_rad_s": 0.0,
      "mc_mode": "STAND_DEFAULT",
      "status": "IDLE",
      "behavior": "HOLD",
      "same_request": false,
      "event_new": false
    }
  },
  "source": "simulated_execution_feedback_not_measured",
  "neural_consumed": true,
  "actor_provenance": {
    "case_id": "context-v1/development/composition-0/case-000",
    "point": "source_lifecycle"
  },
  "event_provenance": {
    "simulated": true,
    "event_sequence": 0,
    "event_owner_request_index": null,
    "last_fresh_neural_consumed_sequence_before": 0
  },
  "target": {
    "behavior": "HOLD",
    "phase": "FINISH",
    "intensity": 0.5
  },
  "annotation": {
    "reason": "ordinary_end",
    "request_outcome": "end_intent"
  }
}
```

来源：`/Users/yang/project/malecns_lab/data/x2_model_redesign_20261024_r1/a_context_factor_v1/execution-r1/A-seed-24102431/predictions.json`；JSON位置：`context_factor_development.frames[session_id=62a5bc5e2560eee443a39710b58572b88f4d773053285c952de1e987c761b333,frame_index=0]`。以下为已保存数据的原值，只有排版调整（不是重新计算）；函数、源码行号不适用于数据对象。

```json
{
  "session_id": "62a5bc5e2560eee443a39710b58572b88f4d773053285c952de1e987c761b333",
  "family_id": "context-v1/development/composition-0",
  "frame_index": 0,
  "request_index": 0,
  "task": "END_INTERACTION",
  "expected_behavior": "HOLD",
  "expected_phase": "FINISH",
  "predicted_behavior": "WAVE_RIGHT",
  "predicted_phase": "START",
  "logits": [
    1.8266533613204956,
    2.6984329223632812,
    -1.3284742832183838,
    0.09633910655975342,
    -3.342045545578003,
    -0.22573840618133545,
    -4.364263534545898,
    3.7117183208465576,
    -3.2778258323669434
  ],
  "joint_correct": false,
  "false_motion": true,
  "false_start": true
}
```

上述9项logits按第3部分CLASSES顺序保存：正确类别索引0的得分为1.8266533613204956；最高得分为索引7的3.7117183208465576，对应WAVE_RIGHT/START。这里只读取保存的logits，没有运行softmax或前向。

来源：`/Users/yang/project/malecns_lab/data/x2_model_redesign_20261024_r1/a_context_factor_v1/execution-r1/A-seed-24102431/predictions.json`；JSON位置：`context_factor_development.requests[request_id=62a5bc5e2560eee443a39710b58572b88f4d773053285c952de1e987c761b333/0]（有关字段摘录）`。以下为已保存数据的原值，只有排版调整（不是重新计算）；函数、源码行号不适用于数据对象。

```json
{
  "session_id": "62a5bc5e2560eee443a39710b58572b88f4d773053285c952de1e987c761b333",
  "request_id": "62a5bc5e2560eee443a39710b58572b88f4d773053285c952de1e987c761b333/0",
  "task": "END_INTERACTION",
  "first_frame": 0,
  "end_frame": 1,
  "observed_frames": 1,
  "visible_final_annotation": "end_intent",
  "fresh_standing_terminal_observed": false,
  "complete_request_evaluation_eligible": false,
  "eligibility_category": "end_intent",
  "all_frames_joint_correct": false,
  "wrong_frame_indices": [
    0
  ],
  "actual_execution_success": null
}
```

缺失字段明确列出：实际44维float32输入张量未逐帧保存；教师memory运行值未逐帧保存；GRU隐藏状态张量未保存；流式logits未保存；本帧单独loss、softmax概率、单帧实际计算时间未保存。初态规则hidden=None可由原代码确认，但不伪造未保存的数值张量。原始observation、target、批量logits、类别预测、评分标志及请求边界均有记录。

# 5. 为什么被判失败

实际调用链为本次worker中的score → 当前a_context_factor_v1/shared.py包装器 → 已绑定的paired_frozen_v4_preparation_r2/shared.py评分函数。第一段是实际包装器完整代码，第二段是实际核心评分完整代码；不是另写一套评分。

来源：`/Users/yang/project/malecns_lab/data/x2_model_redesign_20261024_r1/a_context_factor_v1/shared.py`；函数/常量：`score（完整包装器）`；原始行号 132–152。以下逐字摘录，保留原缩进。

```python
def score(rows, sessions, qualifications):
    metrics, predictions = SCORING.score(rows, sessions, qualifications, HELPER)
    factors = {}
    by_id = {s['session_id']: s for s in sessions}
    for row in rows:
        session = by_id[row['session_id']]
        request = session['requests'][row['request_index']]
        # Metadata is scoring-only. It never reaches vector(), model or loss.
        for scope, context in [('session', session.get('context_factors', {})),
                               ('request', request.get('protocol_context', {}))]:
            if not isinstance(context, dict):
                continue
            for key, value in context.items():
                name = scope + '.' + key + '=' + json.dumps(value, sort_keys=True)
                stat = factors.setdefault(name, {'frames': 0, 'joint_correct': 0, 'false_motion': 0, 'false_start': 0, 'errors': []})
                for field in ('joint_correct', 'false_motion', 'false_start'):
                    stat[field] += row[field]
                stat['frames'] += 1
                if not row['joint_correct']: stat['errors'].append(row)
    metrics['contextual_breakdown']['factor_fields'] = factors
    return metrics, predictions
```

来源：`/Users/yang/project/malecns_lab/data/x2_model_redesign_20261024_r1/paired_frozen_v4_preparation_r2/shared.py`；函数/常量：`score（完整函数，含gate）`；原始行号 159–199。以下逐字摘录，保留原缩进。

```python
def score(rows, sessions, qualifications, helper):
    """Shared scoring for A and B, retaining the accepted definition of failures."""
    cm = [[0] * 5 for _ in BEHAVIORS]
    pm = [[0] * 3 for _ in PHASES]
    requests, outcomes = [], {}
    by_session = {}
    for row in rows:
        cm[BEHAVIORS.index(row['expected_behavior'])][BEHAVIORS.index(row['predicted_behavior'])] += 1
        pm[PHASES.index(row['expected_phase'])][PHASES.index(row['predicted_phase'])] += 1
        by_session.setdefault(row['session_id'], []).append(row)
    for session in sessions:
        local = by_session[session['session_id']]
        assert len(local) == len(session['frames'])
        for request in session['requests']:
            selected = local[request['first_frame']:request['end_frame']]
            q = qualifications[(session['session_id'], request['request_id'])]
            assert q['observed_frames'] == len(selected)
            ok = bool(selected) and all(r['joint_correct'] for r in selected)
            outcome = request.get('expected_fixture_outcome', request.get('expected_source_outcome'))
            counter = outcomes.setdefault(outcome, {'total': 0, 'all_frames_joint_correct': 0})
            counter['total'] += 1; counter['all_frames_joint_correct'] += ok
            requests.append({'session_id': session['session_id'], **request, **q, 'all_frames_joint_correct': ok,
                             'wrong_frame_indices': [r['frame_index'] for r in selected if not r['joint_correct']]})
    eligible = [r for r in requests if r['complete_request_evaluation_eligible']]
    def gate(selected):
        return {'total': len(selected), 'hold_finish_correct': sum(r['predicted_behavior'] == 'HOLD' and r['predicted_phase'] == 'FINISH' for r in selected),
                'motion_predictions': sum(r['predicted_behavior'] != 'HOLD' for r in selected),
                'errors': [r for r in selected if r['predicted_behavior'] != 'HOLD' or r['predicted_phase'] != 'FINISH']}
    metrics = {'frames': len(rows), 'behavior_order': BEHAVIORS, 'phase_order': PHASES,
               'behavior_true_counts': [sum(row) for row in cm], 'behavior_predicted_counts': [sum(row[i] for row in cm) for i in range(5)],
               'behavior_confusion': cm, 'behavior_correct': [cm[i][i] for i in range(5)],
               'phase_confusion': pm, 'joint_correct': sum(r['joint_correct'] for r in rows),
               'joint_accuracy': sum(r['joint_correct'] for r in rows) / len(rows),
               'all_request_count': len(requests), 'request_outcomes': outcomes,
               'unobserved_request_count': sum(r['observed_frames'] == 0 for r in requests),
               'complete_requests': {'total': len(eligible), 'all_frames_joint_correct': sum(r['all_frames_joint_correct'] for r in eligible)},
               'end_checks': gate([r for r in rows if r['task'] == 'END_INTERACTION']),
               'finish_checks': gate([r for r in rows if r['expected_phase'] == 'FINISH']),
               'false_motion_count': sum(r['false_motion'] for r in rows), 'false_start_count': sum(r['false_start'] for r in rows),
               'joint_classes': helper.joint_metrics(rows), 'contextual_breakdown': helper.contextual_metrics(rows, sessions)}
    return metrics, {'frames': rows, 'requests': requests}
```

完整请求资格的来源如下：helper.completion_facts只读取已有会话/request的评分元数据；本次load_data对新增数据另用source_lifecycle和保存的资格表核对。这里并非采用旧helper实验的训练/推理重置规则。

来源：`/Users/yang/project/malecns_lab/data/x2_model_redesign_20261024_r1/a_negative_coverage_v4_r1/experiment.py`；函数/常量：`completion_facts（完整函数）`；原始行号 103–118。以下逐字摘录，保留原缩进。

```python
def completion_facts(session, request):
    """Scoring metadata only. Never determines input, sampling, target or weight."""
    rows = session['frames'][request['first_frame']:request['end_frame']]
    source = request.get('expected_fixture_outcome', request.get('expected_source_outcome'))
    last = rows[-1] if rows else None
    feedback = last['observation']['feedback'] if last else {}
    final_annotation = last.get('annotation', {}).get('request_outcome') if last else None
    aborted = any(f.get('annotation', {}).get('request_outcome') in ('aborted', 'blocked') for f in rows)
    terminal = bool(last and feedback.get('valid') and feedback.get('age_s', float('inf')) <= .75
                    and feedback.get('standing') and feedback.get('status') == 'COMPLETED')
    eligible = bool(rows and source in ('completed', 'prescribed_compound_completion')
                    and final_annotation == 'completed' and terminal and not aborted)
    return {'observed_frames': len(rows), 'source_intent_only': source,
            'visible_final_annotation': final_annotation,
            'fresh_standing_terminal_observed': terminal, 'observed_abort_or_blocked': aborted,
            'complete_request_evaluation_eligible': eligible, 'actual_execution_success': None}
```

来源：`/Users/yang/project/malecns_lab/data/x2_model_redesign_20261024_r1/a_context_factor_v1/shared.py`；函数/常量：`load_data：保存资格读取、核对及qualifications构造节选`；原始行号 98–129。以下逐字摘录，保留原缩进。

```python
    records = read(DATA / 'coverage-and-request-eligibility.json')['request_evaluation']
    saved = {(r['session_id'], r['request_index']): r for r in records}
    assert len(saved) == len(records)
    qualifications = {}
    old_ids = {s['session_id'] for v in old['splits'].values() for s in v}
    for session in by_id.values():
        for index, request in enumerate(session['requests']):
            facts = HELPER.completion_facts(session, request)
            record = saved[(session['session_id'], index)]
            eligible = record['complete_request_evaluation_eligible']
            assert type(eligible) is bool
            selected = session['frames'][request['first_frame']:request['end_frame']]
            assert selected == [f for f in session['frames'] if f['request_index'] == index]
            if session['session_id'] in old_ids:
                assert eligible == facts['complete_request_evaluation_eligible']
            else:
                # New actor intent strings are not evidence of completion.
                # Reviewed source lifecycle plus the existing visible terminal
                # conditions determine eligibility, independent of predictions.
                close_kind = request['source_lifecycle']['close_kind']
                assert close_kind in ('settled_terminal', 'interrupted', 'end_intent', 'blocked')
                no_abort = not any(f.get('annotation', {}).get('request_outcome') in
                                   ('aborted', 'blocked', 'interrupted') for f in selected)
                complete = bool(selected and close_kind == 'settled_terminal'
                                and facts['fresh_standing_terminal_observed']
                                and facts['visible_final_annotation'] == 'completed' and no_abort)
                assert eligible == complete
            qualifications[(session['session_id'], request['request_id'])] = {
                **facts, 'complete_request_evaluation_eligible': eligible,
                'eligibility_category': record['eligibility_category'],
                'reviewed_eligibility_record': record}
    return source, groups, qualifications
```

评分代码未展开的依赖：helper.joint_metrics和helper.contextual_metrics位于同一experiment.py的268–278、281–296行，分别生成联合混淆矩阵和分组计数；它们不修改输入rows、joint_correct或资格/END判定。BEHAVIORS、PHASES、CLASSES已在正文给出；by_id、old_ids、source来自已保存数据的分组加载，read只读取JSON。本次未执行这些函数。

已经确认的事实：

- 单帧错误：第0帧target为HOLD/FINISH，原始预测为WAVE_RIGHT/START；worker.py第124–130行的target==pair为false，false_motion与false_start均为true。
- END门槛失败：核心score第183–186、195行要求END输出HOLD/FINISH，该帧两项均不符合；已保存的end_checks将其列为错误。
- 请求一致性失败：该请求有1条记录，核心score第173–181行对其全部真实帧取all(joint_correct)，保存结果为all_frames_joint_correct=false、wrong_frame_indices=[0]。
- 完整请求指标必须分开：保存资格为complete_request_evaluation_eligible=false、eligibility_category=end_intent；因此这个END请求不进入新增开发的“2/8合资格完整请求”分子或分母，不能把它称为那8个请求之一。
- 记录完整性：所选请求的帧范围[0,1)与observed_frames=1一致；它位于会话起点，没有被遗漏的可见前序。缺失的张量/内部状态已单独标注，不等于缺少该请求的原始输入、标签和预测。
- 本文件仅能确认这一次离线END分类及请求一致性失败；没有机器人执行证据，也没有通过本次材料确定错误根因。

