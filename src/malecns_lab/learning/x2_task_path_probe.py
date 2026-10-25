"""Observation hooks for one fixed 100 ms input→lamina→DN diagnostic.

No runner, graph loader, policy identity, fitting, or device-command entrypoint.
"""
from contextlib import contextmanager
from dataclasses import replace
import time

from . import x2_inference as frozen

SCHEMA = "malecns_lab.x2_task_path_causal.v1"
PREFIX = "x2_behavior_short_trial_task_path_"
POLARITY_SCHEMA = "malecns_lab.x2_task_polarity_causal.v1"
POLARITY_PREFIX = "x2_behavior_short_trial_task_polarity_"
TASKS = ("GREET", "SHOW_FORWARD")
CONTEXTS = ("initial", "after_two_frames")


def configuration(mode="task-path"):
    frozen.require(mode in ("task-path","task-polarity"), "Unknown task-path mode")
    value = {"schema": SCHEMA, "wall_seconds": 180, "total_frames": 8, "context_replay_frames": 0,
        "task_rate_hz": 150, "contexts": list(CONTEXTS), "tasks": list(TASKS),
        "branch_steps": 1000, "branch_ms": 100, "frame_steps": 500, "bin_steps": 10, "bin_ms": 1,
        "graph_nodes": 166700, "graph_edges": 25582938,
        "stimulus": "one unchanged original observation held for 100ms; no fabricated feedback or new request",
        "state": "authenticated complete causal-r1 pre-state; continuous for both 50ms chunks; no reset",
        "baseline": "first 50ms DN, raw logits, and complete final state exactly match causal-r1",
        "targets": "all directly contacted L1/L2 for each task; unchanged full graph",
        "controls": "equal count per type/task, lowest body IDs not directly contacted by any of five task channels",
        "recording": "actual input spikes and stimulation events; lamina spikes/mean v/g; DN spikes in fixed 1ms bins; four 25ms DN voltage means",
        "parameters": "original encoder, theta and heads unchanged; no fitting or parameter search",
        "interpretation": {"primary": "whole 100ms DN spike-time and fixed-train-standardized voltage vectors; no best-bin/window selection",
            "minimum_cross_context_cosine": .2, "both_task_deltas_nonzero": True,
            "threshold_scope": "engineering evidence screen only, not biological or action qualification",
            "no_stable_information": "stop readout training on this fixed encoding/frozen-brain route; do not extend time or sweep gains",
            "stable_information": "localization evidence for review only; never automatic training or deployment"},
        "performance": {"step_offsets_per_branch": [0,500], "total_profiled_steps": 8,
            "segments": ["prior_gpu_wait", "input_schedule", "explicit_state_validation",
                         "inline_guards_and_dynamics", "matvec", "return_bookkeeping"],
            "limitations": "sampling adds synchronizations; inline guards and dynamics are not separately attributable; not uninstrumented throughput"},
        "resource": {"system_reserve_gib": 1, "minimum_safe_budget_gib": 2, "maximum_budget_gib": 6},
        "locked_test_opened": False, "automatic_deployment": False,
        "boundary": "known diagnostic contexts only; no learned action, independent sequence, or click acceptance claim"}
    if mode == "task-polarity":
        value.pop("task_rate_hz")
        value.update(schema=POLARITY_SCHEMA, scheduler_max_rate_hz=150.,
            task_occupancy="(1-one_hot)/4 on five original 60-port task groups only",
            nominal_task_occupancy_sum=60., actual_event_equality_assumed=False,
            baseline="authenticated completed task-path-r1 identity/arrays; zero baseline rerun frames",
            state="all archived causal-r1 arrays/clock unchanged; only policy identity rebound; no reset",
            parameters="only task occupancy polarity changed; scheduler, theta, graph and heads unchanged",
            baseline_replay_frames=0)
        value["interpretation"].update(
            no_stable_information="stop this polarity branch; no parameter sweep or follow-up training",
            stable_information="supports considering independent continuous-sequence training only; not automatic training/deployment")
    return value


def selection(graph, encoder):
    """Select observed cells using topology/type only, before reading responses."""
    import numpy as np
    channels = encoder.identity["channels"]
    assignments = np.asarray(encoder.channel_indices)
    inputs = np.asarray(encoder.input_indices)
    task_channels = [i for i,name in enumerate(channels) if name.startswith("task.")]
    frozen.require(len(task_channels) == 5, "Expected all original task channels")
    all_task_posts = set(graph.counts[:,inputs[np.isin(assignments,task_channels)]].indices.tolist())
    groups, selected = {}, set()
    task_ports = {}
    for task in TASKS:
        ports = np.flatnonzero(assignments == channels.index("task."+task))
        task_ports[task] = ports.tolist()
        posts = set(graph.counts[:,inputs[ports]].indices.tolist())
        for cell_type in ("L1", "L2"):
            members = np.flatnonzero(graph.group_index == graph.group_names.index(cell_type))
            contacted = [int(i) for i in members if i in posts]
            untouched = [int(i) for i in members if i not in all_task_posts]
            frozen.require(contacted and len(untouched) >= len(contacted), "Insufficient lamina targets or untouched controls")
            for kind, indices in (("targets",contacted), ("controls",untouched[:len(contacted)])):
                groups[f"{task}.{cell_type}.{kind}"] = indices
                selected.update(indices)
    ordered = sorted(selected)
    lookup = {index:i for i,index in enumerate(ordered)}
    return {"selected_graph_indices": ordered, "selected_body_ids": graph.body_ids[ordered].tolist(),
        "groups": {name:{"graph_indices":indices, "body_ids":graph.body_ids[indices].tolist(),
                          "columns":[lookup[i] for i in indices]} for name,indices in groups.items()},
        "task_input_ports": task_ports, "input_body_ids": list(encoder.input_body_ids),
        "control_selection": configuration()["controls"]}


class PathCollector:
    def __init__(self, policy, manifest, state):
        import torch
        self.start_step, self.steps = state.brain.step, 0
        self.inputs, self.dn = policy.input_indices, policy.dn_indices
        self.posts = torch.tensor(manifest["selected_graph_indices"], dtype=torch.int64, device=self.inputs.device)
        self.last_events = state.event_counts
        self.values = {"input_spikes":torch.zeros((100,len(self.inputs)),dtype=torch.int64,device=self.inputs.device),
            "input_events":torch.zeros((100,len(self.inputs)),dtype=torch.int64,device=self.inputs.device),
            "dn_spikes":torch.zeros((100,len(self.dn)),dtype=torch.int64,device=self.inputs.device),
            "post_spikes":torch.zeros((100,len(self.posts)),dtype=torch.int64,device=self.inputs.device),
            "post_v":torch.zeros((100,len(self.posts)),dtype=torch.float32,device=self.inputs.device),
            "post_g":torch.zeros((100,len(self.posts)),dtype=torch.float32,device=self.inputs.device)}

    def capture(self, brain, events, spikes):
        frozen.require(brain.step == self.start_step+self.steps+1 and self.steps < 1000, "Unexpected observed neural clock")
        bucket = self.steps//10
        values = self.values
        values["input_spikes"][bucket].add_(spikes[self.inputs].to(values["input_spikes"].dtype))
        values["input_events"][bucket].add_(events-self.last_events)
        values["dn_spikes"][bucket].add_(spikes[self.dn].to(values["dn_spikes"].dtype))
        values["post_spikes"][bucket].add_(spikes[self.posts].to(values["post_spikes"].dtype))
        values["post_v"][bucket].add_(brain.v[self.posts])
        values["post_g"][bucket].add_(brain.g[self.posts])
        self.last_events, self.steps = events, self.steps+1

    def arrays(self):
        frozen.require(self.steps == 1000, "Incomplete fixed 100ms observation")
        result = {name:value.cpu().numpy().copy() for name,value in self.values.items()}
        for name in ("post_v", "post_g"):
            result[name] /= 10
        return result


@contextmanager
def observed_steps(policy, collector, timings):
    """Read-only step outputs; profile just 2/1000 existing steps per branch.

    Explicit synchronization before profiling isolates previous queued work.
    The original core functions and tensors are never rewritten or substituted.
    """
    import torch
    from ..brain import mps_trainable
    sentinel = object()
    saved_step = policy.__dict__.get("_step", sentinel)
    saved_validate = policy.brain.__dict__.get("_validate_state", sentinel)
    original_step, original_validate = policy._step, policy.brain._validate_state
    original_matvec = mps_trainable.fixed_mps_matvec
    active = None
    def validate(state):
        if active is None:
            return original_validate(state)
        torch.mps.synchronize()
        began = time.perf_counter()
        active["input_schedule_s"] = began-active["begin"]
        result = original_validate(state)
        torch.mps.synchronize()
        active["validation_end"] = time.perf_counter()
        active["explicit_state_validation_s"] = active["validation_end"]-began
        return result
    def matvec(*args):
        if active is None:
            return original_matvec(*args)
        torch.mps.synchronize()
        began = time.perf_counter()
        active["inline_guards_and_dynamics_s"] = began-active["validation_end"]
        result = original_matvec(*args)
        torch.mps.synchronize()
        active["matvec_end"] = time.perf_counter()
        active["matvec_s"] = active["matvec_end"]-began
        return result
    def step(*args):
        nonlocal active
        offset = collector.steps
        if offset in (0,500):
            began = time.perf_counter()
            torch.mps.synchronize()
            now = time.perf_counter()
            active = {"step_offset":offset,"prior_gpu_wait_s":now-began,"begin":now}
        output = original_step(*args)
        if active is not None:
            ended = time.perf_counter()
            active["return_bookkeeping_s"] = ended-active["matvec_end"]
            active["observed_step_s_excluding_prior_wait"] = ended-active["begin"]
            timings.append({k:v for k,v in active.items() if k not in ("begin","validation_end","matvec_end")})
            active = None
        brain, phase, events, spikes = output
        collector.capture(brain,events,spikes)
        return output
    policy._step, policy.brain._validate_state = step, validate
    mps_trainable.fixed_mps_matvec = matvec
    try:
        yield
    finally:
        if saved_step is sentinel:
            del policy._step
        else:
            policy._step = saved_step
        if saved_validate is sentinel:
            del policy.brain._validate_state
        else:
            policy.brain._validate_state = saved_validate
        mps_trainable.fixed_mps_matvec = original_matvec


def continue_observation(policy, observation, state):
    """Second 50ms of the same held stimulus, not a fabricated next observation."""
    import torch
    from .x2_behavior_policy import BehaviorOutput
    policy.validate_state(state)
    occupancy = torch.from_numpy(policy.encoder.encode(observation)).to(state.phase.device)
    brain, phase, counts = state.brain, state.phase, state.event_counts
    sums = [torch.zeros(len(policy.dn_indices),device=state.phase.device) for _ in range(2)]
    for step in range(500):
        brain, phase, counts, _ = policy._step(occupancy,brain,phase,counts)
        half = step//250
        sums[half] = sums[half]+brain.v[policy.dn_indices]
    features = torch.cat(sums)/250
    next_state = replace(state,brain=brain,phase=phase,event_counts=counts,frame_index=state.frame_index+1)
    policy.validate_state(next_state)
    behavior, phases, raw, intensity = policy.read_features(features)
    frozen.require(all(torch.isfinite(x).all() for x in (features,behavior,phases,raw)), "Nonfinite continuation")
    return BehaviorOutput(behavior,phases,raw,intensity,features,next_state)


def effect(matrix):
    """All predetermined dimensions; no fitted projection or selected time bin."""
    import numpy as np
    values = np.asarray(matrix,np.float64).reshape(2,2,-1)
    frozen.require(np.isfinite(values).all(), "Nonfinite observed signal")
    delta, state_delta = values[:,1]-values[:,0], values[1]-values[0]
    lengths = np.linalg.norm(delta,axis=1)
    denominator = float(lengths.prod())
    state_rms = float(np.sqrt(np.mean(state_delta**2)))
    return {"task_rms":np.sqrt(np.mean(delta**2,axis=1)).tolist(),
        "task_nonzero":np.count_nonzero(delta,axis=1).tolist(),
        "task_cosine_between_contexts":float(delta[0]@delta[1]/denominator) if denominator else None,
        "state_rms":state_rms,
        "task_to_state_rms_ratio":float(np.sqrt(np.mean(delta**2))/state_rms) if state_rms else None}


def measurements(arrays, manifest, center, scale):
    import numpy as np
    observed = {name:effect(arrays[name]) for name in ("input_events","input_spikes","dn_spikes")}
    observed["lamina_groups"] = {name:{signal:effect(arrays[signal][...,group["columns"]])
        for signal in ("post_spikes","post_v","post_g")} for name,group in manifest["groups"].items()}
    observed["dn_voltage_0_50ms"] = effect(arrays["features_mv"][:,:,0])
    observed["dn_voltage_50_100ms"] = effect(arrays["features_mv"][:,:,1])
    frozen.require(np.shape(center)==np.shape(scale)==(2628,) and np.isfinite(center).all()
        and np.isfinite(scale).all() and np.all(scale>=.1), "Fixed train normalization required")
    observed["dn_voltage_whole_100ms_standardized"] = effect((arrays["features_mv"]-center)/scale)
    delta = arrays["dn_spikes"][:,1]-arrays["dn_spikes"][:,0]
    observed["first_dn_task_difference_bin_ms"] = [int(np.flatnonzero(np.any(d!=0,axis=1))[0])
        if np.any(d!=0) else None for d in delta]
    coherent = {name:all(x>0 for x in observed[name]["task_nonzero"])
        and observed[name]["task_cosine_between_contexts"] is not None
        and observed[name]["task_cosine_between_contexts"] >= configuration()["interpretation"]["minimum_cross_context_cosine"]
        for name in ("dn_spikes","dn_voltage_whole_100ms_standardized")}
    observed["route_evidence"] = {"whole_window_coherence_in_two_known_contexts":coherent,
        "stop_current_readout_route":not any(coherent.values()), "automatic_followup_training":False,
        "deployment_qualified":False, "action_learned":False}
    return observed


def run_probe(policy, states, observations, archived, comparison, directory, deadline, completed, phase,
              *, mode="task-path", baseline_evidence=None):
    """Called only by the existing authenticated/supervised causal worker."""
    import numpy as np
    import torch
    from .resource_guard import write_json
    from .x2_behavior_training import _prediction
    config = configuration(mode)
    polarity = mode == "task-polarity"
    frozen.require(not polarity or baseline_evidence is not None, "Authenticated baseline required for polarity")
    manifest = selection(policy.brain.graph,policy.encoder)
    write_json(directory/"selection.json",manifest)
    branches, evidence = [], []
    for context_index,(context,state) in enumerate(zip(CONTEXTS,states)):
        before = policy.state_identity(state)
        expected = comparison["contexts"][context_index]["prestate"]
        if polarity:
            # Restore originally under the strict parent identity, then bind the
            # exact same numerical state to this explicit input intervention.
            expected = {k:v for k,v in expected.items() if k != "state_id"}
            expected["policy_id"] = policy.policy_id
            expected["state_id"] = frozen.canonical_hash(expected)
        frozen.require(before == expected, "Original fixed state changed")
        context_arrays, context_evidence = [], []
        for task_index,observation in enumerate(observations[:2]):
            frozen.require(time.monotonic()<deadline and policy.state_identity(state)==before, "Budget exhausted or pre-state polluted")
            probe, timings = PathCollector(policy,manifest,state), []
            outputs, frame_evidence = [], []
            with torch.no_grad(), observed_steps(policy,probe,timings):
                current = state
                for frame in range(2):
                    frozen.require(time.monotonic()<deadline, "Wall budget exhausted before required continuation")
                    phase("fixed_task_path_branch",context=context,task=str(observation.task),frame=frame)
                    began = time.monotonic()
                    output = (policy(observation,current,checkpoint_blocks=False) if frame==0
                              else continue_observation(policy,observation,current))
                    torch.mps.synchronize()
                    wall = time.monotonic()-began
                    completed()
                    actual = output.features_mv.cpu().numpy().copy()
                    raw = _prediction((output.behavior_logits,output.phase_logits,output.parameter_raw,output.intensity))
                    identity = policy.state_identity(output.state)
                    if frame == 0 and not polarity:
                        old = comparison["contexts"][context_index]["branches"][task_index]
                        frozen.require(np.array_equal(actual,archived["features_mv"][context_index,task_index])
                            and raw==old["raw"] and identity==old["result_state"], "First 50ms failed exact causal reproduction")
                    outputs.append(actual)
                    frame_evidence.append({"frame":frame,"raw":raw,"result_state":identity,"wall_seconds":wall,
                        "same_observation_seq":observation.observation_seq})
                    current = output.state
            frozen.require(policy.state_identity(state)==before and current.brain.step==state.brain.step+1000
                and current.observation_seq==observation.observation_seq, "Continuation polluted pre-state or changed observation identity")
            arrays = probe.arrays()
            frozen.require(np.array_equal(arrays["input_events"].sum(0),
                (current.event_counts-state.event_counts).cpu().numpy()), "Observed events differ from actual sensory state")
            arrays["features_mv"] = np.asarray(outputs)
            context_arrays.append(arrays)
            context_evidence.append({"task":str(observation.task),"before":before,"prestate_unchanged":True,
                "frames":frame_evidence,"performance_samples":timings})
            del output,current,probe
        branches.append(context_arrays)
        evidence.append({"context":context,"branches":context_evidence})
    arrays = {name:np.asarray([[item[name] for item in context] for context in branches]) for name in branches[0][0]}
    observed = measurements(arrays,manifest,archived["fixed_train_center"],archived["fixed_train_scale"])
    baseline_record = {"baseline_four_frames_exact":True}
    if polarity:
        arrays["encoded_occupancy"] = np.asarray([policy.encoder.encode(o) for o in observations[:2]])
        ports = list(policy.encoder.task_ports)
        observed["actual_task_input"] = {name:{"all_100ms":arrays[name][...,ports].sum(axis=(-1,-2)).tolist(),
            "first_50ms":arrays[name][:,:,:50,ports].sum(axis=(-1,-2)).tolist(),
            "second_50ms":arrays[name][:,:,50:,ports].sum(axis=(-1,-2)).tolist()}
            for name in ("input_events","input_spikes")}
        observed["polarity_screen"] = {"supports_considering_continuous_training":
            not observed["route_evidence"]["stop_current_readout_route"],
            "stop_polarity_branch":observed["route_evidence"]["stop_current_readout_route"],
            "automatic_training":False,"automatic_deployment":False}
        baseline_record = {"baseline_replayed":False,"baseline_binding":baseline_evidence}
    with (directory/"responses.npz").open("xb") as stream:
        np.savez_compressed(stream,**arrays,fixed_train_center=archived["fixed_train_center"],
            fixed_train_scale=archived["fixed_train_scale"])
    write_json(directory/"comparison.json", {"configuration":config,"policy_identity":policy.identity,
        "selection_sha256":frozen.sha256(directory/"selection.json"),"conditions":evidence,"measurements":observed,
        "arrays_sha256":frozen.sha256(directory/"responses.npz"),**baseline_record,
        "observation_axes":["context","task","1ms_bin","cell"],
        "voltage_axes":["context","task","50ms_chunk","two_25ms_DN_halves"],
        "performance_limitations":config["performance"]["limitations"]})
    return {"reason":"fixed_100ms_task_polarity_measured" if polarity else "fixed_100ms_task_path_measured",
            "measurements":observed,**baseline_record}
