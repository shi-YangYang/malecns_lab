"""One bounded theta-only update; preview never loads a brain."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

from . import x2_inference as frozen
from .x2_behavior_trial import MINIMUM_BUDGET, output_directory as guarded_output
from .x2_interaction_trial import score_requests
from .x2_interaction_paired import OWN_SOURCES as PAIRED_SOURCES, branch_observations, pair_targets
from .x2_theta_trial_policy import bind_theta_trial_policy

SCHEMA = "malecns_lab.x2_theta_trial.v1"
PREFIX = "x2_behavior_short_trial_theta_"
VALIDATION_SEED = 24102401
VALIDATION_FAMILY = {"yaw": 1.4, "source_age": .06, "turn": .075, "inference_s": 7.4}
VALIDATION_ORDER = (("GREET", False), ("SHOW_FORWARD", False), ("FACE_RIGHT", False),
    ("END_INTERACTION", False), ("SHOW_FORWARD", False), ("SHOW_FORWARD", True),
    ("END_INTERACTION", False), ("FACE_LEFT", False))
INPUT_FILES = ("registration.json", "summary.json", "dataset.json", "train-features.npz",
    "paired-observations.json", "paired-train-features.npz", "initial-state.json", "initial-state.npz",
    "checkpoint/checkpoint.json", "checkpoint/arrays.npz")
OWN_SOURCES = (*PAIRED_SOURCES, "learning/x2_theta_trial.py", "learning/x2_theta_trial_policy.py")


def configuration():
    return {"schema": SCHEMA, "wall_seconds": 600, "total_neural_frames": 48,
        "prefix_replay_frames": 8, "gradient_frames": 8, "train_after_frames": 8,
        "validation_before_frames": 12, "validation_after_frames": 12,
        "decision_steps": 500, "precondition_steps_each": 600, "preconditions": 2,
        "branches": 6, "maximum_credit_steps": 1000, "theta_updates": 1,
        "optimizer": {"name": "Adam", "lr": 1e-4, "gradient_l2_cap": 1.},
        "surrogate_gain": .01, "heads": "six parameters frozen", "loss": "equal branch means of behavior CE plus phase CE",
        "graph_nodes": 166700, "graph_edges": 25582938,
        "validation_seed": VALIDATION_SEED, "validation_family": VALIDATION_FAMILY,
        "validation_order": [list(x) for x in VALIDATION_ORDER],
        "validation_requests": {"total": 8, "forward": 3, "end": 2},
        "resource": {"system_reserve_gib": 1, "minimum_safe_budget_gib": 2, "maximum_budget_gib": 6},
        "screen": "train CE and validation CE decrease; correct requests increase; forward/end/greet do not regress",
        "credit_limit": "combined loss boundary gradient does not isolate final-frame credit; prefix history detached",
        "coverage_limits": ["simulated feedback; fixed observations do not react to mistakes", "no WAVE; GREET is HOLD/FINISH",
            "one update only", "conditional train pre-states remain source-theta states", "no ordinary CONTINUE samples"],
        "qualification_only": True, "automatic_deployment": False, "locked_test_opened": False}


def output_directory(output, root, **kwargs):
    frozen.require(Path(output).name.startswith(PREFIX) and len(Path(output).name) > len(PREFIX), "Dedicated theta output prefix required")
    return guarded_output(output, root, **kwargs)


def trial_dataset(original):
    from .x2_interaction_data import _session
    from ..control.x2_behavior import digest
    value = deepcopy(original)
    value["splits"]["validation"] = [_session("validation", "theta-validation", value["identity"]["profile"],
        VALIDATION_SEED, order=VALIDATION_ORDER, family=VALIDATION_FAMILY)]
    identity = value["identity"]
    identity["source_dataset_id"] = identity.pop("dataset_id")
    identity["split_sha256"] = {k:digest(v) for k,v in value["splits"].items()}
    identity["theta_trial_sha256"] = frozen.sha256(Path(__file__))
    identity["session_families"] = {"train": ["train-a", "train-b"], "validation": ["theta-validation"]}
    identity["dataset_id"] = digest(identity)
    frozen.require([len(e["frames"]) for e in value["splits"]["train"]] == [12,12]
        and len(value["splits"]["validation"][0]["frames"]) == 12, "Registered frame counts differ")
    return value


def training_branches(episode):
    from ..control.x2_behavior import Observation
    first = branch_observations(Observation.from_dict(episode["frames"][4]["observation"]))
    targets = pair_targets(first)
    branches = []
    for index, observation in enumerate(first):
        rows = [{"observation": observation.to_dict(), "target": targets[index]}]
        if index == 0:
            second = Observation.from_dict(episode["frames"][5]["observation"])
            second = replace(second, request_id=observation.request_id,
                feedback=replace(second.feedback, request_id=observation.request_id))
            rows.append({"observation": second.to_dict(), "target": episode["frames"][5]["target"]})
        branches.append({"family": episode["family_id"], "task": str(observation.task), "frames": rows})
    # Canonical JSON representation matches the sealed observations (whose
    # tuples/enums have already been serialized), without changing their values.
    return json.loads(json.dumps(branches))


def row_ce(row, target):
    from ..control.x2_behavior import BEHAVIORS, PHASES
    total = 0.
    for key, choices in (("behavior", BEHAVIORS), ("phase", PHASES)):
        logits = row[key+"_logits"]
        maximum = max(logits)
        total += maximum + math.log(sum(math.exp(v-maximum) for v in logits)) - logits[tuple(map(str,choices)).index(target[key])]
    return total


def train_ce(rows, branches):
    cursor, losses = 0, []
    for branch in branches:
        count = len(branch["frames"])
        losses.append(sum(row_ce(r,f["target"]) for r,f in zip(rows[cursor:cursor+count],branch["frames"]))/count)
        cursor += count
    frozen.require(cursor == len(rows) == 8 and len(branches) == 6, "Incomplete train comparison")
    return sum(losses)/len(losses)


def validation_score(episode, rows):
    score = score_requests([episode], rows)
    greet = [r for r in score["requests"] if r["task"] == "GREET"]
    score["greet_sequences"] = sum(r["raw_sequence_match"] for r in greet)
    score["greet_false_advance_requests"] = sum(any(x["prediction"]["behavior"] == "ADVANCE" for x in r["raw"]) for r in greet)
    score["greet_false_motion_requests"] = sum(any(x["prediction"]["behavior"] != "HOLD" for x in r["raw"]) for r in greet)
    score["ce"] = sum(row_ce(r,f["target"]) for r,f in zip(rows,episode["frames"]))/len(rows)
    return score


def intensity_report(rows, targets):
    active = [(r["intensity"]-t["intensity"])**2 for r,t in zip(rows,targets)
        if t["behavior"] in ("ADVANCE","TURN_LEFT","TURN_RIGHT") and t["phase"] in ("START","CONTINUE")]
    return {"minimum": min(r["intensity"] for r in rows), "maximum": max(r["intensity"] for r in rows),
        "active_target_mse": sum(active)/len(active) if active else None, "active_frames": len(active)}


def learning_screen(train_before, train_after, before, after):
    checks = {"train_ce_decreased": train_after < train_before, "validation_ce_decreased": after["ce"] < before["ce"],
        "correct_requests_increased": after["request_sequence_matches"] > before["request_sequence_matches"],
        "forward_not_regressed": all(after[k] >= before[k] for k in ("forward_starts","forward_sequences")),
        "end_not_regressed": after["end_sequences"] >= before["end_sequences"],
        "greet_not_regressed": after["greet_sequences"] >= before["greet_sequences"]
            and after["greet_false_motion_requests"] <= before["greet_false_motion_requests"]}
    return {"checks": checks, "supports_further_internal_learning": all(checks.values()), "deployment_qualified": False}


def execute(output, source_run, *, root=frozen.ROOT):
    import psutil
    from .resource_guard import resource_budget, supervise, write_json
    from .x2_behavior_training import inspect_checkpoint, source_inventory
    from .x2_behavior_session import checkpoint_registration_metadata
    from .x2_interaction_paired_policy import SCHEMA as PAIRED_SCHEMA
    root, source_run = Path(root).resolve(), Path(source_run).resolve()
    frozen.require(source_run.is_relative_to(root/"data"), "Archived project source required")
    output = output_directory(output, root, inputs=(source_run,))
    started = time.monotonic()
    with frozen.numerical_lock(root):
        preflight = frozen.simulator_preflight(root)
        resources = resource_budget(psutil.virtual_memory().available, reserve_bytes=frozen.GIB)
        frozen.require(resources["effective_rss_limit_bytes"] >= MINIMUM_BUDGET, "Insufficient safe complete-brain memory")
        metadata = inspect_checkpoint(source_run/"checkpoint")
        source_reg = frozen.read_json(source_run/"registration.json")
        frozen.bound(source_reg,"registration_id")
        source_summary = frozen.read_json(source_run/"summary.json")
        original = frozen.read_json(source_run/"dataset.json")
        frozen.bound(original["identity"],"dataset_id")
        frozen.require(metadata["policy"]["schema"] == PAIRED_SCHEMA
            and metadata["context"]["registration_id"] == source_reg["registration_id"]
            and metadata["context"]["dataset_id"] == source_reg["dataset_id"] == original["identity"]["dataset_id"]
            and frozen.sha256(source_run/"dataset.json") == source_reg["dataset_sha256"]
            and source_summary["status"] == "completed"
            and source_summary["worker"]["completed_frames"] == 54
            and source_summary["worker"]["checkpoint_id"] == metadata["checkpoint_id"], "Paired source provenance differs")
        frozen.require(original["identity"]["profile"] == metadata["policy"]["interaction_profile"]
            and not original["identity"]["profile"]["wave_enabled"], "Registered no-WAVE profile required")
        dataset = trial_dataset(original)
        output.mkdir(exist_ok=False)
        sources = source_inventory(root)
        sources.update({n:frozen.sha256(root/"src/malecns_lab"/n) for n in OWN_SOURCES})
        for name in sources:
            target = output/"sources/malecns_lab"/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root/"src/malecns_lab"/name,target)
            target.chmod(0o444)
        write_json(output/"dataset.json",dataset)
        reg = {"configuration":configuration(),"root":str(root),"output":str(output),"parent_pid":os.getpid(),
            "deadline_monotonic":started+600,"source_run":str(source_run),"checkpoint":str(source_run/"checkpoint"),
            "checkpoint_id":metadata["checkpoint_id"],"checkpoint_metadata":checkpoint_registration_metadata(metadata),
            "source_input_sha256":{n:frozen.sha256(source_run/n) for n in INPUT_FILES},
            "dataset_id":dataset["identity"]["dataset_id"],"dataset_sha256":frozen.sha256(output/"dataset.json"),
            "source_sha256":sources,"resources":resources}
        reg["registration_id"] = frozen.canonical_hash(reg)
        write_json(output/"registration.json",reg)
        write_json(output/"simulator_preflight.json",preflight)
        remaining = reg["deadline_monotonic"]-time.monotonic()
        frozen.require(remaining>0,"Preparation exhausted hard budget")
        supervisor = supervise([sys.executable,"-m","malecns_lab.learning.x2_theta_trial","--_worker",str(output)],
            output,rss_limit_bytes=resources["effective_rss_limit_bytes"],wall_seconds=remaining,reserve_bytes=frozen.GIB)
        worker = frozen.read_json(output/"worker_result.json") if (output/"worker_result.json").exists() else {}
        result = {"status":"completed" if supervisor["status"] == worker.get("status") == "completed" else "failed",
            "output":str(output),"registration_id":reg["registration_id"],"supervisor":supervisor,"worker":worker,
            "wall_seconds":time.monotonic()-started}
        write_json(output/"summary.json",result)
        return result


def run_worker(directory):
    from .resource_guard import write_json
    directory = Path(directory).absolute()
    frozen.require(not directory.is_symlink() and directory.resolve() == directory,"No worker symlink")
    reg = frozen.read_json(directory/"registration.json")
    frozen.bound(reg,"registration_id")
    frozen.require(reg["parent_pid"] == os.getppid() and reg["output"] == str(directory),"Supervising parent required")
    root, source = Path(reg["root"]), Path(reg["source_run"])
    output_directory(directory,root,created=True,inputs=(source,))
    frozen.require(reg["configuration"] == configuration(),"Registered protocol changed")
    with (root/"data/.continuous-training.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(lock.fileno(),fcntl.LOCK_UN)
            raise ValueError("Parent must own the single-brain lock")
    with (directory/"worker.claim").open("x") as stream:
        stream.write(str(os.getpid()))
    started = time.monotonic()
    result = {"status":"failed","reason":None,"registration_id":reg["registration_id"],
        "completed_frames":0,"required_frames":48,"optimizer_steps":0,
        "qualification_only":True,"automatic_deployment":False,"locked_test_opened":False}
    def phase(name,**details):
        frozen.require(time.monotonic()<reg["deadline_monotonic"],"Hard wall budget exhausted")
        write_json(directory/"phase.json",{"phase":name,"wall_seconds":time.monotonic()-started,
            "completed_frames":result["completed_frames"],"required_frames":48,**details})
    def audit_sources():
        for name, expected in reg["source_sha256"].items():
            frozen.require(frozen.sha256(root/"src/malecns_lab"/name) == expected
                and frozen.sha256(directory/"sources/malecns_lab"/name) == expected,"Registered source changed")
        for name, expected in reg["source_input_sha256"].items():
            frozen.require(frozen.sha256(source/name) == expected,"Archived source changed")
    try:
        audit_sources()
        frozen.require(frozen.sha256(directory/"dataset.json") == reg["dataset_sha256"],"Dataset changed")
        result.update(_run_trial(directory,reg,result,phase,audit_sources))
    except BaseException as error:
        result["reason"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    finally:
        result["wall_seconds"] = time.monotonic()-started
        write_json(directory/"worker_result.json",result)
    return result


def _run_trial(directory, reg, result, phase, audit_sources):
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    import numpy as np
    import torch
    from torch.nn import functional as F
    from ..control.x2_behavior import Observation, BEHAVIORS, PHASES
    from .mps_memory import MPSMemorySampler
    from .resource_guard import write_json
    from .x2_behavior_policy import array_identity
    from .x2_behavior_session import checkpoint_registration_metadata, load_candidate_policy
    from .x2_behavior_training import inspect_checkpoint, save_checkpoint, restore_checkpoint, hashlib_counts, _prediction, _gradient
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    source, root = Path(reg["source_run"]), Path(reg["root"])
    dataset = frozen.read_json(directory/"dataset.json")
    frozen.bound(dataset["identity"],"dataset_id")
    frozen.require(dataset["identity"]["dataset_id"] == reg["dataset_id"],"Dataset identity differs")
    episodes = dataset["splits"]["train"]
    validation = dataset["splits"]["validation"][0]
    branches = [b for e in episodes for b in training_branches(e)]
    cached_contexts = frozen.read_json(source/"paired-observations.json")["contexts"]
    with np.load(source/"train-features.npz",allow_pickle=False) as archive:
        cached_train = {k:archive[k].copy() for k in archive.files}
    with np.load(source/"paired-train-features.npz",allow_pickle=False) as archive:
        cached_pairs = {k:archive[k].copy() for k in archive.files}
    sampler = MPSMemorySampler(directory/"mps_memory.csv",reg["resources"]["effective_rss_limit_bytes"])
    step_counts = {"entered":0,"completed":0,"backward_recompute_entered":0,"backward_recompute_completed":0}
    in_backward = False
    policy = None
    def save_arrays(name, **arrays):
        with (directory/(name+".npz")).open("xb") as stream:
            np.savez_compressed(stream,**arrays)
    def save_rows(name, rows, features, inputs, frames):
        save_arrays(name,features_mv=np.asarray(features),encoded_inputs=np.asarray(inputs),
            behavior_logits=np.asarray([r["behavior_logits"] for r in rows],np.float32),
            phase_logits=np.asarray([r["phase_logits"] for r in rows],np.float32),
            parameter_raw=np.asarray([r["parameter_raw"] for r in rows],np.float32),
            intensity=np.asarray([r["intensity"] for r in rows],np.float32))
        write_json(directory/(name+".json"),{"rows":[{"observation":f["observation"],"target":f["target"],"raw":r}
            for f,r in zip(frames,rows)],"arrays_sha256":frozen.sha256(directory/(name+".npz"))})
    try:
        phase("strict_source_restore")
        metadata = inspect_checkpoint(reg["checkpoint"])
        frozen.require(metadata["checkpoint_id"] == reg["checkpoint_id"]
            and checkpoint_registration_metadata(metadata) == reg["checkpoint_metadata"],"Source checkpoint changed")
        policy = load_candidate_policy(root,reg["checkpoint_metadata"],reg["checkpoint"])
        bind_theta_trial_policy(policy,metadata["checkpoint_id"])
        policy.eval()
        write_json(directory/"policy.json",policy.identity)
        counts = hashlib_counts(policy)
        frozen.require(policy.brain.graph.n == 166700 and policy.brain.graph.counts.nnz == 25582938,"Complete graph required")
        parameters_before = {n:p.detach().cpu().numpy().copy() for n,p in policy.named_parameters()}
        heads_before = {n:array_identity(v) for n,v in parameters_before.items() if n != "brain.theta"}
        save_checkpoint(directory/"checkpoint-before",policy,{"registration_id":reg["registration_id"],"optimizer_steps":0})
        original_step = policy._step
        def observed_step(*args,**kwargs):
            step_counts["entered"] += 1
            if in_backward:
                step_counts["backward_recompute_entered"] += 1
            out = original_step(*args,**kwargs)
            step_counts["completed"] += 1
            if in_backward:
                step_counts["backward_recompute_completed"] += 1
            return out
        policy._step = observed_step
        phase("before_blank_precondition")
        template = policy.initial_state("theta-template")
        initial = policy.state_identity(template)
        frozen.require(initial["arrays"] == frozen.read_json(source/"initial-state.json")["arrays"],"Source blank initial state differs")
        save_arrays("initial-before",**policy.state_arrays(template))
        write_json(directory/"initial-before.json",initial)

        def consume(observation,state,*,grad=False):
            before_step = state.brain.step
            with torch.set_grad_enabled(grad):
                out = policy(observation,state,checkpoint_blocks=grad)
            torch.mps.synchronize()
            frozen.require(out.state.brain.step == before_step+500,"Decision must consume 500 steps")
            result["completed_frames"] += 1
            return out

        def validation_run(name,initial_template):
            state = replace(initial_template,session_id=validation["episode_id"])
            start = policy.state_identity(state)
            rows, features, inputs = [],[],[]
            for frame in validation["frames"]:
                phase(name,frame_index=frame["frame_index"])
                obs = Observation.from_dict(frame["observation"])
                out = consume(obs,state)
                rows.append(_prediction((out.behavior_logits,out.phase_logits,out.parameter_raw,out.intensity)))
                features.append(out.features_mv.cpu().numpy().copy())
                inputs.append(policy.encoder.encode(obs))
                state = out.state
            save_rows(name,rows,features,inputs,validation["frames"])
            write_json(directory/(name+"-states.json"),{"initial":start,"final":policy.state_identity(state),
                "resets":1,"gate_only_polls_never_consumed":True})
            return rows

        val_before = validation_run("validation-before",template)
        states, replay_evidence, prefix_rows, prefix_features, prefix_inputs = [],[],[],[],[]
        prefix_frames = []
        for episode_index, episode in enumerate(episodes):
            state = replace(template,session_id=episode["episode_id"])
            for frame in episode["frames"][:4]:
                phase("replay_train_prefix",family=episode["family_id"],frame_index=frame["frame_index"])
                obs = Observation.from_dict(frame["observation"])
                out = consume(obs,state)
                feature, encoded = out.features_mv.cpu().numpy().copy(),policy.encoder.encode(obs)
                cache_index = episode_index*12+frame["frame_index"]
                frozen.require(np.array_equal(feature,cached_train["features_mv"][cache_index])
                    and np.array_equal(encoded,cached_train["encoded_inputs"][cache_index]),"Prefix response/input differs from sealed cache")
                state = out.state
                prefix_rows.append(_prediction((out.behavior_logits,out.phase_logits,out.parameter_raw,out.intensity)))
                prefix_features.append(feature)
                prefix_inputs.append(encoded)
                prefix_frames.append(frame)
                replay_evidence.append({"family":episode["family_id"],"cache_index":cache_index,
                    "features_exact":True,"input_exact":True,"post_state":policy.state_identity(state)})
            context = next(c for c in cached_contexts if c["family"] == episode["family_id"] and c["frame_index"] == 4)
            current = policy.state_identity(state)
            old = context["branches"][0]["prestate_before"]
            frozen.bound(old,"state_id")
            frozen.require(all(current[k] == old[k] for k in ("arrays","frame_index","neural_step","observation_seq","session_id","schema")),
                "Actual train pre-state differs from sealed context")
            for offset,b in enumerate(branches[episode_index*3:episode_index*3+3]):
                frozen.require(b["frames"][0]["observation"] == context["branches"][offset]["observation"]
                    and b["frames"][0]["target"] == context["branches"][offset]["target"],"Branch differs from authenticated train pair")
                b["prestate_id"] = current["state_id"]
                b["source_prestate_id"] = old["state_id"]
            save_arrays("train-prestate-"+str(episode_index),**policy.state_arrays(state))
            write_json(directory/("train-prestate-"+str(episode_index)+".json"),{"current":current,"source":old,
                "allowed_identity_rebind":"policy identity only; all arrays, session and clocks exact"})
            states.append(state.detach())
        save_rows("prefix-replay",prefix_rows,prefix_features,prefix_inputs,prefix_frames)
        write_json(directory/"train-protocol.json",{"branches":branches,"replay":replay_evidence,
            "train_loss_reduction":"mean of six branch means; forward has two connected frames",
            "prefix_credit":"prefix states detached; no claim of credit before frame4"})
        optimizer = torch.optim.Adam([policy.brain.theta],lr=1e-4)
        optimizer.zero_grad(set_to_none=True)
        gradients, gradient_reports, boundary_gradients = [],[],{}
        train_before, before_features, before_inputs = [],[],[]
        train_frames = [f for b in branches for f in b["frames"]]
        for branch_index, branch in enumerate(branches):
            state = states[branch_index//3]
            immutable = policy.state_identity(state)
            previous = policy.brain.theta.grad.detach().clone() if policy.brain.theta.grad is not None else torch.zeros_like(policy.brain.theta)
            losses, boundary = [],None
            for frame_index, frame in enumerate(branch["frames"]):
                phase("theta_gradient_forward",branch=branch_index,frame=frame_index)
                obs = Observation.from_dict(frame["observation"])
                out = consume(obs,state,grad=True)
                feature, encoded = out.features_mv.detach().cpu().numpy().copy(),policy.encoder.encode(obs)
                if frame_index == 0:
                    cache_index = (branch_index//3)*9+3+branch_index%3
                    expected = cached_pairs
                else:
                    cache_index = (branch_index//3)*12+5
                    expected = cached_train
                frozen.require(np.array_equal(feature,expected["features_mv"][cache_index])
                    and np.array_equal(encoded,expected["encoded_inputs"][cache_index]),"Surrogate changed sealed train forward")
                train_before.append(_prediction((out.behavior_logits,out.phase_logits,out.parameter_raw,out.intensity)))
                before_features.append(feature)
                before_inputs.append(encoded)
                bi = tuple(map(str,BEHAVIORS)).index(frame["target"]["behavior"])
                pi = tuple(map(str,PHASES)).index(frame["target"]["phase"])
                losses.append(F.cross_entropy(out.behavior_logits[None],torch.tensor([bi],device=out.behavior_logits.device))
                    +F.cross_entropy(out.phase_logits[None],torch.tensor([pi],device=out.phase_logits.device)))
                state = out.state  # Keep both forward decisions in the same autograd graph.
                if frame_index == 0 and len(branch["frames"]) == 2:
                    boundary = state.brain.v
                    boundary.retain_grad()
            loss = torch.stack(losses).mean()/6
            phase("theta_backward",branch=branch_index)
            in_backward = True
            loss.backward()
            in_backward = False
            torch.mps.synchronize()
            contribution = policy.brain.theta.grad-previous
            gradients.append(contribution.detach().cpu().numpy().copy())
            gradient_reports.append({"branch":branch_index,"theta":_gradient(contribution),
                "combined_loss_boundary_v":_gradient(boundary.grad if boundary is not None else None),
                "isolated_final_frame_credit_tested":False})
            if boundary is not None and boundary.grad is not None:
                boundary_gradients["branch_"+str(branch_index)] = boundary.grad.detach().cpu().numpy().copy()
            frozen.require(policy.state_identity(states[branch_index//3]) == immutable,"Gradient branch mutated fixed pre-state")
            del out,state,loss,losses,boundary,contribution,previous
        save_rows("train-before",train_before,before_features,before_inputs,train_frames)
        raw_gradient = policy.brain.theta.grad.detach().cpu().numpy().copy()
        grad_stats = _gradient(policy.brain.theta.grad)
        save_arrays("theta-raw-gradient",theta=raw_gradient,branches=np.asarray(gradients),**boundary_gradients)
        group_names = list(policy.brain.graph.group_names)
        selected = {name:{"index":group_names.index(name),"gradient":float(raw_gradient[group_names.index(name)]),
            "theta_before":float(parameters_before["brain.theta"][group_names.index(name)])}
            for name in ("R1-R6","L1","L2") if name in group_names}
        write_json(directory/"gradient-audit.json",{"total":grad_stats,"branches":gradient_reports,
            "selected_types":selected,"group_names":group_names,"isolated_final_frame_credit_tested":False})
        frozen.require(grad_stats["finite"] and grad_stats["l2"] is not None and grad_stats["l2"]>0,
            "Missing, nonfinite or zero theta gradient; update refused")
        factor = min(1.,1./grad_stats["l2"])
        policy.brain.theta.grad.mul_(factor)
        save_arrays("theta-clipped-gradient",theta=policy.brain.theta.grad.detach().cpu().numpy().copy())
        frozen.require(all(p.grad is None for n,p in policy.named_parameters() if n != "brain.theta"),"Frozen head received gradients")
        phase("single_theta_adam_update")
        optimizer.step()
        result["optimizer_steps"] = 1
        policy.validate_parameters()
        after_theta = policy.brain.theta.detach().cpu().numpy().copy()
        delta = after_theta.astype(np.float64)-parameters_before["brain.theta"].astype(np.float64)
        frozen.require(np.isfinite(after_theta).all() and np.any(delta != 0),"Theta update did not produce a finite change")
        head_hashes = {n:array_identity(p.detach().cpu().numpy()) for n,p in policy.named_parameters() if n != "brain.theta"}
        frozen.require(head_hashes == heads_before and hashlib_counts(policy) == counts,"Head or graph mutated")
        update_audit = {"gradient_l2_clip_factor":factor,"changed_theta":int(np.count_nonzero(delta)),
            "theta_delta_l2":float(np.linalg.norm(delta)),"theta_delta_max_abs":float(np.abs(delta).max()),
            "heads_before":heads_before,"heads_after":head_hashes,"heads_exact":True,"graph":counts,"graph_exact":True,
            "selected_types":{n:{**v,"theta_after":float(after_theta[v["index"]])} for n,v in selected.items()}}
        write_json(directory/"update-audit.json",update_audit)
        saved = save_checkpoint(directory/"checkpoint",policy,{"registration_id":reg["registration_id"],
            "source_checkpoint_id":reg["checkpoint_id"],"dataset_id":reg["dataset_id"],"optimizer_steps":1,
            "optimizer":{"name":"Adam","lr":1e-4,"gradient_l2_cap":1.},"heads_frozen":True,"qualification_only":True},optimizer=optimizer)
        train_after, after_features, after_inputs = [],[],[]
        for branch_index,branch in enumerate(branches):
            state = states[branch_index//3]
            immutable = policy.state_identity(state)
            for frame_index,frame in enumerate(branch["frames"]):
                phase("train_after_conditional_probe",branch=branch_index,frame=frame_index)
                obs = Observation.from_dict(frame["observation"])
                out = consume(obs,state)
                train_after.append(_prediction((out.behavior_logits,out.phase_logits,out.parameter_raw,out.intensity)))
                after_features.append(out.features_mv.cpu().numpy().copy())
                after_inputs.append(policy.encoder.encode(obs))
                state = out.state
            frozen.require(policy.state_identity(states[branch_index//3]) == immutable,"After probe mutated source pre-state")
        save_rows("train-after",train_after,after_features,after_inputs,train_frames)
        phase("after_independent_blank_precondition")
        after_template = policy.initial_state("theta-template-after")
        save_arrays("initial-after",**policy.state_arrays(after_template))
        write_json(directory/"initial-after.json",policy.state_identity(after_template))
        val_after = validation_run("validation-after",after_template)
        before_score,after_score = validation_score(validation,val_before),validation_score(validation,val_after)
        before_ce,after_ce = train_ce(train_before,branches),train_ce(train_after,branches)
        comparison = {"train":{"before_ce":before_ce,"after_ce":after_ce,
            "before_correct_frames":sum(r["behavior"]==f["target"]["behavior"] and r["phase"]==f["target"]["phase"] for r,f in zip(train_before,train_frames)),
            "after_correct_frames":sum(r["behavior"]==f["target"]["behavior"] and r["phase"]==f["target"]["phase"] for r,f in zip(train_after,train_frames)),
            "frames":8,"branches":6,"state_semantics":"same original train prestates for before/after conditional probe"},
            "validation":{"before":before_score,"after":after_score},
            "screen":learning_screen(before_ce,after_ce,before_score,after_score)}
        for split,old,new,frames in (("train",train_before,train_after,train_frames),("validation",val_before,val_after,validation["frames"])):
            old_int,new_int = intensity_report(old,[f["target"] for f in frames]),intensity_report(new,[f["target"] for f in frames])
            comparison[split]["intensity"] = {"before":old_int,"after":new_int,
                "active_mse_regressed":new_int["active_target_mse"]>old_int["active_target_mse"]}
        write_json(directory/"comparison.json",comparison)
        phase("same_instance_scientific_restore")
        restore_checkpoint(directory/"checkpoint",policy)
        restored_hashes = {n:array_identity(p.detach().cpu().numpy()) for n,p in policy.named_parameters()}
        frozen.require(all(restored_hashes[n] == saved["arrays"]["parameter."+n] for n in restored_hashes),"Scientific parameter restoration differs")
        # Reuse saved DN; no second brain or neural replay is needed for restoration.
        restore_rows = train_before+train_after+val_before+val_after
        all_features = before_features+after_features
        for name in ("validation-before","validation-after"):
            with np.load(directory/(name+".npz"),allow_pickle=False) as archive:
                all_features.extend(archive["features_mv"])
        with torch.no_grad():
            recovered = [_prediction(policy.read_features(torch.from_numpy(np.asarray(v)).to("mps"))) for v in all_features]
        frozen.require(recovered == restore_rows,"Frozen readouts changed after scientific restoration")
        write_json(directory/"restore-audit.json",{"parameters_exact":True,"saved_dn_readouts_exact":True,
            "dn_rows":len(all_features),"desktop_loader_extended":False,"same_graph_instance":True})
        audit_sources()
        frozen.require(result["completed_frames"] == 48 and result["optimizer_steps"] == 1,"Incomplete registered trial")
        result.update(status="completed",reason="single_theta_update_and_complete_continuous_comparison",
            checkpoint_id=saved["checkpoint_id"],policy_id=policy.policy_id,screen=comparison["screen"],
            original_train_prefix_exact=True,gradient_forward_cache_exact=True,heads_unchanged=True,graph_unchanged=True,
            coverage_limits=configuration()["coverage_limits"])
    finally:
        if policy is not None and "original_step" in locals():
            policy._step = original_step
        result["step_calls"] = {**step_counts,"interpretation":"Python step entry/completion counts; checkpoint early-stop may interrupt a recomputation call"}
        result["mps_memory"] = sampler.close()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--source-run",type=Path,default=frozen.ROOT/"data/x2_behavior_short_trial_interaction_paired_20261024_r1")
    parser.add_argument("--_worker",type=Path,help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker:
        return 0 if run_worker(args._worker)["status"] == "completed" else 1
    if not args.execute:
        print(json.dumps({"mode":"preview","brain_loaded":False,"configuration":configuration()},indent=2))
        return 0
    if args.output is None:
        parser.error("--execute requires a fresh --output")
    try:
        result = execute(args.output,args.source_run)
    except Exception as error:
        print(f"{type(error).__name__}: {error}",file=sys.stderr)
        return 1
    print(json.dumps({"status":result["status"],"output":result["output"],"reason":result["worker"].get("reason")},indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
