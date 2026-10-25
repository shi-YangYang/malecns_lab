"""One six-frame local-derivative diagnostic; no fitting or validation access."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

from . import x2_inference as frozen
from .x2_behavior_trial import MINIMUM_BUDGET, output_directory as guarded_output
from .x2_theta_trial import OWN_SOURCES, row_ce
from .x2_task_rate_causal import SOURCE_FILES as STATE_SOURCES, restore_state, validate_state_arrays

SCHEMA = "malecns_lab.x2_theta_local_derivative.v1"
PREFIX = "x2_behavior_short_trial_theta_derivative_"
EPSILON = 1./16
INPUT_FILES = ("registration.json", "summary.json", "train-protocol.json", "train-prestate-0.json",
    "train-prestate-0.npz", "train-before.json", "train-before.npz", "train-after.json",
    "theta-raw-gradient.npz", "checkpoint-before/checkpoint.json", "checkpoint-before/arrays.npz",
    "checkpoint/checkpoint.json", "checkpoint/arrays.npz")


def configuration():
    return {"schema": SCHEMA, "wall_seconds": 180, "total_frames": 6, "decision_steps": 500,
        "branch": "continuous-train-a / SHOW_FORWARD / two connected frames", "new_preconditioning_steps": 0,
        "conditions": [0., EPSILON, -EPSILON], "direction": "sealed theta_after minus theta_before",
        "loss": "mean of two behavior+phase CE values; archived branch gradient multiplied by 6",
        "backward": "baseline only: zero spike-value gradient, retain direct gain paths; binary forward unchanged",
        "spike_trace": "all 166700 nodes, 1000 steps per condition, packed bit order little",
        "graph_nodes": 166700, "graph_edges": 25582938, "optimizer_steps": 0,
        "resource": {"system_reserve_gib": 1, "minimum_safe_budget_gib": 2, "maximum_budget_gib": 6},
        "interpretation": "local branch derivative and fixed secant; trajectory changes or numerical cancellation may be inconclusive",
        "stop": "one fixed epsilon only; no smaller-step retry, fitting, validation selection or deployment",
        "validation_opened": False, "automatic_deployment": False}


def output_directory(output, root, **kwargs):
    frozen.require(Path(output).name.startswith(PREFIX) and len(Path(output).name)>len(PREFIX), "Dedicated derivative prefix required")
    return guarded_output(output, root, **kwargs)


def source_inputs(source):
    """Read only the sealed training branch, checkpoints and complete pre-state."""
    import numpy as np
    from .x2_behavior_training import inspect_checkpoint
    from .x2_theta_trial_policy import SCHEMA as THETA_SCHEMA
    reg, summary = frozen.read_json(source/"registration.json"), frozen.read_json(source/"summary.json")
    frozen.bound(reg,"registration_id")
    before, after = [inspect_checkpoint(source/name) for name in ("checkpoint-before","checkpoint")]
    frozen.require(summary["status"] == summary["worker"]["status"] == "completed"
        and summary["worker"]["completed_frames"] == 48 and summary["worker"]["optimizer_steps"] == 1
        and summary["registration_id"] == reg["registration_id"]
        and all(m["context"]["registration_id"] == reg["registration_id"] for m in (before,after))
        and before["policy"] == after["policy"] and before["policy"]["schema"] == THETA_SCHEMA
        and summary["worker"]["checkpoint_id"] == after["checkpoint_id"], "Sealed theta trial differs")
    branch = frozen.read_json(source/"train-protocol.json")["branches"][0]
    identity = frozen.read_json(source/"train-prestate-0.json")["current"]
    frozen.require(branch["family"] == "continuous-train-a" and branch["task"] == "SHOW_FORWARD"
        and len(branch["frames"]) == 2 and branch["prestate_id"] == identity["state_id"]
        and identity["policy_id"] == before["policy"]["policy_id"] and identity["neural_step"] == 2600,
        "Expected first real train-a FORWARD branch and state")
    with np.load(source/"train-prestate-0.npz",allow_pickle=False) as z:
        arrays = validate_state_arrays({k:z[k].copy() for k in z.files},identity)
    saved = frozen.read_json(source/"train-before.json")
    frozen.require(saved["arrays_sha256"] == frozen.sha256(source/"train-before.npz")
        and all(f["observation"] == r["observation"] and f["target"] == r["target"]
                for f,r in zip(branch["frames"],saved["rows"][:2])), "Training response binding differs")
    with np.load(source/"train-before.npz",allow_pickle=False) as z:
        cached = {k:z[k][:2].copy() for k in z.files}
    with np.load(source/"theta-raw-gradient.npz",allow_pickle=False) as z:
        sg = z["branches"][0].astype(np.float64)*6
    values = []
    for name in ("checkpoint-before","checkpoint"):
        with np.load(source/name/"arrays.npz",allow_pickle=False) as z:
            values.append({k:z[k].copy() for k in z.files if k.startswith("parameter.")})
    frozen.require(set(values[0]) == set(values[1]) and all(np.array_equal(values[0][k],values[1][k])
        for k in values[0] if k != "parameter.brain.theta"), "Source heads changed")
    theta0, theta1 = [v["parameter.brain.theta"] for v in values]
    frozen.require(theta0.shape == theta1.shape == sg.shape == (11752,)
        and all(np.isfinite(x).all() for x in (theta0,theta1,sg)) and np.any(theta0!=theta1), "Invalid sealed update direction")
    return {"registration":reg,"before":before,"after":after,"branch":branch,"identity":identity,"arrays":arrays,
        "cached":cached,"rows":saved["rows"][:2],"theta0":theta0,"direction":theta1.astype(np.float64)-theta0.astype(np.float64),
        "sg":sg,"full_step_rows":frozen.read_json(source/"train-after.json")["rows"][:2]}


def block_spike_gradient(spikes):
    """A hook changes only the derivative; direct gain*spike gradients survive."""
    import torch
    if spikes.requires_grad:
        spikes.register_hook(lambda gradient: torch.zeros_like(gradient))
    return spikes


class StepObserver:
    """Observe actual binary spikes; ignore checkpoint backward recomputation."""
    def __init__(self, original, start_step, *, block, nodes=166700):
        self.original, self.start_step, self.block, self.nodes = original,start_step,block,nodes
        self.backward, self.pending, self.chunks, self.steps = False,[],[],0
        self.recompute_calls = 0

    def __call__(self, *args, **kwargs):
        import torch
        import numpy as np
        if self.backward:
            self.recompute_calls += 1
        output = self.original(*args,**kwargs)
        spikes = output[-1]
        if self.block:
            block_spike_gradient(spikes)
        if not self.backward:
            frozen.require(output[0].step == self.start_step+self.steps+1 and spikes.shape == (self.nodes,), "Unexpected spike trace clock/shape")
            self.pending.append(spikes.detach().to(torch.bool))
            self.steps += 1
            if len(self.pending) == 50:
                self.chunks.append(np.packbits(torch.stack(self.pending).cpu().numpy(),axis=1,bitorder="little"))
                self.pending.clear()
        return output

    def trace(self):
        import numpy as np
        frozen.require(self.steps == 1000 and not self.pending,"Incomplete 1000-step spike trace")
        return np.concatenate(self.chunks)


def spike_difference(reference, other, *, start_step, nodes=166700):
    import numpy as np
    frozen.require(reference.shape == other.shape == (1000,(nodes+7)//8),"Invalid packed full-brain traces")
    changed = np.bitwise_xor(reference,other)
    lookup = np.unpackbits(np.arange(256,dtype=np.uint8)[:,None],axis=1).sum(1)
    counts = lookup[changed].sum(1)
    indices = np.flatnonzero(counts)
    return {"spike_bits_equal":not bool(len(indices)),"changed_spike_bits":int(counts.sum()),
        "steps_with_differences":int(len(indices)),"first_different_neural_step":start_step+int(indices[0]) if len(indices) else None,
        "different_bits_per_step":counts.tolist()}


def execute(output, source, *, root=frozen.ROOT):
    import psutil
    from .resource_guard import resource_budget, supervise, write_json
    from .x2_behavior_training import source_inventory
    root,source = Path(root).resolve(),Path(source).absolute()
    frozen.require(source.resolve() == source and source.is_dir() and source.is_relative_to(root/"data"),"Sealed project run required")
    output = output_directory(output,root,inputs=(source,))
    started = time.monotonic()
    with frozen.numerical_lock(root):
        preflight = frozen.simulator_preflight(root)
        resources = resource_budget(psutil.virtual_memory().available,reserve_bytes=frozen.GIB)
        frozen.require(resources["effective_rss_limit_bytes"] >= MINIMUM_BUDGET,"Insufficient safe complete-brain budget")
        assets = source_inputs(source)
        paired = Path(assets["registration"]["checkpoint"])
        frozen.require(paired.is_relative_to(root/"data") and paired.resolve() == paired,"Local strict source checkpoint required")
        output.mkdir(exist_ok=False)
        sources = source_inventory(root)
        for name in {*OWN_SOURCES,*STATE_SOURCES,"learning/x2_theta_local_derivative.py"}:
            sources[name] = frozen.sha256(root/"src/malecns_lab"/name)
        for name in sources:
            target = output/"sources/malecns_lab"/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(root/"src/malecns_lab"/name,target)
            target.chmod(0o444)
        reg = {"configuration":configuration(),"root":str(root),"source_run":str(source),"output":str(output),
            "parent_pid":os.getpid(),"deadline_monotonic":started+180,"resources":resources,"source_sha256":sources,
            "input_sha256":{n:frozen.sha256(source/n) for n in INPUT_FILES},
            "source_registration_id":assets["registration"]["registration_id"],
            "paired_checkpoint":str(paired),"paired_sha256":{n:frozen.sha256(paired/n) for n in ("checkpoint.json","arrays.npz")}}
        reg["registration_id"] = frozen.canonical_hash(reg)
        write_json(output/"registration.json",reg)
        write_json(output/"simulator_preflight.json",preflight)
        del assets
        remaining = reg["deadline_monotonic"]-time.monotonic()
        frozen.require(remaining>0,"Preparation exhausted hard budget")
        supervisor = supervise([sys.executable,"-m","malecns_lab.learning.x2_theta_local_derivative","--_worker",str(output)],
            output,rss_limit_bytes=resources["effective_rss_limit_bytes"],wall_seconds=remaining,reserve_bytes=frozen.GIB)
        worker = frozen.read_json(output/"worker_result.json") if (output/"worker_result.json").exists() else {}
        result = {"status":"completed" if supervisor["status"] == worker.get("status") == "completed" else "failed",
            "registration_id":reg["registration_id"],"output":str(output),"supervisor":supervisor,"worker":worker,
            "wall_seconds":time.monotonic()-started}
        write_json(output/"summary.json",result)
        return result


def run_worker(directory):
    from .resource_guard import write_json
    directory = Path(directory).absolute()
    frozen.require(directory.resolve() == directory,"No worker symlink")
    reg = frozen.read_json(directory/"registration.json")
    frozen.bound(reg,"registration_id")
    root,source = Path(reg["root"]),Path(reg["source_run"])
    output_directory(directory,root,created=True,inputs=(source,reg["paired_checkpoint"]))
    frozen.require(reg["parent_pid"] == os.getppid() and reg["output"] == str(directory)
        and reg["configuration"] == configuration(),"Registered supervising parent/configuration required")
    with (root/"data/.continuous-training.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(lock.fileno(),fcntl.LOCK_UN)
            raise ValueError("Parent must own shared full-brain lock")
    with (directory/"worker.claim").open("x") as stream:
        stream.write(str(os.getpid()))
    started = time.monotonic()
    result = {"status":"failed","reason":None,"registration_id":reg["registration_id"],"completed_frames":0,
        "required_frames":6,"optimizer_steps":0,"validation_opened":False,"automatic_deployment":False}
    def audit():
        for name,value in reg["source_sha256"].items():
            frozen.require(frozen.sha256(root/"src/malecns_lab"/name) == value
                and frozen.sha256(directory/"sources/malecns_lab"/name) == value,"Registered source changed")
        for path,mapping in ((source,reg["input_sha256"]),(Path(reg["paired_checkpoint"]),reg["paired_sha256"])):
            for name,value in mapping.items():
                frozen.require(frozen.sha256(path/name) == value,"Sealed diagnostic input changed")
    def phase(name,**details):
        frozen.require(time.monotonic()<reg["deadline_monotonic"],"Hard 180-second deadline exhausted")
        write_json(directory/"phase.json",{"phase":name,"wall_seconds":time.monotonic()-started,
            "completed_frames":result["completed_frames"],"required_frames":6,**details})
    try:
        audit()
        probe(directory,reg,result,phase)
        audit()
        frozen.require(result["completed_frames"] == 6,"Incomplete diagnostic; no shortened success")
        result.update(status="completed",reason="fixed_train_branch_local_derivative_and_six_frames_recorded")
    except BaseException as error:
        result["reason"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    finally:
        result["wall_seconds"] = time.monotonic()-started
        write_json(directory/"worker_result.json",result)
    return result


def probe(directory, reg, result, phase):
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    import numpy as np
    import torch
    from torch.nn import functional as F
    from ..control.x2_behavior import Observation,BEHAVIORS,PHASES
    from .mps_memory import MPSMemorySampler
    from .resource_guard import write_json
    from .x2_behavior_policy import array_identity
    from .x2_behavior_session import load_candidate_policy
    from .x2_behavior_training import restore_checkpoint,hashlib_counts,_prediction,_gradient
    from .x2_theta_trial_policy import bind_theta_trial_policy
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    source,root = Path(reg["source_run"]),Path(reg["root"])
    assets = source_inputs(source)
    frozen.require(assets["registration"]["registration_id"] == reg["source_registration_id"],"Source identity changed")
    sampler = MPSMemorySampler(directory/"mps_memory.csv",reg["resources"]["effective_rss_limit_bytes"])
    policy,original_step = None,None
    def save(name,**values):
        with (directory/(name+".npz")).open("xb") as stream:
            np.savez_compressed(stream,**values)
    try:
        phase("strict_source_and_sealed_prestate_restore")
        sr = assets["registration"]
        policy = load_candidate_policy(root,sr["checkpoint_metadata"],reg["paired_checkpoint"])
        bind_theta_trial_policy(policy,sr["checkpoint_id"])
        restore_checkpoint(source/"checkpoint-before",policy)
        policy.eval()
        original_step = policy._step
        shared = restore_state(policy,assets["identity"],assets["arrays"])
        frozen.require(policy.brain.graph.n == 166700 and policy.brain.graph.counts.nnz == 25582938,"Full graph required")
        graph_hash = hashlib_counts(policy)
        head_hashes = {n:array_identity(p.detach().cpu().numpy()) for n,p in policy.named_parameters() if n != "brain.theta"}
        records,features,inputs,traces,theta_values = [],[],[],[],[]
        local_gradient = None
        for condition in configuration()["conditions"]:
            theta = (assets["theta0"].astype(np.float64)+condition*assets["direction"]).astype(np.float32)
            with torch.no_grad():
                policy.brain.theta.copy_(torch.from_numpy(theta).to(policy.brain.theta.device))
            frozen.require(np.array_equal(policy.brain.theta.detach().cpu().numpy(),theta),"Device theta differs from registered condition")
            policy.zero_grad(set_to_none=True)
            frozen.require(policy.state_identity(shared) == assets["identity"],"Previous condition mutated sealed state")
            observer = StepObserver(original_step,shared.brain.step,block=condition == 0)
            policy._step = observer
            state,rows,losses,condition_features,condition_inputs = shared,[],[],[],[]
            for index,frame in enumerate(assets["branch"]["frames"]):
                phase("condition_forward",multiplier=condition,frame=index)
                observation = Observation.from_dict(frame["observation"])
                with torch.set_grad_enabled(condition == 0):
                    out = policy(observation,state,checkpoint_blocks=condition == 0)
                    if condition == 0:
                        target = frame["target"]
                        bi,pi = tuple(map(str,BEHAVIORS)).index(target["behavior"]),tuple(map(str,PHASES)).index(target["phase"])
                        losses.append(F.cross_entropy(out.behavior_logits[None],torch.tensor([bi],device=out.behavior_logits.device))
                            +F.cross_entropy(out.phase_logits[None],torch.tensor([pi],device=out.phase_logits.device)))
                torch.mps.synchronize()
                feature,encoded = out.features_mv.detach().cpu().numpy().copy(),policy.encoder.encode(observation)
                raw = _prediction((out.behavior_logits,out.phase_logits,out.parameter_raw,out.intensity))
                frozen.require(np.array_equal(encoded,assets["cached"]["encoded_inputs"][index]),"Input changed")
                if condition == 0:
                    frozen.require(np.array_equal(feature,assets["cached"]["features_mv"][index])
                        and raw == assets["rows"][index]["raw"],"Gradient hook changed sealed binary forward")
                rows.append(raw)
                condition_features.append(feature)
                condition_inputs.append(encoded)
                state = out.state  # No detach across the two FORWARD decisions.
                result["completed_frames"] += 1
            trace = observer.trace()
            if condition == 0:
                phase("baseline_local_branch_backward")
                observer.backward = True
                torch.stack(losses).mean().backward()
                torch.mps.synchronize()
                frozen.require(policy.brain.theta.grad is not None,"Local theta derivative missing")
                local_gradient = policy.brain.theta.grad.detach().cpu().numpy().copy()
                stats = _gradient(policy.brain.theta.grad)
                frozen.require(stats["finite"],"Nonfinite local derivative")
                write_json(directory/"local-gradient.json",{"gradient":stats,"checkpoint_recompute_calls":observer.recompute_calls,
                    "meaning":"computed derivative within the baseline discrete spike/reset/refractory branch; no spike-value derivative"})
            records.append({"multiplier":condition,"theta":array_identity(theta),"prestate":assets["identity"],
                "final_state":policy.state_identity(state),"rows":rows,
                "ce":sum(row_ce(r,f["target"]) for r,f in zip(rows,assets["branch"]["frames"]))/2})
            features.append(condition_features)
            inputs.append(condition_inputs)
            traces.append(trace)
            theta_values.append(theta)
            frozen.require(policy.state_identity(shared) == assets["identity"],"Condition mutated source pre-state")
            policy._step = original_step
            del out,state,losses,observer
        direction = assets["direction"]
        local_dot = float(local_gradient.astype(np.float64)@direction)
        sg_dot = float(assets["sg"]@direction)
        actual_plus = theta_values[1].astype(np.float64)-assets["theta0"].astype(np.float64)
        actual_minus = theta_values[2].astype(np.float64)-assets["theta0"].astype(np.float64)
        save("responses",features_mv=np.asarray(features),encoded_inputs=np.asarray(inputs),
            behavior_logits=np.asarray([[r["behavior_logits"] for r in c["rows"]] for c in records],np.float32),
            phase_logits=np.asarray([[r["phase_logits"] for r in c["rows"]] for c in records],np.float32),
            packed_spikes=np.asarray(traces),theta_conditions=np.asarray(theta_values),
            direction=direction,local_gradient=local_gradient,archived_surrogate_gradient=assets["sg"])
        comparison = {"configuration":configuration(),"branch":assets["branch"],"conditions":records,
            "diagnostic_identity":{"source_policy_id":policy.policy_id,"source_checkpoint_before":assets["before"]["checkpoint_id"],
                "source_checkpoint_after":assets["after"]["checkpoint_id"],"backward_override":"zero spike-value gradients at baseline only",
                "implementation_sha256":frozen.sha256(Path(__file__)),"production_policy_changed":False},
            "local_directional_derivative":local_dot,"archived_surrogate_directional_derivative":sg_dot,
            "central_difference":(records[1]["ce"]-records[2]["ce"])/(2*EPSILON),
            "positive_step_ce_delta":records[1]["ce"]-records[0]["ce"],
            "archived_full_step_ce":sum(row_ce(r["raw"],r["target"]) for r in assets["full_step_rows"])/2,
            "effective_plus_projection":float(actual_plus@direction/(direction@direction)),
            "effective_minus_projection":float(actual_minus@direction/(direction@direction)),
            "plus_rounding_residual_l2":float(np.linalg.norm(actual_plus-EPSILON*direction)),
            "minus_rounding_residual_l2":float(np.linalg.norm(actual_minus+EPSILON*direction)),
            "spike_differences":{label:spike_difference(traces[0],traces[i],start_step=shared.brain.step)
                for i,label in ((1,"plus"),(2,"minus"))},
            "arrays_sha256":frozen.sha256(directory/"responses.npz"),
            "interpretation_limit":"effects can coexist; finite differences crossing event branches are not exact derivatives; tiny CE changes may be numerically inconclusive",
            "further_training_authorized":False,"automatic_deployment":False}
        write_json(directory/"comparison.json",comparison)
        frozen.require(graph_hash == hashlib_counts(policy) and head_hashes == {
            n:array_identity(p.detach().cpu().numpy()) for n,p in policy.named_parameters() if n != "brain.theta"},"Graph or frozen heads changed")
        result.update(baseline_two_frames_exact=True,heads_unchanged=True,graph_unchanged=True,
            local_directional_derivative=local_dot,archived_surrogate_directional_derivative=sg_dot,
            central_difference=comparison["central_difference"],diagnostic_only=True)
    finally:
        if policy is not None:
            if original_step is not None:
                policy._step = original_step
            with torch.no_grad():
                policy.brain.theta.copy_(torch.from_numpy(assets["theta0"]).to(policy.brain.theta.device))
            result["theta_restored"] = bool(np.array_equal(policy.brain.theta.detach().cpu().numpy(),assets["theta0"]))
        result["mps_memory"] = sampler.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--source-run",type=Path,default=frozen.ROOT/"data/x2_behavior_short_trial_theta_20261024_r1")
    parser.add_argument("--_worker",type=Path,help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker:
        return 0 if run_worker(args._worker)["status"] == "completed" else 1
    if not args.execute:
        print(json.dumps({"mode":"preview","brain_loaded":False,"configuration":configuration()},indent=2))
        return 0
    if args.output is None:
        parser.error("--execute requires a new --output")
    try:
        result = execute(args.output,args.source_run)
    except Exception as error:
        print(f"{type(error).__name__}: {error}",file=sys.stderr)
        return 1
    print(json.dumps({"status":result["status"],"output":result["output"],"reason":result["worker"].get("reason")}))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
