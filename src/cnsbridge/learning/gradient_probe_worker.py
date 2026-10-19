"""Claimed diagnostic child: heavy imports happen only after parent/lock checks."""
import fcntl
import os
from pathlib import Path
import time
import traceback


def claim_parent(directory):
    from .training_contract import read_json, require
    reg = read_json(directory / "registration.json")
    require(os.getppid() == reg["parent_pid"], "Worker must belong to its supervising parent")
    lock_path = Path(__file__).resolve().parents[3] / "data/.continuous-training.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            raise ValueError("Parent must own the shared numerical lock")
    with (directory / "worker.claim").open("x") as stream:
        stream.write(f"{os.getpid()}\n")
    return reg


def run_worker(directory):
    from .resource_guard import write_json
    directory = Path(directory).resolve()
    reg = claim_parent(directory)
    began = time.monotonic()
    result = {"status": "failed", "registration_id": reg["registration_id"], "completed_branches": [],
              "optimizer_updates": 0}
    sampler = None

    def phase(name, **details):
        write_json(directory / "phase.json", {"phase": name, "wall_seconds": time.monotonic() - began, **details})
        if sampler is not None:
            sampler.sample(name)

    try:
        phase("imports")
        import numpy as np
        import torch
        from ..brain.mps_trainable import require_mps
        from .training_contract import read_json, require, sha256
        from .joint_probe_worker import raw_counts_digest
        from .mps_policy import MPSPolicy
        from .mps_state import restore, boundary_arrays
        from .mps_memory import MPSMemorySampler
        from .gradient_probe_evidence import (load_anchor, load_r1, validate_registration, branch_comparisons,
                                              classify_repeatability, BRANCHES)
        from .gradient_probe_math import compute_branch, logit_loss_check

        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        require_mps()
        phase("validate_registration")
        run, previous, final, selected, old, event, anchor = load_anchor()
        validate_registration(reg, anchor)
        sampler = MPSMemorySampler(directory / "mps_memory.csv", reg["resources"]["effective_rss_limit_bytes"])
        write_json(directory / "mps_budget.json", sampler.budget)
        phase("load_full_policy")
        cpu_run = read_json(Path(run["cpu_parent"]["path"]) / "run.json")
        policy = MPSPolicy.from_reference(run["graph"]["path"], cpu_run["policy"])
        require(policy.identity == run["policy"] and policy.frame_steps == 500, "Full policy or frame clock differs")
        result["raw_counts_sha256_before"] = raw_counts_digest(policy.brain.graph)
        require(result["raw_counts_sha256_before"] == anchor["raw_counts_sha256"], "Actual complete graph digest differs")
        np.savez(directory / "selection.npz", **selected)
        r1_values, r1_report, _ = load_r1()
        baseline = baseline_report = None
        reference_reports = []
        for name, horizon, observed in BRANCHES:
            branch = directory / name
            branch.mkdir(exist_ok=False)
            phase("restore_all_state", branch=name, horizon=horizon)
            bridge, state, restored = restore(policy, previous)
            # Restore verifies all 18 arrays including dormant CPU Adam state;
            # neither the bridge nor any other optimizer is stepped by this probe.
            if baseline is None:
                np.savez(directory / "resume.npz", **restored)
                write_json(directory / "resume.json", {"parent_checkpoint_id": previous["checkpoint_id"],
                           "exact": True, "arrays_sha256": sha256(directory / "resume.npz"),
                           "source_state": previous["metadata"]["state"]})
            del restored
            values, report = compute_branch(policy, state, selected, horizon, observed=observed,
                                            phase=lambda phase_name, **details: phase(phase_name, branch=name, **details))
            # Only parameter/Adam comparison here: branch state is separately saved
            # and the source state was never mutated by pure checkpoint closures.
            actual = boundary_arrays(policy, bridge, state)
            require(all(np.array_equal(actual[k], v) for k, v in previous["arrays"].items()),
                    "Diagnostic changed initial state, parameters or Adam")
            report["comparisons"] = branch_comparisons(values, report, old, final, event, r1_values, r1_report,
                                                       baseline=baseline, baseline_report=baseline_report)
            np.savez(branch / "values.npz", **values)
            write_json(branch / "report.json", report)
            require(all(np.isfinite(v).all() for v in values.values()), "Nonfinite branch value/gradient")
            require(all(c["passed"] for c in report["comparisons"].values()),
                    "Replay comparison failed; evidence retained under registered r2 criteria")
            require(all(item.get("finite", True) for b in report["boundaries"] for item in b["adjoints"].values()),
                    "Nonfinite boundary adjoint")
            result["completed_branches"].append(name)
            if not observed:
                reference_reports.append(report)
            if baseline is None:
                baseline, baseline_report = values, report
            if name == "reference-2":
                write_json(directory / "repeatability.json", classify_repeatability(*reference_reports))
            del values, report, state, bridge, actual
            phase("branch_sealed", branch=name, horizon=horizon)
        phase("logit_loss_fixed_features")
        values, report = logit_loss_check(policy, old["features_mv"], selected)
        np.savez(directory / "logit_loss.npz", **values)
        write_json(directory / "logit_loss.json", report)
        require(np.array_equal(values["actions"], baseline["actions"]), "Logit diagnostic changed original action")
        require(report["trial_logit_loss"] < report["loss"], "Controlled logit descent did not reduce loss")
        result["raw_counts_sha256_after"] = raw_counts_digest(policy.brain.graph)
        require(result["raw_counts_sha256_after"] == result["raw_counts_sha256_before"], "Graph mutated")
        names = ["selection.npz", "resume.npz", "resume.json", "logit_loss.npz", "logit_loss.json", "repeatability.json"]
        names += [f"{name}/{file}" for name, _, _ in BRANCHES for file in ("values.npz", "report.json")]
        result["artifacts_sha256"] = {name: sha256(directory / name) for name in names}
        result.update(status="completed", reason="repeatability_and_fixed_horizon_diagnostic_sealed", sealed_monotonic=time.monotonic())
    except BaseException as error:
        result.update(reason=f"{type(error).__name__}:{error}")
        traceback.print_exc()
    finally:
        if sampler is not None:
            result["mps_memory"] = sampler.close()
            if result["mps_memory"]["error"] is not None:
                result.update(status="failed", reason="mps_memory_sampling_failed")
        result["wall_seconds"] = time.monotonic() - began
        write_json(directory / "worker_result.json", result)
        write_json(directory / "phase.json", {"phase": "finished", "wall_seconds": result["wall_seconds"]})
    return result
