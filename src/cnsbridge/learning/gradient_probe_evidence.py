"""Torch-free fixed-source registration and saved-array diagnostic checks."""
from pathlib import Path

import numpy as np

from ..brain.metadata import canonical_hash
from .mps_archive import inspect_checkpoint
from .mps_contract import source_hashes as training_sources, validate_run
from .mps_evidence import scan_run, memory_boundary
from .training_archive import checkpoint_state_digest
from .training_contract import (PARAMETERS, bound, next_selection, read_arrays, read_json, require, sha256)

ROOT = Path(__file__).resolve().parents[3]
SOURCE_RUN = ROOT / "data/neural_training_mps_20261019_r3"
RUN_ID = "b7807f8e1997da44e46e83f39155e21bdce84aba020314a1ed68a145851413b7"
PROTOCOL = {"schema": "cnsbridge.mps_credit_probe.v1", "parent_window": 159, "input_window": 160,
            "horizons": [1000, 500, 250, 100], "forward_steps": 1000, "observe_every_steps": 50,
            "loss": "original_two_frame_action_mse_plus_gain_regularization",
            "cut": "detach_v_g_refractory_pending_only;retain_DN_running_sum_gradients",
            "comparisons": "bitwise_baseline_to_saved160_and_short_forward_to_baseline",
            "optimizer_updates": 0, "role": "diagnostic_not_training"}
NEW_SOURCES = ("learning/gradient_probe.py", "learning/gradient_probe_evidence.py",
               "learning/gradient_probe_worker.py", "learning/gradient_probe_math.py")


def source_hashes():
    return {**training_sources(), **{name: sha256(ROOT / "src/cnsbridge" / name) for name in NEW_SOURCES}}


def load_anchor():
    """Original readers validate the entire sealed lineage and complete archives."""
    run = read_json(SOURCE_RUN / "run.json")
    require(run["training_run_id"] == RUN_ID, "Wrong fixed GPU source run")
    data, initial = validate_run(run)
    chain = scan_run(SOURCE_RUN, run, data, initial)
    require(chain["completed_windows"] == 160, "Source epoch must be completely sealed")
    windows = {}
    for number in (159, 160):
        matches = list((SOURCE_RUN / "segments").glob(f"*/windows/{number:06d}"))
        require(len(matches) == 1, "Ambiguous diagnostic source window")
        windows[number] = matches[0]
    previous = inspect_checkpoint(windows[159] / "checkpoint", run)
    final = inspect_checkpoint(windows[160] / "checkpoint", run)
    old = read_arrays(windows[160] / "evidence.npz")
    event = read_json(windows[160] / "event.json")
    require(final["context"]["parent_checkpoint_id"] == previous["checkpoint_id"]
            and event["parent_checkpoint_id"] == previous["checkpoint_id"], "Fixed source lineage mismatch")
    _, selected = next_selection(data, 159)
    require(all(old[k].dtype == v.dtype and np.array_equal(old[k], v) for k, v in selected.items()), "Selected input differs")
    require(previous["context"]["completed_windows"] == 159 and previous["context"]["next_frame"] == 78,
            "Wrong fixed initial cursor")
    anchor = {"source_run": str(SOURCE_RUN), "source_run_id": RUN_ID, "run_sha256": sha256(SOURCE_RUN / "run.json"),
              "parent_checkpoint_id": previous["checkpoint_id"], "final_checkpoint_id": final["checkpoint_id"],
              "graph": run["graph"], "raw_counts_sha256": run["policy"]["model"]["graph"]["raw_ordered_counts_sha256"],
              "dataset_id": run["dataset_id"], "policy_id": run["policy"]["policy_id"],
              "files_sha256": {str(path.relative_to(SOURCE_RUN)): sha256(path) for number, directory in windows.items()
                                for path in [directory / "event.json", directory / "evidence.npz",
                                             directory / "checkpoint/checkpoint.json", directory / "checkpoint/arrays.npz"]}}
    return run, previous, final, selected, old, event, anchor


def validate_registration(reg, anchor):
    bound(reg, "registration_id")
    require(reg["protocol"] == PROTOCOL and reg["protocol_id"] == canonical_hash(PROTOCOL)
            and reg["anchor"] == anchor and reg["implementation_sha256"] == source_hashes(),
            "Diagnostic protocol, source or anchor changed")
    from .resource_guard import GIB, resource_budget
    budget = resource_budget(reg["resources"]["available_bytes_before_launch"], rss_bytes=6 * GIB, reserve_bytes=GIB)
    require(reg["resources"] == {**budget, "wall_limit_seconds": 600, "sampling_seconds": .1}
            and type(reg["parent_pid"]) is int and reg["parent_pid"] > 0, "Diagnostic resource registration differs")


def compare_arrays(actual, expected, keys):
    result = {}
    for key in keys:
        a, b = actual[key], expected[key]
        same_shape = a.shape == b.shape and a.dtype == b.dtype
        result[key] = {"exact": bool(same_shape and np.array_equal(a, b)),
                       "max_abs_difference": float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
                       if same_shape and a.size else None}
    return result


def comparisons(values, record, old, final, event, baseline=None):
    if baseline is None:
        reference = {**old, **{f"end.{key}": value for key, value in final["arrays"].items()
                              if key.startswith(("brain.", "drive."))}}
        keys = ["actions", "features_mv", *[f"pre_clip_{key}" for key in PARAMETERS],
                *[key for key in reference if key.startswith("end.")]]
        target = "original_saved_window160"
        states = record["start_state"] == event["start_state"] and record["end_state"] == event["end_state"]
    else:
        reference = baseline
        keys = ["actions", "features_mv", "pre_clip_readout_weight", "pre_clip_readout_bias",
                *[key for key in reference if key.startswith("end.")]]
        target = "same_run_horizon1000"
        states = record["start_state"] == event["start_state"] and record["end_state"] == event["end_state"]
    checked = compare_arrays(values, reference, keys)
    return {"target": target, "arrays": checked, "state_metadata_exact": states,
            "passed": bool(states and all(v["exact"] for v in checked.values()))}


def validate_completed(directory, reg, final, old, event):
    worker = read_json(directory / "worker_result.json")
    require(worker["registration_id"] == reg["registration_id"] and worker["status"] == "completed"
            and worker["completed_horizons"] == PROTOCOL["horizons"], "Incomplete diagnostic worker")
    expected_names = {"selection.npz", "resume.npz", "resume.json", "logit_loss.npz", "logit_loss.json"}
    expected_names.update(f"horizon-{h}/{name}" for h in PROTOCOL["horizons"] for name in ("values.npz", "report.json"))
    require(set(worker["artifacts_sha256"]) == expected_names, "Diagnostic artifact inventory differs")
    for name, digest in worker["artifacts_sha256"].items():
        require(sha256(directory / name) == digest, "Diagnostic artifact changed")
    baseline = None
    for horizon in PROTOCOL["horizons"]:
        values = read_arrays(directory / f"horizon-{horizon}/values.npz")
        report = read_json(directory / f"horizon-{horizon}/report.json")
        require(all(np.isfinite(v).all() for v in values.values()), "Nonfinite diagnostic evidence")
        actual = comparisons(values, report, old, final, event, baseline)
        require(report["comparison"] == actual and actual["passed"], "Diagnostic exactness comparison failed")
        require(report["forward_steps"] == 1000 and report["horizon_steps"] == horizon
                and [b["window_step"] for b in report["boundaries"]] == list(range(50, 1001, 50)), "Boundary inventory differs")
        require(all(b["cut_after_boundary"] == (b["window_step"] % horizon == 0 and b["window_step"] < 1000)
                    for b in report["boundaries"]), "Credit cut schedule differs")
        require(all(item.get("finite", True) for b in report["boundaries"] for item in b["adjoints"].values()),
                "Nonfinite boundary adjoint")
        actual_state = checkpoint_state_digest(final["metadata"]["state"],
                                               {key[4:]: value for key, value in values.items() if key.startswith("end.")})
        require(actual_state == report["end_state"], "Saved end-state digest differs")
        if baseline is None:
            baseline = values
    logit = read_arrays(directory / "logit_loss.npz")
    require(np.array_equal(logit["actions"], baseline["actions"]), "Fixed-feature loss changed deployment action")
    require(np.isfinite(logit["gradient"]).all() and np.allclose(logit["gradient"], logit["expected_gradient"], rtol=1e-5, atol=1e-7),
            "Logit gradient comparison failed")
    require(worker["raw_counts_sha256_before"] == worker["raw_counts_sha256_after"]
            == reg["anchor"]["raw_counts_sha256"], "Graph count digest changed")
    memory_boundary(directory, reg, worker["sealed_monotonic"])
    require(worker["mps_memory"]["error"] is None, "Memory sampling failed")
    return worker
