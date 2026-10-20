"""Torch-free fixed-source registration and saved-array diagnostic checks."""
from pathlib import Path
import hashlib
import subprocess

import numpy as np

from ..brain.metadata import canonical_hash
from .mps_archive import inspect_checkpoint
from .mps_contract import source_hashes as training_sources, validate_run
from .mps_evidence import scan_run, memory_boundary
from .training_archive import checkpoint_state_digest
from .training_contract import bound, next_selection, read_arrays, read_json, require, sha256

ROOT = Path(__file__).resolve().parents[3]
SOURCE_RUN = ROOT / "data/neural_training_mps_20261019_r3"
RUN_ID = "b7807f8e1997da44e46e83f39155e21bdce84aba020314a1ed68a145851413b7"
BRANCHES = (("reference-1", 1000, False), ("reference-2", 1000, False),
            ("horizon-500", 500, True), ("horizon-250", 250, True), ("horizon-100", 100, True))
THETA_TOLERANCE = {"rtol": 1e-5, "atol": 1e-7,
                   "source": "tests/test_mps_continuation.py", "require_same_zero_support_and_nonzero_sign": True}
PROTOCOL = {"schema": "cnsbridge.mps_credit_probe.v2", "parent_window": 159, "input_window": 160,
            "horizons": [1000, 500, 250, 100], "branches": [list(b) for b in BRANCHES],
            "forward_steps": 1000, "observe_every_steps": 50,
            "reference_forward": "original_policy.forward_tensor_without_hooks",
            "reused_observed_1000": "r1_bound_files_and_source_commit",
            "theta_tolerance": THETA_TOLERANCE,
            "loss": "original_two_frame_action_mse_plus_gain_regularization",
            "cut": "detach_v_g_refractory_pending_only;retain_DN_running_sum_gradients",
            "comparisons": "exact_forward_state_readout;numeric_full_theta;short_theta_unconstrained",
            "optimizer_updates": 0, "role": "diagnostic_not_training"}
R1_DIRECTORY = ROOT / "data/mps_gradient_probe_20261019_r1"
R1_COMMIT = "a0dcb3283d371339abf344278ab41c81b1da4100"
R1_FILES = {
    "registration.json": "0ab3a8ca7c45fd266baedd4a065efe98ed623948ae11150a91d36eccb0b6346e",
    "horizon-1000/values.npz": "6800351871f1398faa8575dfa964114af1f507a561c1b03439357e489f19ebfa",
    "horizon-1000/report.json": "38a74a9e3109cbe4bc80e1a506ee600711c57b3a774205c76f1d29d596338087",
    "worker_result.json": "04db480f404b1389cf057dbcf91994b3bf02146f7810a75e8f5cea0fa697c43b",
    "result.json": "d82b249fd6016fe906820e400cfe52f5d655c5ad34daeed974f00d76ae029daf"}
NEW_SOURCES = ("learning/gradient_probe.py", "learning/gradient_probe_evidence.py",
               "learning/gradient_probe_worker.py", "learning/gradient_probe_math.py")


def source_hashes():
    return {**training_sources(), **{name: sha256(ROOT / "src/malecns_lab" / name) for name in NEW_SOURCES}}


def load_r1():
    """Bind the failed experiment to its original saved bytes and Git source."""
    require(all(sha256(R1_DIRECTORY / name) == digest for name, digest in R1_FILES.items()),
            "Previously failed r1 evidence changed")
    reg = read_json(R1_DIRECTORY / "registration.json")
    bound(reg, "registration_id")
    require(reg["protocol"]["schema"] == "cnsbridge.mps_credit_probe.v1"
            and read_json(R1_DIRECTORY / "result.json")["status"] == "failed", "Wrong historical r1")
    current = training_sources()
    require(set(reg["implementation_sha256"]) == set(current) | set(NEW_SOURCES)
            and all(reg["implementation_sha256"][name] == digest for name, digest in current.items()),
            "Historical training source changed")
    for name in NEW_SOURCES:
        original = subprocess.check_output(["git", "show", f"{R1_COMMIT}:src/cnsbridge/{name}"], cwd=ROOT)
        require(hashlib.sha256(original).hexdigest() == reg["implementation_sha256"][name],
                "Historical diagnostic source does not match original commit")
    identity = {"path": str(R1_DIRECTORY), "source_commit": R1_COMMIT, "files_sha256": R1_FILES,
                "registration_id": reg["registration_id"]}
    return read_arrays(R1_DIRECTORY / "horizon-1000/values.npz"), read_json(R1_DIRECTORY / "horizon-1000/report.json"), identity


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
    _, _, r1 = load_r1()
    require(read_json(R1_DIRECTORY / "registration.json")["anchor"] == anchor, "r1 fixed anchor differs")
    anchor["observed_r1"] = r1
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


def theta_comparison(actual, expected):
    """Saved float32 gradient comparison; exactness is distinct from tolerance."""
    a, b = np.asarray(actual), np.asarray(expected)
    compatible = a.shape == b.shape and a.dtype == b.dtype == np.dtype("float32")
    result = {"same_shape_dtype": compatible, "tolerance": THETA_TOLERANCE,
              "finite": bool(np.isfinite(a).all() and np.isfinite(b).all()), "passed": False}
    if not compatible or not result["finite"]:
        return result
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    difference = np.abs(a64 - b64)
    nonzero = b != 0
    support_differences = int(np.count_nonzero((a == 0) != (b == 0)))
    sign_differences = int(np.count_nonzero((a != 0) & nonzero & (np.signbit(a) != np.signbit(b))))
    def ordered_bits(value):
        bits = value.view(np.uint32).astype(np.int64)
        return np.where(bits >= 0x80000000, 0x80000000 - (bits & 0x7fffffff), 0x80000000 + bits)
    ulps = np.abs(ordered_bits(a) - ordered_bits(b))
    reference_norm = float(np.linalg.norm(b64))
    difference_norm = float(np.linalg.norm(a64 - b64))
    within = bool(np.allclose(a, b, rtol=THETA_TOLERANCE["rtol"], atol=THETA_TOLERANCE["atol"]))
    result.update(exact=bool(np.array_equal(a, b)), difference_count=int(np.count_nonzero(a != b)),
                  max_abs_difference=float(difference.max(initial=0)),
                  max_relative_difference_on_nonzero_reference=float((difference[nonzero] / np.abs(b64[nonzero])).max(initial=0)),
                  max_ulp_difference=int(ulps.max(initial=0)),
                  ulp_difference_count=int(np.count_nonzero(ulps)),
                  relative_l2=difference_norm / reference_norm if reference_norm else (0. if not difference_norm else None),
                  zero_support_difference_count=support_differences, nonzero_sign_difference_count=sign_differences,
                  within_tolerance=within, passed=bool(within and not support_differences and not sign_differences))
    return result


def replay_comparison(values, record, reference, reference_record, target, *, full_theta):
    keys = ["actions", "features_mv", "pre_clip_readout_weight", "pre_clip_readout_bias",
            *[key for key in reference if key.startswith("end.")]]
    checked = compare_arrays(values, reference, keys)
    states = all(record[key] == reference_record[key] for key in ("start_state", "end_state"))
    result = {"target": target, "arrays": checked, "state_metadata_exact": states,
              "passed": bool(states and all(v["exact"] for v in checked.values()))}
    result["theta"] = theta_comparison(values["pre_clip_theta"], reference["pre_clip_theta"])
    result["theta_required_to_match"] = full_theta
    if full_theta:
        result["passed"] = result["passed"] and result["theta"]["passed"]
    return result


def branch_comparisons(values, record, old, final, event, r1_values, r1_report, *, baseline=None, baseline_report=None):
    if record["observed"]:
        require(baseline is not None and baseline_report is not None, "Short branch needs replay baseline")
        return {"baseline": replay_comparison(values, record, baseline, baseline_report, "reference-1", full_theta=False)}
    saved = {**old, **{f"end.{key}": value for key, value in final["arrays"].items()
                       if key.startswith(("brain.", "drive."))}}
    checked = {"saved160": replay_comparison(values, record, saved, event, "original_saved_window160", full_theta=True),
               "observed_r1": replay_comparison(values, record, r1_values, r1_report, "r1_observed_horizon1000", full_theta=True)}
    if baseline is not None:
        checked["repeat"] = replay_comparison(values, record, baseline, baseline_report, "reference-1", full_theta=True)
    return checked


def classify_repeatability(first, second):
    require(all(c["passed"] for report in (first, second) for c in report["comparisons"].values()),
            "Cannot classify failed numeric replay")
    checked = second["comparisons"]
    if not checked["repeat"]["theta"]["exact"]:
        label = "ordinary_replay_last_bit_variation_observed"
    elif not checked["observed_r1"]["theta"]["exact"]:
        label = "observer_version_or_environment_related_difference"
    elif not checked["saved160"]["theta"]["exact"]:
        label = "saved_window_difference_unresolved"
    else:
        label = "all_reference_theta_gradients_exact"
    return {"classification": label, "numeric_replay_passed": True,
            "specific_hardware_reduction_cause_proven": False,
            "r1_strict_result": "failed_unchanged"}


def validate_completed(directory, reg, final, old, event):
    worker = read_json(directory / "worker_result.json")
    require(worker["registration_id"] == reg["registration_id"] and worker["status"] == "completed"
            and worker["completed_branches"] == [b[0] for b in BRANCHES], "Incomplete diagnostic worker")
    expected_names = {"selection.npz", "resume.npz", "resume.json", "logit_loss.npz", "logit_loss.json", "repeatability.json"}
    expected_names.update(f"{name}/{file}" for name, _, _ in BRANCHES for file in ("values.npz", "report.json"))
    require(set(worker["artifacts_sha256"]) == expected_names, "Diagnostic artifact inventory differs")
    for name, digest in worker["artifacts_sha256"].items():
        require(sha256(directory / name) == digest, "Diagnostic artifact changed")
    r1_values, r1_report, identity = load_r1()
    require(identity == reg["anchor"]["observed_r1"], "r1 evidence identity changed")
    baseline = baseline_report = None
    reference_reports = []
    for name, horizon, observed in BRANCHES:
        values = read_arrays(directory / name / "values.npz")
        report = read_json(directory / name / "report.json")
        require(all(np.isfinite(v).all() for v in values.values()), "Nonfinite diagnostic evidence")
        actual = branch_comparisons(values, report, old, final, event, r1_values, r1_report,
                                   baseline=baseline, baseline_report=baseline_report)
        require(report["comparisons"] == actual and all(c["passed"] for c in actual.values()), "Diagnostic comparison failed")
        require(report["forward_steps"] == 1000 and report["horizon_steps"] == horizon and report["observed"] == observed,
                "Branch forward or observation differs")
        require(report["forward_path"] == ("observed_frame" if observed else "policy.forward_tensor"), "Forward path differs")
        expected_steps = list(range(50, 1001, 50)) if observed else []
        require([b["window_step"] for b in report["boundaries"]] == expected_steps, "Boundary inventory differs")
        require(all(b["cut_after_boundary"] == (b["window_step"] % horizon == 0 and b["window_step"] < 1000)
                    for b in report["boundaries"]), "Credit cut schedule differs")
        require(all(item.get("finite", True) for b in report["boundaries"] for item in b["adjoints"].values()),
                "Nonfinite boundary adjoint")
        actual_state = checkpoint_state_digest(final["metadata"]["state"],
                                               {key[4:]: value for key, value in values.items() if key.startswith("end.")})
        require(actual_state == report["end_state"], "Saved end-state digest differs")
        if not observed:
            reference_reports.append(report)
        if baseline is None:
            baseline, baseline_report = values, report
    require(read_json(directory / "repeatability.json") == classify_repeatability(*reference_reports),
            "Replay classification differs")
    logit = read_arrays(directory / "logit_loss.npz")
    require(np.array_equal(logit["actions"], baseline["actions"]), "Fixed-feature loss changed deployment action")
    require(np.isfinite(logit["gradient"]).all() and np.allclose(logit["gradient"], logit["expected_gradient"], rtol=1e-5, atol=1e-7),
            "Logit gradient comparison failed")
    require(worker["raw_counts_sha256_before"] == worker["raw_counts_sha256_after"]
            == reg["anchor"]["raw_counts_sha256"], "Graph count digest changed")
    memory_boundary(directory, reg, worker["sealed_monotonic"])
    require(worker["mps_memory"]["error"] is None, "Memory sampling failed")
    return worker
