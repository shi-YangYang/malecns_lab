"""Register, validate or execute an explicit offline complete-connectome study."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..diagnostics.runtime_v2 import NeuralReplayV2
from .contracts_v2 import RIDGE_CANDIDATES, digest, write_json
from .dataset_v2 import assert_paired_rows, load_dataset
from .evaluation_v2 import paired_scene_comparison, score_predictions
from .experiment_protocol_v2 import register_experiment, source_from_entry, template, validate_registration
from .research_replay_v2 import ResearchNeuralReplayV2, computation_identity, record_replay
from .train_v2 import fit_readouts, for_input_seed, save_fit


def _active_splits(protocol: dict) -> tuple[str, ...]:
    return ("train", "val", "test") if protocol["phase"] == "confirmatory" else ("train", "val")


def _planned_sources(protocol: dict) -> tuple[list, list, list]:
    entries = [source for source in protocol["sources"] if source["split"] in _active_splits(protocol)]
    sources, planned, labels = [], [], []
    for index, entry in enumerate(entries):
        source = source_from_entry(entry)
        if source.metadata != entry["input_source"]:
            raise ValueError("Source metadata changed after registration")
        sources.append(source)
        for segment in entry["segments"]:
            current = {**segment, "segment_id": f"source_{index:03d}_{segment['segment_id']}", "source_index": index, "repetition": 0}
            planned.append((source, current))
            labels.append({"segment_id": current["segment_id"], "labels": entry["labels"], "split": entry["split"],
                           "label_format": entry.get("label_format", "v2")})
    return sources, planned, labels


def _assert_fair_datasets(real, control, splits):
    for split in splits:
        assert_paired_rows(real, control, split)
    if not np.array_equal(real.features["visual"], control.features["visual"], equal_nan=True):
        raise ValueError("Graph conditions changed shared visual features; no fair comparison")


def _acute_result(real_dataset, dataset, real_result: dict, graph: dict) -> dict:
    """Explicit P-only intervention using a G readout; never a general schema bypass."""
    if graph["role"] not in ("P_target", "P_matched") or graph["parameters"].get("intervention_mode") != "acute":
        raise ValueError("Fixed-readout evaluation is restricted to registered acute P interventions")
    splits = _active_splits(dataset.protocol)
    _assert_fair_datasets(real_dataset, dataset, splits)
    model = real_result["models"]["neural"]
    if model.schema != real_dataset.schemas["neural"]:
        raise ValueError("Acute readout must be the registered G model for this input seed")
    original, intervention = model.schema.as_dict(), dataset.schemas["neural"].as_dict()
    if {key: value for key, value in original.items() if key != "source_id"} != {
            key: value for key, value in intervention.items() if key != "source_id"}:
        raise ValueError("Acute intervention changed output identities, order or time contract")
    valid = dataset.eligible & np.array([row["split"] in splits for row in dataset.rows])
    prediction = np.full(dataset.targets.shape, np.nan)
    x = dataset.features["neural"][valid]
    prediction[valid] = ((x - model.mean) / model.scale) @ model.weights + model.bias
    if not np.isfinite(prediction[valid]).all():
        raise ValueError("Acute predictions are nonfinite")
    predictions = {"neural": prediction, **{name: real_result["predictions"][name].copy() for name in ("visual", "mean", "zero")}}
    metrics = {split: {name: score_predictions(dataset, values, model.supported, split=split)
                       for name, values in predictions.items()} for split in splits}
    return {"predictions": predictions, "metrics": metrics, "active_splits": list(splits),
            "training_id": real_result["training_id"], "readout_application": "acute G readout on registered P features; no refit",
            "trained_schema": original, "evaluated_schema": intervention,
            "graph_source_id": graph["source_id"], "input_seed": dataset.segments[0]["run_seed"]}


def _save_acute(out: Path, dataset, result: dict):
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "acute_report.json", {key: value for key, value in result.items() if key != "predictions"})
    write_json(out / "rows.json", dataset.rows)
    with (out / "predictions.npz").open("xb") as stream:
        np.savez_compressed(stream, row_key=np.asarray([row["row_key"] for row in dataset.rows]),
                            targets=dataset.targets, supervised_mask=dataset.mask, prediction_valid=dataset.eligible,
                            **{f"prediction_{name}": value for name, value in result["predictions"].items()})


def execute_experiment(registration_path: str | Path, out: str | Path) -> dict:
    """Every graph/seed gets independent extraction and fit; incomplete plans stay incomplete."""
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    jobs, fit_jobs = [], []
    stage = "registration"
    summary = None

    def check_budget(current_stage: str):
        nonlocal stage
        stage = current_stage
        if time.perf_counter() >= deadline:
            raise TimeoutError(f"Total budget exhausted at {stage}")

    try:
        registration = validate_registration(registration_path)
        write_json(out / "registration.json", registration)
        protocol = registration["protocol"]
        deadline = began + protocol["budget_seconds"]
        splits = _active_splits(protocol)
        graphs = sorted(protocol["graphs"], key=lambda row: row["role"] != "G")
        real_id = next(graph["id"] for graph in graphs if graph["role"] == "G")
        sources, planned, labels = _planned_sources(protocol)
        for graph in graphs:
            for seed in protocol["input_seeds"]:
                jobs.append({"graph_id": graph["id"], "role": graph["role"], "input_seed": seed,
                             "path": str(out / "replay" / f"{graph['id']}_seed_{seed}"), "status": "pending"})
                fit_jobs.append({"graph_id": graph["id"], "input_seed": seed, "status": "pending"})
        write_json(out / "execution_plan.json", {"registration_id": registration["registration_id"], "replay_jobs": jobs,
                   "fit_jobs": fit_jobs, "active_splits": list(splits), "ridge_candidates": list(RIDGE_CANDIDATES),
                   "held_out_sources_not_executed": [entry["id"] for entry in protocol["sources"] if entry["split"] not in splits]})
        common = None
        for job in jobs:
            check_budget(f"replay_before:{job['graph_id']}:{job['input_seed']}")
            graph = next(graph for graph in graphs if graph["id"] == job["graph_id"])
            started = time.perf_counter()
            try:
                job["status"] = "running"
                stage = f"replay:{job['graph_id']}:{job['input_seed']}"
                engine = (NeuralReplayV2(protocol["parent"], run_seed=job["input_seed"]) if graph["role"] == "G" else
                          ResearchNeuralReplayV2(protocol["parent"], graph["path"], run_seed=job["input_seed"]))
                if engine.graph.meta["source_id"] != graph["source_id"]:
                    raise ValueError("Runtime graph identity differs from registered graph")
                current = computation_identity(engine.provenance)
                if common is not None and common != current:
                    raise ValueError("Registered graphs differ in non-graph computation")
                common = current
                budget = min(protocol["replay_budget_seconds"], max(0, deadline - started))
                replay_summary = record_replay(engine, sources, planned, Path(job["path"]), budget_seconds=budget,
                                                started=started, registration_id=registration["registration_id"])
                job.update(status=replay_summary["status"], source_id=engine.source_id,
                           coverage=replay_summary["source_coverage"], common_computation_id=current)
                del engine
                check_budget(stage)
            except TimeoutError:
                raise
            except Exception as error:
                job.update(status="failed", error=repr(error))
            check_budget(stage)
        datasets, fit_results, condition_metrics = {}, {}, {}
        incomplete = any(job["status"] != "complete" for job in jobs)
        if not incomplete:
            for graph in graphs:
                check_budget(f"dataset:{graph['id']}")
                entries = [{"path": str(Path(job["path"]) / label["segment_id"]),
                            **{key: value for key, value in label.items() if key != "segment_id"}}
                           for job in jobs if job["graph_id"] == graph["id"] for label in labels]
                learning = {key: protocol[key] for key in ("protocol_id", "phase", "task", "practical_delta", "budget_seconds", "bootstrap")}
                learning.update(format="cnsbridge.learning_protocol.v2", segments=entries,
                                experiment_registration_id=registration["registration_id"])
                learning_path = out / f"learning_{graph['id']}.json"
                write_json(learning_path, learning)
                datasets[graph["id"]] = load_dataset(learning_path)
                check_budget(stage)
            for graph in graphs:
                check_budget(f"fairness:{graph['id']}")
                _assert_fair_datasets(datasets[real_id], datasets[graph["id"]], splits)
                check_budget(stage)
            for graph in graphs:
                dataset = datasets[graph["id"]]
                aggregate = {name: np.full(dataset.targets.shape, np.nan) for name in ("neural", "visual", "mean", "zero")}
                for seed in protocol["input_seeds"]:
                    item = next(item for item in fit_jobs if item["graph_id"] == graph["id"] and item["input_seed"] == seed)
                    check_budget(f"fit_before:{graph['id']}:{seed}")
                    try:
                        item["status"] = "running"
                        current = for_input_seed(dataset, seed)
                        destination = out / "fits" / f"{graph['id']}_seed_{seed}"
                        item["path"] = str(destination)
                        acute = graph["role"].startswith("P_") and graph["parameters"]["intervention_mode"] == "acute"
                        if acute:
                            check_budget(f"acute_compute:{graph['id']}:{seed}")
                            result = _acute_result(for_input_seed(datasets[real_id], seed), current,
                                                   fit_results[(real_id, seed)], graph)
                            check_budget(stage)
                            check_budget(f"acute_persist:{graph['id']}:{seed}")
                            _save_acute(destination, current, result)
                        else:
                            check_budget(f"fit_compute:{graph['id']}:{seed}")
                            result = fit_readouts(current, deadline=deadline)
                            check_budget(stage)
                            check_budget(f"fit_persist:{graph['id']}:{seed}")
                            save_fit(destination, current, result)
                        check_budget(stage)
                        if graph["id"] == real_id:
                            fit_results[(real_id, seed)] = result
                        selected = np.array([row["run_seed"] == seed for row in dataset.rows])
                        for name in aggregate:
                            aggregate[name][selected] = result["predictions"][name]
                        item.update(status="complete", training_id=result["training_id"], path=str(destination),
                                    method="fixed_G_readout_acute_P" if acute else "independent_nine_candidate_fit")
                    except TimeoutError:
                        raise
                    except Exception as error:
                        item.update(status="failed", error=repr(error))
                        incomplete = True
                    check_budget(stage)
                if all(item["status"] == "complete" for item in fit_jobs if item["graph_id"] == graph["id"]):
                    supported = np.array([True, False, False, False, False, True])
                    metrics = {}
                    for split in splits:
                        metrics[split] = {}
                        for name, values in aggregate.items():
                            check_budget(f"aggregate_score:{graph['id']}:{split}:{name}")
                            metrics[split][name] = score_predictions(dataset, values, supported, split=split)
                            check_budget(stage)
                    condition_metrics[graph["id"]] = metrics
                    if any(value["coverage_status"] != "all_scenes_axes_seeds_scored" for split in metrics.values() for value in split.values()):
                        incomplete = True
                    check_budget(f"aggregate_persist:{graph['id']}")
                    write_json(out / f"metrics_{graph['id']}.json", metrics)
                    check_budget(stage)
        else:
            for item in fit_jobs:
                item.update(status="not_executed", reason="at_least_one_registered_replay_incomplete; no successful-subset fitting")
        comparisons = []
        if not incomplete:
            for comparison in protocol["comparisons"]:
                for split in splits:
                    control = comparison["control"]
                    check_budget(f"comparison:{real_id}:{control}:{split}")
                    real = condition_metrics[real_id][split]["neural"]
                    other = (condition_metrics[real_id][split][control] if control in ("visual", "mean", "zero") else
                             condition_metrics[control][split]["neural"])
                    result = paired_scene_comparison(real, other, practical_delta=protocol["practical_delta"],
                                                      seed=protocol["bootstrap"]["seed"], replicates=protocol["bootstrap"]["replicates"])
                    check_budget(stage)
                    # Multiple graph/seed conditions are descriptive, not an automatic corrected confirmatory claim.
                    result["calculated_interval_interpretation"] = result.pop("interpretation")
                    result["interpretation"] = "exploratory_pilot_only" if protocol["phase"] == "pilot" or split != "test" else "registered_descriptive_comparison"
                    comparisons.append({**comparison, "split": split, "result": result})
        check_budget("final_summary")
        summary = {"format": "cnsbridge.connectome_experiment_run.v2", "status": "incomplete" if incomplete else "complete",
                   "registration_id": registration["registration_id"], "phase": protocol["phase"], "replay_jobs": jobs,
                   "fit_jobs": fit_jobs, "comparisons": comparisons,
                   "graph_variability": [{"graph_id": graph["id"], "role": graph["role"], "graph_seed": graph["graph_seed"],
                                           "metrics": condition_metrics.get(graph["id"])} for graph in graphs],
                   "common_computation_id": common, "wall_seconds": time.perf_counter() - began,
                   "test_evaluated": protocol["phase"] == "confirmatory" and bool(condition_metrics),
                   "claim_scope": "offline registered conditions; no graph-seed ranking test, multiple-testing correction, flight or transfer claim",
                   "incomplete_policy": "all registered jobs retained; no comparison after any failed or incomplete condition"}
        check_budget("final_persist")
        write_json(out / "run_summary.json", summary)
        check_budget(stage)
        return summary
    except BaseException as error:
        reason = "total_budget_exhausted" if isinstance(error, TimeoutError) else "execution_stopped"
        for item in jobs + fit_jobs:
            if item["status"] == "pending":
                item.update(status="not_executed", reason=reason, stopped_stage=stage)
            elif item["status"] == "running":
                item.update(status="failed", reason=reason, stopped_stage=stage, error=repr(error))
        failure = {"status": "incomplete", "reason": reason, "stage": stage, "error": repr(error), "replay_jobs": jobs,
                   "fit_jobs": fit_jobs, "wall_seconds": time.perf_counter() - began,
                   "comparisons": [], "claim": "partial artifacts are not completed experiment results"}
        # The final write itself may exhaust the budget. Revoke only this run's
        # just-written summary, preserving every other local artifact for audit.
        summary_path = out / "run_summary.json"
        if summary_path.exists():
            summary.update(failure)
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        write_json(out / "run_failed.json", failure)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("template", help="Write incomplete template; never fills missing empirical tolerances")
    create.add_argument("--out", type=Path, required=True)
    register = commands.add_parser("register", help="Freeze explicit graphs, real sources, labels, splits and budgets")
    register.add_argument("--protocol", type=Path, required=True)
    register.add_argument("--out", type=Path, required=True)
    check = commands.add_parser("validate", help="Read-only hash/identity check of frozen registration")
    check.add_argument("--registration", type=Path, required=True)
    execute = commands.add_parser("execute", help="Offline extraction and independent fits; no hardware")
    execute.add_argument("--registration", type=Path, required=True)
    execute.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "template":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.out, template())
        print(f"Saved unregistered template {args.out}")
    elif args.command == "register":
        result = register_experiment(args.protocol, args.out)
        print(f"Registered {result['registration_id']}")
    elif args.command == "validate":
        result = validate_registration(args.registration)
        print(f"Frozen files and code unchanged: {result['registration_id']}")
    else:
        result = execute_experiment(args.registration, args.out)
        print(f"Saved {args.out}; status={result['status']}")


if __name__ == "__main__":
    main()
