"""P1 algorithm/lifecycle fixtures; real full-brain evidence is recorded separately."""

import json
import time

import numpy as np
import pytest
from scipy import sparse

from cnsbridge.brain.connectome import Connectome
from cnsbridge.brain.input_events import bernoulli_events
from cnsbridge.brain.lif import LIFNetwork, LIFParams
from cnsbridge.diagnostics.history_v2 import CausalHistoryV2
from cnsbridge.diagnostics.replay_storage_v2 import run_segment, source_coverage
from cnsbridge.diagnostics.runtime_v2 import NeuralReplayV2
from cnsbridge.diagnostics.sources_v2 import GeneratedSourceV2, segment_ranges
from cnsbridge.senses.encoding_v2 import INPUT_GROUPS, MAX_RATES, GroupInputV2
from cnsbridge.senses.vision_v2 import VisionV2Encoder


def contract_engine(seed=11):
    """Test-only wiring; the public constructor requires a verified full_reference."""
    engine = NeuralReplayV2.__new__(NeuralReplayV2)
    n = 64
    engine.graph = Connectome("contract_fixture", sparse.eye(n, format="csc"), np.full(n, "fixture"), np.full(n, "L"),
                               body_ids=np.arange(100, 100+n), groups={name: np.array([2*i, 2*i+1]) for i, name in enumerate(INPUT_GROUPS)})
    engine.run_seed = seed
    engine.vision = VisionV2Encoder()
    engine.encoder = GroupInputV2(engine.graph)
    engine.output_indices = np.arange(31, 64)
    engine.output_body_ids = engine.graph.body_ids[engine.output_indices]
    engine.history = CausalHistoryV2(engine.output_body_ids)
    engine.params = LIFParams()
    engine.net = LIFNetwork(engine.graph.weights, engine.params, seed=seed)
    engine.net.set_bias(engine.output_indices, 8.0)
    engine.group_names = tuple(INPUT_GROUPS)
    engine.groups = engine.encoder.groups
    engine.source_id = "test_contract_only"
    engine.reset()
    return engine


def test_common_events_survive_order_subset_time_partition_and_other_ablation():
    ids = np.array([104, 200, 301, 999])
    rates = np.array([100, 200, 300, 400])
    args = dict(seed=9, start_step=7, steps=200, dt_ms=0.5)
    full = bernoulli_events(ids, rates, **args)
    assert np.array_equal(full[:, [3, 0]], bernoulli_events(ids[[3, 0]], rates[[3, 0]], **args))
    reduced = rates.copy()
    reduced[1] = 0
    assert np.array_equal(full[:, [0, 2, 3]], bernoulli_events(ids, reduced, **args)[:, [0, 2, 3]])
    split = np.vstack((bernoulli_events(ids, rates, seed=9, start_step=7, steps=50, dt_ms=.5),
                       bernoulli_events(ids, rates, seed=9, start_step=57, steps=150, dt_ms=.5)))
    assert np.array_equal(full, split)
    assert not np.array_equal(full, bernoulli_events(ids, rates, **{**args, "seed": 10}))
    with pytest.raises(ValueError, match="probabilities"):
        bernoulli_events(ids, np.full(4, 3000), **args)


def test_external_events_replace_legacy_rng_and_validate_before_advancing():
    net = LIFNetwork(sparse.csc_matrix((3, 3)), seed=1)
    net.set_input(np.array([1, 2]), 999999)
    state = json.dumps(net.rng.bit_generator.state)
    net.step(external_events=np.array([0]))
    assert net.g[0] > 0 and net.g[1] == 0 and net.g[2] == 0
    assert json.dumps(net.rng.bit_generator.state) == state
    previous = (net._t, net.g.copy())
    with pytest.raises(ValueError):
        net.step(external_events=np.array([0, 0]))
    assert net._t == previous[0] and np.array_equal(net.g, previous[1])


def test_history_order_dimensions_ema_and_warmup():
    history = CausalHistoryV2(range(100, 133))
    for index in range(4):
        neural, visual = history.append(np.full(33, index + 1), np.full(14, 0.25))
    assert neural.shape == (132,) and visual.shape == (98,)
    assert np.array_equal(neural.reshape(4, 33)[:, 0], [1, 2, 3, 4])
    assert np.allclose(visual[56:70], 0.25 * (1-np.exp(-4)))
    for _ in range(36):
        history.append(np.ones(33), np.zeros(14))
    assert not history.ready
    history.append(np.ones(33), np.zeros(14))
    assert history.ready
    history.reset()
    assert history.steps == 0 and not history.neural.any() and not history.ema.any()


def test_group_broadcast_and_invalid_observation_rejected():
    engine = contract_engine()
    source = GeneratedSourceV2("right")
    obs = engine.vision.encode(source.frame(0), 0.0)
    rates, group_rates = engine.encoder.rates(obs)
    assert np.array_equal(group_rates, MAX_RATES * obs.features)
    for i, name in enumerate(INPUT_GROUPS):
        assert np.array_equal(rates[engine.encoder.group_positions[name]], [group_rates[i]] * 2)
    assert not any(name.startswith("HAL") for name in engine.encoder.groups)
    bad = engine.vision.encode(None, 0.05)
    with pytest.raises(ValueError, match="Invalid visual"):
        engine.encoder.rates(bad)


def test_replay_failure_freezes_all_temporal_state_and_reset_reproduces():
    engine = contract_engine()
    source = GeneratedSourceV2("static")
    first = [engine.step(source.frame(index), index / 20) for index in range(4)]
    assert first[0]["target_time_s"] == .05 and first[0]["simulation_end_ms"] == 50
    frozen = [value.copy() for value in (engine.net.v, engine.net.g, engine.net._buf, engine.net._ref_left,
                                         engine.history.neural, engine.history.visual, engine.history.ema,
                                         engine.neuron_total_counts, engine.input_total_events)]
    steps = engine.net._t
    failed = engine.step(None, .2)
    assert not failed["valid"] and "input_rates_hz" not in failed
    assert failed["simulation_start_ms"] == failed["simulation_end_ms"] == 200
    assert engine.net._t == steps
    for previous, current in zip(frozen, (engine.net.v, engine.net.g, engine.net._buf, engine.net._ref_left,
                                         engine.history.neural, engine.history.visual, engine.history.ema,
                                         engine.neuron_total_counts, engine.input_total_events)):
        assert np.array_equal(previous, current)
    assert not engine.step(source.frame(5), .25)["valid"]
    engine.reset()
    second = [engine.step(source.frame(index), index / 20) for index in range(4)]
    for a, b in zip(first, second):
        for key in ("input_event_counts", "neural_counts", "neural_history_hz", "visual_history"):
            assert np.array_equal(a[key], b[key])


def test_storage_failure_coverage_prefix_missing_and_budget_zero(tmp_path):
    engine = contract_engine()
    source = GeneratedSourceV2("static", observation_seconds=.1)
    original = source.frame
    source.frame = lambda index: None if index == 2 else original(index)
    segment = {"segment_id": "one", "start_frame": 0, "end_frame": 42}
    result = run_segment(engine, source, segment, tmp_path / "failed")
    assert result["coverage"]["valid_input_frames"] == 2
    assert result["coverage"]["failed_frames"] == 1
    assert result["coverage"]["unprocessed_tail_frames"] == 39
    with np.load(tmp_path / "failed/rows.npz", allow_pickle=False) as rows:
        assert np.isnan(rows["neural_history_hz"][-1]).all()
        assert (rows["input_event_counts"][-1] == -1).all()
        assert rows["valid"].tolist() == [True, True, False]
    metadata = (tmp_path / "failed/metadata.json").read_text()
    assert "NaN" not in metadata.replace('"NaN for uncomputed floating values;', '"missing')
    zero = run_segment(engine, source, segment, tmp_path / "budget", deadline=time.perf_counter()-1)
    assert zero["status"] == "budget_exhausted"
    assert zero["coverage"]["attempted_frames"] == 0 and zero["coverage"]["unprocessed_tail_frames"] == 42
    with pytest.raises(FileExistsError):
        run_segment(engine, source, segment, tmp_path / "failed")


def test_predeclared_segments_cannot_overlap_or_escape_directory(tmp_path):
    path = tmp_path / "segments.json"
    for payload in ([{"segment_id":"../bad", "start_frame":0, "end_frame":10}],
                    [{"segment_id":"a", "start_frame":0, "end_frame":20}, {"segment_id":"b", "start_frame":10, "end_frame":30}]):
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError):
            segment_ranges(path, 40)


def test_whole_source_coverage_includes_predeclared_gaps_and_zero_valid_scenes():
    rows = [{"segment": {"source_index": 0, "repetition": 0},
             "input_source": {"scene_id": "a", "session_id": "s", "family_id": "f", "total_frames": 100},
             "coverage": {"planned_frames": 40, "valid_input_frames": 0, "failed_frames": 1,
                          "unprocessed_tail_frames": 39, "prediction_valid_frames": 0}}]
    row = source_coverage(rows)[0]
    assert row["outside_predeclared_segments"] == 60
    assert row["valid_fraction_of_whole_source"] == 0 and row["prediction_fraction_of_whole_source"] == 0
    assert row["failed_frames"] + row["unprocessed_tail_frames"] + row["outside_predeclared_segments"] == 100
