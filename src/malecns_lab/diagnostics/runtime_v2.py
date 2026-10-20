"""Full-reference neural replay; labels and device commands have no entry point."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..brain.input_events import EVENT_PROTOCOL, bernoulli_events
from ..brain.lif import LIFNetwork, LIFParams, StepStats
from ..brain.metadata import canonical_hash, file_hashes
from ..brain.reference import load_reference
from ..senses.encoding_v2 import INPUT_GROUPS, GroupInputV2
from ..senses.vision_v2 import FRAME_INTERVAL_S, VisionV2Config, VisionV2Encoder
from .history_v2 import CausalHistoryV2

HIGH_RATE_HZ = 200.0  # engineering diagnostic, not an exclusion or optimization rule
MODEL_FILES = ("brain/connectome.py", "brain/metadata.py", "brain/reference.py", "brain/lif.py", "brain/input_events.py",
               "senses/encoding_v2.py", "senses/vision_v2.py", "diagnostics/history_v2.py", "diagnostics/runtime_v2.py")


class NeuralReplayV2:
    def __init__(self, brain_path: str | Path, *, run_seed: int = 0, vision_config: VisionV2Config | None = None):
        if isinstance(run_seed, bool) or not isinstance(run_seed, int) or not 0 <= run_seed < 2**64:
            raise ValueError("run_seed must be an unsigned 64-bit integer")
        self.graph = load_reference(brain_path)
        self.run_seed = run_seed
        self.vision = VisionV2Encoder(vision_config)
        self.encoder = GroupInputV2(self.graph)
        mapping = self.graph.meta["provenance"]["group_mapping"]
        self.output_body_ids = np.asarray(mapping["output_body_ids"], dtype=np.int64)
        self.output_indices = np.searchsorted(self.graph.body_ids, self.output_body_ids)
        self.history = CausalHistoryV2(self.output_body_ids)
        # Explicit frozen values, independent of V1 defaults/config merges.
        self.params = LIFParams(v_rest=-52.0, v_reset=-52.0, v_th=-45.0, tau_m=20.0, tau_syn=5.0,
                                t_ref=2.2, delay=1.8, w_syn=0.275, poisson_factor=250.0, dt=0.5, noise_mv=0.0)
        self.net = LIFNetwork(self.graph.weights, self.params, seed=run_seed)
        self.net.set_bias(self.graph.group("DNg02_L"), 8.0)
        self.net.set_bias(self.graph.group("DNg02_R"), 8.0)
        self.group_names = tuple(sorted(mapping["rules"]))
        self.groups = {name: self.graph.group(name) for name in self.group_names}
        package = Path(__file__).parents[1]
        self.provenance = {
            "protocol": "cnsbridge.neural_replay.v2", "graph_source_id": self.graph.meta["source_id"],
            "graph_sha256": self.graph.meta["graph_sha256"], "node_count": self.graph.n,
            "edge_count": self.graph.n_connections, "group_mapping": mapping, "vision": self.vision.config.identity(),
            "input_encoding": self.encoder.identity(), "events": EVENT_PROTOCOL, "lif": self.params.to_dict(),
            "actual_delay_ms": self.net.delay_steps * self.params.dt, "actual_refractory_ms": self.net.ref_steps * self.params.dt,
            "bias_mv": {"DNg02_L": 8.0, "DNg02_R": 8.0}, "features": self.history.identity(),
            "time": {"clock": "20Hz_video_playback_not_camera_exposure", "input_cutoff": "frame_index * 0.05 s",
                     "simulation_window_ms": 50.0, "target_horizon_s": 0.05,
                     "prediction_valid": "after 40 warmup windows; noninitial valid frame with target inside segment",
                     "host_time": "optional dequeue diagnostic only"},
            "diagnostics": {"high_rate_hz": HIGH_RATE_HZ, "evidence": "engineering_summary_not_exclusion"},
            "source_sha256": {name: file_hashes(package / name)["sha256"] for name in MODEL_FILES},
        }
        self.source_id = canonical_hash(self.provenance)
        self.reset()

    def reset(self):
        self.vision.reset()
        self.history.reset()
        self.net.reset()
        self.net.rng = np.random.default_rng(self.run_seed)
        self.net.stats = StepStats()
        self.net.set_input(np.zeros(0, dtype=np.int64), 0.0)
        self.neuron_total_counts = np.zeros(self.graph.n, dtype=np.int64)
        self.input_total_events = np.zeros(14, dtype=np.int64)

    def step(self, image, video_time_s: float, *, target_available: bool = True) -> dict:
        started = time.perf_counter()
        observation = self.vision.encode(image, video_time_s)
        row = {"vision": observation, "valid": observation.valid, "prediction_valid": False,
               "input_cutoff_s": observation.input_cutoff_s,
               "target_time_s": None if observation.input_cutoff_s is None else observation.input_cutoff_s + FRAME_INTERVAL_S,
               "target_available": target_available, "simulation_start_ms": self.net.t_ms,
               "simulation_end_ms": self.net.t_ms, "processing_ms": 0.0}
        if not observation.valid:
            row["processing_ms"] = (time.perf_counter() - started) * 1000
            return row
        rates, group_rates = self.encoder.rates(observation)
        events = bernoulli_events(self.encoder.body_ids, rates, seed=self.run_seed, start_step=self.net._t,
                                  steps=100, dt_ms=self.params.dt)
        counts = np.zeros(self.graph.n, dtype=np.int32)
        for step_events in events:
            spikes = self.net.step(external_events=self.encoder.indices[step_events])
            counts[spikes] += 1
        if not np.isfinite(self.net.v).all() or not np.isfinite(self.net.g).all():
            raise FloatingPointError("neural_nonfinite_state; stop this segment")
        neural_counts = counts[self.output_indices]
        neural_rates = neural_counts.astype(np.float64) * 20.0
        neural_history, visual_history = self.history.append(neural_rates, observation.features)
        event_counts = self.encoder.count_events(events)
        self.neuron_total_counts += counts
        self.input_total_events += event_counts
        full_rates = counts.astype(np.float64) * 20.0
        row.update({
            "prediction_valid": self.history.ready and not observation.initialization and target_available,
            "neural_counts": neural_counts, "neural_rates_hz": neural_rates, "neural_history_hz": neural_history,
            "visual_features": observation.features, "visual_history": visual_history, "input_rates_hz": group_rates,
            "input_event_counts": event_counts, "group_rates_hz": np.array([full_rates[idx].mean() for idx in self.groups.values()]),
            "simulation_end_ms": self.net.t_ms,
            "network": {"spikes": int(counts.sum()), "active_neurons": int(np.count_nonzero(counts)),
                        "silent_fraction": float(np.mean(counts == 0)), "high_rate_fraction": float(np.mean(full_rates >= HIGH_RATE_HZ)),
                        "rate_p50_hz": float(np.percentile(full_rates, 50)), "rate_p99_hz": float(np.percentile(full_rates, 99)),
                        "max_rate_hz": float(full_rates.max())},
        })
        row["processing_ms"] = (time.perf_counter() - started) * 1000
        return row

    def segment_summary(self) -> dict:
        return {"simulated_ms": self.net.t_ms, "steps": self.net.stats.steps, "spikes": self.net.stats.spikes,
                "synaptic_events": self.net.stats.synaptic_events, "groups": list(INPUT_GROUPS),
                "input_events": self.input_total_events.tolist(), "neurons_ever_active": int(np.count_nonzero(self.neuron_total_counts)),
                "network_nodes": self.graph.n, "output_body_ids": self.output_body_ids.tolist()}
