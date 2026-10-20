"""Torch-free identity construction shared by registration and actual MPS objects."""
from copy import deepcopy
from pathlib import Path
from .metadata import canonical_hash, file_hashes

ACTIVATION_CHECKPOINT = {
    "strategy": "fixed_neural_step_blocks", "neural_steps_per_block": 50,
    "use_reentrant": False, "preserve_rng_state": False, "determinism_check": "default", "early_stop": True,
    "sensory_replay": "immutable block snapshot; private SpatialDrive; original float64 phase/events",
    "gradient_boundary": "no detach between blocks or frames; full v/g/pending/refractory",
    "dn_accumulation": "original per-step addition order; running sums passed between blocks",
}


def model_identity(graph, config, inputs, tonic, reference):
    value = {"schema": "cnsbridge.mps_lamina_lif.v1", "graph": deepcopy(graph), "config": deepcopy(config),
             "input_body_ids": list(inputs), "dtype": "torch.float32", "backend": "torch_mps_coo_forward_adjoint",
             "gain": "1 + 0.5*tanh(theta[type(pre)]); strict float32 bounds",
             "gradient": "fast_sigmoid; reset/refractory gates detached; all-cell sparse adjoint",
             "event_order": "arrivals; exact active integration; explicit voltage input; strict threshold; reset; delayed enqueue",
             "reference": reference,
             "tonic": {**{k: deepcopy(tonic[k]) for k in ("mode", "types", "members", "member_count", "mapping_sha256", "initial_offset_mv")},
                       "formula": "16*sigmoid(rho[type]); rho=0; nonmembers=0"},
             "implementation_sha256": {name: file_hashes(Path(__file__).with_name(name))["sha256"]
                 for name in ("mps_trainable.py", "mps_identity.py", "trainable.py", "trainable_graph.py", "tonic.py")}}
    return {**value, "model_id": canonical_hash(value)}


def policy_identity(cpu):
    model = cpu["model"]
    value = deepcopy(cpu)
    value.pop("policy_id")
    value.update(schema="cnsbridge.mps_neural_policy.v1", backend="mps_forward_backward_cpu_adam",
                 cpu_source_policy_id=cpu["policy_id"],
                 activation_checkpoint=deepcopy(ACTIVATION_CHECKPOINT),
                 model=model_identity(model["graph"], model["config"], model["input_body_ids"], model["tonic"], model["reference"]))
    package = Path(__file__).resolve().parents[1]
    value["implementation_sha256"].update({name: file_hashes(package / name)["sha256"]
        for name in ("brain/mps_trainable.py", "brain/mps_identity.py", "learning/mps_policy.py", "learning/mps_recompute.py")})
    return {**value, "policy_id": canonical_hash(value)}
