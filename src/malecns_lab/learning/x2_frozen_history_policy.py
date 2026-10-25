"""Qualification-only online adapter for the sealed B current/history committee.

Importing this module does not import torch, load a model, or run a brain. The
only inputs to the readout are actual frozen MaleCNS current features. Archived
failures remain failures; this adapter does not confer deployment qualification.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

from ..control.x2_behavior import (Decision, JOINT_READOUT_CLASSES,
    JOINT_READOUT_SCHEMA, digest, require)
from ..control.x2_interaction import frozen_full_interaction_profile
from ..control.x2_skill_profile import frozen_full_skill_profile

SCHEMA = "malecns_lab.x2_frozen_history_policy.v1"
RESEARCH = Path("data/x2_model_redesign_20261024_r1")
CANDIDATE_RELATIVE = RESEARCH / "b_current_history_v1"
SEEDS = (24102431, 24102432, 24102433)
ENGINE_ID = "2ec50952001b75f2f628252d916106bdbddcdd2c2eb8d39834f2181486c790c1"
CHECKPOINT_SHA256 = (
    "a562d8fd7eb1ba21a44f09f95ace64d84b342cf5b8a2ebea277f75bec63efd3a",
    "c31a4ab129aa8a0638d76f087e406b01926f460cbe642406cc4b9ddc4f49769b",
    "649a748da178ec4a2c08fa1d1bcbd5b02aa134934de936442eb810ef9947f292")
NORMALIZER_SHA256 = "bb4b04be16067e313fb456e03108da106ac914cbf10195c386474f0fb3198cb4"
PINNED = {
    "b_current_history_v1/protocol.json": "a50604ddd9481ea234f543f2a600963d8e3e7f4936f9f73cea72f21c2fc266ed",
    "b_current_history_v1/worker.py": "a363343b41199478719ba15ed3f425df171290bcf89e0955553d349ec4f4a365",
    "b_current_history_v1/shared.py": "a2484bc2206e108c2f20a2f2972cd5174200d21b3bb02d170fa071d6bebd3887",
    "b_current_history_v1/execution-r1/result-review.json": "bc7c2f435da6e77e0ca1cc4a37b60c1fb80220389284e7c7cd42576ddc72f05d",
    "request_prefix_cache_v1/protocol.json": "063373e83b85e46c3e1a18caf025d8c218d104f0856a30667a72fdab879c715e",
    "request_prefix_brain_v2/protocol.json": "0cc1279808658a961f0d2abe0a2ef83833d2f29df10f1496e844d747ebf8345b",
}
RUNTIME_SOURCES = ("contract/teaching_v4.py", "a_measured_prefix_data_v1/shared.py",
    "measured_semantic_clock_probe_v1/probe.py", "neural_features_v2.py",
    "frozen_validation_backend_v1.py", "frozen_sequence_engine_v1.py",
    "request_prefix_brain_v2/engine.py")


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _check(path, expected):
    require(Path(path).is_file() and not Path(path).is_symlink()
            and _sha(path) == expected, "Frozen asset changed: " + str(path))


def inspect_frozen_history_candidate(root, path):
    """Authenticate a single fixed candidate without importing model code."""
    root, path = Path(root).resolve(), Path(path).resolve()
    require(path == root / CANDIDATE_RELATIVE, "Only the fixed B candidate is supported")
    artifacts = {str(RESEARCH / name): value for name, value in PINNED.items()}
    for name, expected in artifacts.items():
        _check(root / name, expected)
    cache = _read(root / RESEARCH / "request_prefix_cache_v1/protocol.json")
    origin = _read(root / RESEARCH / "request_prefix_brain_v2/protocol.json")
    sources = {str(RESEARCH / name): cache["source_sha256"][str(RESEARCH / name)]
               for name in RUNTIME_SOURCES}
    for name, expected in sources.items():
        _check(root / name, expected)
    archive = Path(origin["archive_path"])
    for name, expected in origin["archive_assets_sha256"].items():
        artifacts[str(archive / name)] = expected
    # Current numeric core must remain the sealed core, while the new adapter
    # and transport are explicitly different source identities.
    for name in ("brain/mps_trainable.py", "brain/trainable.py", "senses/x2_behavior.py",
                 "learning/x2_behavior_policy.py"):
        expected = origin["archived_source_sha256"][name]
        sources[str(Path("src/malecns_lab") / name)] = expected
    for seed, expected in zip(SEEDS, CHECKPOINT_SHA256):
        folder = CANDIDATE_RELATIVE / "execution-r1" / f"B-seed-{seed}"
        artifacts[str(folder / "checkpoint-600.pt")] = expected
        artifacts[str(folder / "normalizer.npz")] = NORMALIZER_SHA256
        artifacts[str(folder / "normalizer.json")] = "9be452803c71eaa7097a3b4939e2a0ad884cdf5e6063c033e6c3e00617d9caf9"
    for name, expected in {**artifacts, **sources}.items():
        _check(root / name, expected)
    sources[str(Path(__file__).resolve().relative_to(root))] = _sha(__file__)
    input_map = dict(numeric_engine_id=ENGINE_ID, channels=origin["plan"]["channels"],
        numeric_contract=cache["numeric_contract"], input="real observation only; foreign behavior HOLD mask")
    policy = dict(schema=SCHEMA, numeric_engine_id=ENGINE_ID,
        interaction_profile=frozen_full_interaction_profile(),
        execution_profile=frozen_full_skill_profile(), qualification_only=True,
        can_deploy=False, raw_input_to_readout_bypass=False,
        initialization="same frozen 600-step root per request; zero all three GRU histories",
        readout=dict(seeds=list(SEEDS), epoch=600, classes=list(JOINT_READOUT_CLASSES),
            architecture="7110->Linear32/GELU->GRU32; [current32,history32]->Linear32/GELU/Linear9",
            combination="equal mean of three softmax(T=1) probabilities, then argmax",
            normalizer="saved train-only mean/scale; no recalculation", parameter_count_per_member=236265),
        input_map_id=digest(input_map), artifacts=artifacts, source_sha256=sources)
    policy["policy_id"] = digest(policy)
    return dict(schema=SCHEMA, checkpoint_id=digest(dict(artifacts=artifacts,
                policy_id=policy["policy_id"])), policy=policy)


def _pure(path, names):
    """Extract only frozen encoder/clock definitions; never load a teacher/run."""
    tree = ast.parse(Path(path).read_text())
    selected = []
    for node in tree.body:
        name = node.name if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else (
            node.targets[0].id if isinstance(node, ast.Assign)
            and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) else None)
        if name in names:
            selected.append(node)
    require(len(selected) == len(names), "Frozen definition inventory changed")
    namespace = {"math": math}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


def load_input_contract(root):
    """Pure input adapter; callers authenticate source hashes before loading."""
    root = Path(root) / RESEARCH
    encoder = _pure(root / "contract/teaching_v4.py", ("TASKS", "BEHAVIORS", "STATUSES",
        "FRESH_SECONDS", "validate_observation", "observable", "encode_channels"))
    encode = _pure(root / "a_measured_prefix_data_v1/shared.py", ("encoded_channels",)).encoded_channels
    clock = _pure(root / "measured_semantic_clock_probe_v1/probe.py", ("semantic_key",)).semantic_key
    return encoder, encode, clock


@contextmanager
def _research_modules(root):
    """Private temporary aliases needed by the unchanged frozen modules."""
    paths = (("neural_features_v2", "neural_features_v2.py"),
             ("frozen_validation_backend_v1", "frozen_validation_backend_v1.py"),
             ("frozen_sequence_engine_v1", "frozen_sequence_engine_v1.py"),
             ("x2_online_prefix_engine", "request_prefix_brain_v2/engine.py"))
    previous, loaded = {}, {}
    try:
        for name, path in paths:
            previous[name] = sys.modules.get(name)
            spec = importlib.util.spec_from_file_location(name, Path(root) / RESEARCH / path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded[name] = module
        yield loaded
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def readout_class(torch, root):
    # Use the exact frozen class AST, including original parameter names/order.
    tree = ast.parse((Path(root) / CANDIDATE_RELATIVE / "worker.py").read_text())
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
               and node.name == "NeuralCurrentHistory"]
    require(len(matches) == 1, "Frozen readout class missing")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=matches, type_ignores=[]), "<frozen B readout class>", "exec"), namespace)
    return namespace["NeuralCurrentHistory"]


def load_frozen_history_policy(root, path, metadata):
    full = inspect_frozen_history_candidate(root, path)
    require(metadata["checkpoint_id"] == full["checkpoint_id"]
            and all(full["policy"].get(k) == v for k, v in metadata["policy"].items()),
            "Registered frozen candidate identity differs")
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from .x2_behavior_training import inspect_checkpoint
    from .x2_behavior_session import checkpoint_registration_metadata, load_candidate_policy
    root, path = Path(root), Path(path)
    p = _read(root / RESEARCH / "request_prefix_cache_v1/protocol.json")
    origin = _read(root / RESEARCH / "request_prefix_brain_v2/protocol.json")
    archive = root / origin["archive_path"] / "checkpoint"
    old = inspect_checkpoint(archive)
    with torch.no_grad(), _research_modules(root) as modules:
        base = load_candidate_policy(root, checkpoint_registration_metadata(old), archive)
        base.eval().requires_grad_(False)
        reference = modules["neural_features_v2"].NeuralFeatureEngine(
            base, origin["plan"]["channels"], p["numeric_contract"])
        engine = modules["x2_online_prefix_engine"].PrefixEngine(reference, p["numeric_contract"])
        require(engine.identity == p["expected_engine_identity"] and engine.engine_id == ENGINE_ID,
                "Frozen numeric engine identity differs")
        require(reference.identity["complete_graph"] == p["graph_identity"]
                and reference.identity["theta"] == p["theta_identity"], "Frozen graph/theta differs")
    members, mean, scale = load_frozen_readouts(root, path)
    return FrozenHistoryPolicy(engine, members, mean, scale,
        full["policy"], load_input_contract(root))


def load_frozen_readouts(root, path):
    """CPU readouts only, for the fixed bounded cached-prefix reload control."""
    import numpy as np
    import torch
    root, path = Path(root), Path(path)
    require(path.resolve() == (root / CANDIDATE_RELATIVE).resolve(), "Foreign B readout path")
    _check(root / CANDIDATE_RELATIVE / "worker.py", PINNED["b_current_history_v1/worker.py"])
    cls = readout_class(torch, root)
    members, mean, scale = [], None, None
    for seed, checkpoint_hash in zip(SEEDS, CHECKPOINT_SHA256):
        folder = path / "execution-r1" / f"B-seed-{seed}"
        _check(folder / "checkpoint-600.pt", checkpoint_hash)
        _check(folder / "normalizer.npz", NORMALIZER_SHA256)
        # The exact file SHA was authenticated before this trusted local load.
        checkpoint = torch.load(folder / "checkpoint-600.pt", map_location="cpu", weights_only=False)
        identity = checkpoint["identity"]
        require(checkpoint["epoch"] == 600 and identity["seed"] == seed
                and identity["engine_id"] == ENGINE_ID
                and identity["protocol_sha256"] == PINNED["b_current_history_v1/protocol.json"],
                "B checkpoint identity differs")
        with np.load(folder / "normalizer.npz", allow_pickle=False) as saved:
            m, s = saved["mean"].copy(), saved["scale"].copy()
        require(m.shape == s.shape == (7110,) and m.dtype == s.dtype == np.float32
                and np.isfinite(m).all() and np.isfinite(s).all() and (s >= 1e-6).all(),
                "Invalid frozen normalizer")
        require(np.array_equal(m, checkpoint["normalizer"]["mean"])
                and np.array_equal(s, checkpoint["normalizer"]["scale"]), "Checkpoint normalizer differs")
        if mean is not None:
            require(np.array_equal(m, mean) and np.array_equal(s, scale), "Committee normalizers differ")
        mean, scale = m, s
        member = cls()
        member.load_state_dict(checkpoint["model"], strict=True)
        member.eval().requires_grad_(False)
        members.append(member)
    return tuple(members), torch.from_numpy(mean), torch.from_numpy(scale)


@contextmanager
def capture_actual_steps(owner, name, telemetry):
    """Observe the exact spikes returned by this instance; never recompute."""
    if telemetry is None:
        yield
        return
    original = getattr(owner, name)
    had_own = name in vars(owner)
    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        telemetry.capture(result[3], args[1].step)
        if result[0].step % 50 == 0:
            telemetry.flush(result[0].step)
            require(getattr(telemetry, "error", None) is None, "Actual spike telemetry failed")
        return result
    setattr(owner, name, observed)
    try:
        yield
    finally:
        if had_own:
            setattr(owner, name, original)
        else:
            delattr(owner, name)


def _copy_prefix(prefix, *, clone):
    def tensor(value):
        return value.detach().clone() if clone else value.detach()
    n, b = prefix.native, prefix.native.brain
    brain = replace(b, v=tensor(b.v), g=tensor(b.g), refractory_until=tensor(b.refractory_until),
                    pending=tuple(tensor(x) for x in b.pending))
    return replace(prefix, native=replace(n, brain=brain, phase=tensor(n.phase),
                                         event_counts=tensor(n.event_counts)))


@dataclass(frozen=True)
class HistoryState:
    prefix: object
    hidden: tuple
    session_id: str
    request_id: str | None = None
    observation_seq: int = -1
    frame_index: int = 0
    semantic_key: dict | None = None
    event_cursor: int = -1
    features: object = None
    joint: dict | None = None

    @property
    def brain(self):
        return self.prefix.native.brain

    def detach(self):
        return replace(self, prefix=_copy_prefix(self.prefix, clone=False),
            hidden=tuple(None if x is None else x.detach() for x in self.hidden),
            features=None if self.features is None else self.features.detach())


@dataclass(frozen=True)
class HistoryOutput:
    features_mv: object  # Original 7110 signed-current feature, legacy field name.
    state: HistoryState
    joint_readout: dict


class FrozenHistoryPolicy:
    frame_steps, precondition_steps = 500, 600
    telemetry_spike_source = ("actual same-instance PrefixEngine kernel._advance spikes; "
        "precondition from the same base._step; no cache/reconstructed spike stream")

    def __init__(self, engine, members, mean, scale, identity, input_contract):
        self.engine, self.members, self.mean, self.scale = engine, tuple(members), mean, scale
        self.identity, self.policy_id = deepcopy(identity), identity["policy_id"]
        self.brain = engine.reference.base.brain
        self.encoder = SimpleNamespace(input_map_id=identity["input_map_id"])
        self.contract, self.encode, self.clock = input_contract
        self._root, self._context = None, None

    def eval(self):
        for member in self.members:
            member.eval()
        return self

    def set_runtime_context(self, observation_seq, protection_state, event_sequence):
        require(type(observation_seq) is int and observation_seq >= 0
                and protection_state in ("READY", "LATCHED", "RECOVERING")
                and type(event_sequence) is int and event_sequence >= -1, "Invalid actual runtime context")
        self._context = observation_seq, protection_state, event_sequence

    def initial_state(self, session_id, *, telemetry=None):
        import torch
        require(self._root is None and isinstance(session_id, str) and session_id,
                "One live session per frozen policy instance")
        with torch.no_grad(), capture_actual_steps(self.engine.reference.base, "_step", telemetry):
            self._root = self.engine.initial_prefix()
        return HistoryState(_copy_prefix(self._root, clone=True), (None, None, None), session_id)

    def observation_record(self, observation, state):
        require(self._context is not None and self._context[0] == observation.observation_seq,
                "Missing matching actual runtime context")
        _, protection, event = self._context
        require(event >= state.event_cursor, "Actual event cursor went backwards")
        f = observation.feedback
        require(f.valid, "Missing actual body feedback must not consume model history")
        edge = observation.request_id != state.request_id
        require(observation.request_changed == edge, "Request boundary differs from consumed history")
        fresh = f.valid and f.age_s <= .75
        return dict(task=str(observation.task), request_changed=edge,
            capabilities=[str(x) for x in observation.capabilities], protection_state=protection,
            feedback=dict(valid=f.valid, age_s=f.age_s, status=str(f.status), behavior=str(f.behavior),
                same_request=bool(f.request_id and f.request_id == observation.request_id),
                event_new=bool(fresh and event > state.event_cursor), standing=f.standing,
                heading_error_rad=f.heading_error_rad, vx_m_s=f.vx_m_s, wz_rad_s=f.wz_rad_s,
                mc_mode=f.mc_mode if f.mc_mode in ("STAND_DEFAULT", "WALK") else "unknown")), event

    def consume_feature(self, features, hidden):
        """Readout-only entry used by online inference and bounded CPU replay."""
        import torch
        require(features.shape == (7110,) and torch.isfinite(features).all(), "Invalid neural features")
        normalized = (features.detach().cpu().to(torch.float32) - self.mean) / self.scale
        logits, following = [], []
        with torch.no_grad():
            for member, previous in zip(self.members, hidden):
                current = member.project(normalized[None, :])
                history, next_hidden = member.gru(current[:, None, :], previous)
                row = member.head(torch.cat((current, history[:, 0, :]), dim=-1))[0]
                require(torch.isfinite(row).all(), "Nonfinite B readout")
                logits.append(row)
                following.append(next_hidden)
            rows = torch.stack(logits)
            probabilities = torch.softmax(rows, dim=-1).mean(dim=0)
        return rows.tolist(), probabilities.tolist(), tuple(following)

    def __call__(self, observation, state, *, checkpoint_blocks=False, telemetry=None):
        import torch
        require(observation.session_id == state.session_id and observation.observation_seq > state.observation_seq,
                "Foreign, duplicate or reordered observation")
        record, event = self.observation_record(observation, state)
        key, edge = self.clock(record), record["request_changed"]
        advanced = edge or state.semantic_key is None or key != state.semantic_key
        prefix = _copy_prefix(self._root, clone=True) if edge else state.prefix
        hidden = (None, None, None) if edge else state.hidden
        features, joint = state.features, deepcopy(state.joint)
        if advanced:
            channels = self.encode(self.contract, record)
            require(tuple(channels) == tuple(self.engine.channel_names), "Frozen input44 ordering differs")
            start = prefix.native.brain.step
            with torch.no_grad(), capture_actual_steps(self.engine.kernel, "_advance", telemetry):
                features, prefix, _ = self.engine.advance_tokens(prefix, struct.pack("<44f", *channels.values()))
                rows, probabilities, hidden = self.consume_feature(features, hidden)
            selected = max(range(9), key=probabilities.__getitem__)
            joint = dict(schema=JOINT_READOUT_SCHEMA, classes=list(JOINT_READOUT_CLASSES),
                member_logits=rows, probabilities=probabilities, selected_index=selected,
                model_advanced_this_frame=True, source_observation_seq=observation.observation_seq,
                source_request_id=observation.request_id, source_neural_start=start,
                source_neural_end=prefix.native.brain.step)
        else:
            require(joint is not None and features is not None, "No consumed output to hold")
            joint["model_advanced_this_frame"] = False
        cursor = event if record["feedback"]["event_new"] else state.event_cursor
        following = HistoryState(prefix, hidden, state.session_id, observation.request_id,
            observation.observation_seq, state.frame_index + int(advanced), deepcopy(key), cursor, features, joint)
        self._context = None
        return HistoryOutput(features, following, joint)

    def state_arrays(self, state):
        import numpy as np
        import torch
        # Session identity/checkpoint callers are outside inference no_grad.
        with torch.no_grad():
            arrays = self.engine.state_arrays(state.prefix.native)
        for i, hidden in enumerate(state.hidden):
            arrays[f"readout.hidden.{i}"] = (np.zeros((1, 1, 32), dtype=np.float32)
                if hidden is None else hidden.detach().cpu().numpy().copy())
        if state.features is not None:
            arrays["readout.last_features"] = state.features.detach().cpu().numpy().copy()
        return arrays

    def state_identity(self, state, arrays=None):
        import numpy as np
        arrays = self.state_arrays(state) if arrays is None else arrays
        identities = {}
        for name, value in arrays.items():
            value = np.ascontiguousarray(value)
            identities[name] = dict(shape=list(value.shape), dtype=value.dtype.str,
                sha256=hashlib.sha256(value.tobytes()).hexdigest())
        payload = dict(schema=SCHEMA + ".state", policy_id=self.policy_id, session_id=state.session_id,
            request_id=state.request_id, observation_seq=state.observation_seq, frame_index=state.frame_index,
            neural_step=state.brain.step, prefix_node=state.prefix.node_id,
            semantic_key=state.semantic_key, consumed_event_cursor=state.event_cursor,
            held_readout=state.joint, arrays=identities)
        return dict(payload, state_id=digest(payload))

    def decision(self, observation, output, *, checkpoint_id, state_before_id, decision_seq):
        joint = output.joint_readout
        behavior, phase = JOINT_READOUT_CLASSES[joint["selected_index"]].split("/")
        return Decision(observation.run_id, observation.session_id, decision_seq, observation.request_id,
            observation.observation_seq, state_before_id, self.state_identity(output.state)["state_id"],
            (), (), (0.,), behavior, phase, .5, self.policy_id, checkpoint_id, self.encoder.input_map_id,
            joint["source_neural_start"], joint["source_neural_end"], joint_readout=joint)
