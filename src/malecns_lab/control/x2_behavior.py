"""Versioned X2 behavior messages and admission; no brain or robot imports.

A refusal never edits the decoded neural output. Emergency protection is an
execution event, distinct from an ordinary END_INTERACTION task or HOLD output.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import math

SCHEMA = "malecns_lab.x2_behavior.v1"


class StrEnum(str, Enum):
    """Python 3.10 ROS-compatible string enum."""
    def __str__(self):
        return self.value


class Task(StrEnum):
    GREET = "GREET"
    FACE_LEFT = "FACE_LEFT"
    FACE_RIGHT = "FACE_RIGHT"
    SHOW_FORWARD = "SHOW_FORWARD"
    END_INTERACTION = "END_INTERACTION"


class Behavior(StrEnum):
    HOLD = "HOLD"
    ADVANCE = "ADVANCE"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    WAVE_RIGHT = "WAVE_RIGHT"


class Phase(StrEnum):
    START = "START"
    CONTINUE = "CONTINUE"
    FINISH = "FINISH"


class ExecutionStatus(StrEnum):
    IDLE = "IDLE"
    ACCEPTED = "ACCEPTED"
    STARTED = "STARTED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


BEHAVIORS = tuple(Behavior)
PHASES = tuple(Phase)
ACTIVE = frozenset((ExecutionStatus.ACCEPTED, ExecutionStatus.STARTED, ExecutionStatus.RUNNING))
TERMINAL = frozenset((ExecutionStatus.COMPLETED, ExecutionStatus.CANCELLED, ExecutionStatus.REJECTED,
                      ExecutionStatus.FAILED, ExecutionStatus.UNKNOWN))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def identifier(value):
    return isinstance(value, str) and 0 < len(value) <= 256


class Message:
    def to_dict(self):
        return {"schema": SCHEMA, "kind": type(self).__name__, **asdict(self)}

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        require(value.pop("schema", None) == SCHEMA and value.pop("kind", None) == cls.__name__, "Message schema/kind differs")
        if cls is Observation:
            value["feedback"] = Feedback(**value["feedback"])
            value["perception"] = Perception(**value["perception"])
        return cls(**value)


@dataclass(frozen=True)
class Perception:
    """Deployable visual measurement, not privileged target/teacher truth."""
    valid: bool = False
    age_s: float = 0.
    source: str = "missing"
    # 4 x 4 row-major measured occupancy, explicitly absent when invalid.
    visual_cells: tuple[float, ...] = (0.,) * 16

    def __post_init__(self):
        object.__setattr__(self, "visual_cells", tuple(self.visual_cells))
        require(type(self.valid) is bool and finite(self.age_s) and self.age_s >= 0 and identifier(self.source), "Invalid perception metadata")
        require(len(self.visual_cells) == 16 and all(finite(x) and 0 <= x <= 1 for x in self.visual_cells), "Invalid visual cells")
        require(self.valid or not any(self.visual_cells), "Missing perception must not carry visual values")
        require(not self.valid or self.source != "missing", "Valid perception needs actual source")


@dataclass(frozen=True)
class Feedback:
    """Observed executor/body facts. A service ACK is only ACCEPTED."""
    valid: bool = False
    age_s: float = 0.
    source: str = "missing"
    status: ExecutionStatus = ExecutionStatus.IDLE
    behavior: Behavior = Behavior.HOLD
    execution_id: str = ""
    request_id: str = ""
    standing: bool = False
    yaw_rad: float = 0.
    vx_m_s: float = 0.
    wz_rad_s: float = 0.
    # Relative heading supplied by an explicit interaction target/measurement,
    # not a ground-truth task-progress flag or future completion label.
    heading_error_rad: float = 0.
    mc_mode: str = "unknown"

    def __post_init__(self):
        object.__setattr__(self, "status", ExecutionStatus(self.status))
        object.__setattr__(self, "behavior", Behavior(self.behavior))
        require(type(self.valid) is bool and type(self.standing) is bool and identifier(self.source), "Invalid feedback metadata")
        require(all(finite(x) for x in (self.age_s, self.yaw_rad, self.vx_m_s, self.wz_rad_s, self.heading_error_rad)) and self.age_s >= 0, "Invalid feedback measurement")
        require(not self.valid or self.source != "missing", "Valid feedback needs actual source")
        require(self.status == ExecutionStatus.IDLE or (identifier(self.execution_id) and identifier(self.request_id)), "Execution feedback needs attribution")
        require(identifier(self.mc_mode), "Missing MC mode")


@dataclass(frozen=True)
class Observation(Message):
    run_id: str
    session_id: str
    observation_seq: int
    request_id: str
    task: Task
    received_monotonic: float
    feedback: Feedback = field(default_factory=Feedback)
    perception: Perception = field(default_factory=Perception)
    capabilities: tuple[Behavior, ...] = (Behavior.HOLD,)
    request_changed: bool = False

    def __post_init__(self):
        object.__setattr__(self, "task", Task(self.task))
        object.__setattr__(self, "capabilities", tuple(Behavior(x) for x in self.capabilities))
        require(all(identifier(x) for x in (self.run_id, self.session_id, self.request_id)), "Invalid observation identity")
        require(type(self.observation_seq) is int and self.observation_seq >= 0 and finite(self.received_monotonic) and self.received_monotonic >= 0, "Invalid observation clock")
        require(isinstance(self.feedback, Feedback) and isinstance(self.perception, Perception) and type(self.request_changed) is bool, "Invalid observation fields")
        require(Behavior.HOLD in self.capabilities and len(set(self.capabilities)) == len(self.capabilities), "Capabilities must be unique and include HOLD")


@dataclass(frozen=True)
class Decision(Message):
    run_id: str
    session_id: str
    decision_seq: int
    request_id: str
    observation_seq: int
    state_before_id: str
    state_after_id: str
    behavior_logits: tuple[float, ...]
    phase_logits: tuple[float, ...]
    parameter_raw: tuple[float, ...]
    behavior: Behavior
    phase: Phase
    intensity: float
    policy_id: str
    checkpoint_id: str
    input_map_id: str
    neural_start_step: int
    neural_end_step: int

    def __post_init__(self):
        object.__setattr__(self, "behavior", Behavior(self.behavior))
        object.__setattr__(self, "phase", Phase(self.phase))
        for key in ("behavior_logits", "phase_logits", "parameter_raw"):
            object.__setattr__(self, key, tuple(getattr(self, key)))
        require(all(identifier(x) for x in (self.run_id, self.session_id, self.request_id, self.state_before_id,
                self.state_after_id, self.policy_id, self.checkpoint_id, self.input_map_id)), "Invalid decision identity")
        require(all(type(x) is int and x >= 0 for x in (self.decision_seq, self.observation_seq, self.neural_start_step))
                and type(self.neural_end_step) is int and self.neural_end_step > self.neural_start_step, "Invalid decision clock")
        require(len(self.behavior_logits) == 5 and len(self.phase_logits) == 3 and len(self.parameter_raw) == 1
                and all(finite(x) for x in self.behavior_logits + self.phase_logits + self.parameter_raw), "Invalid raw heads")
        require(self.behavior == BEHAVIORS[max(range(5), key=self.behavior_logits.__getitem__)]
                and self.phase == PHASES[max(range(3), key=self.phase_logits.__getitem__)], "Decoded class differs from raw logits")
        # Stable sigmoid; same contract used by a device readout.
        x = self.parameter_raw[0]
        expected = 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))
        require(finite(self.intensity) and 0 <= self.intensity <= 1 and abs(self.intensity-expected) <= 1e-6, "Decoded intensity differs from raw parameter")


@dataclass(frozen=True)
class ExecutionEvent(Message):
    run_id: str
    session_id: str
    event_seq: int
    decision_seq: int
    execution_id: str
    request_id: str
    backend: str
    skill_pack_id: str
    status: ExecutionStatus
    behavior: Behavior
    received_monotonic: float
    mc_request: dict = field(default_factory=dict)
    raw_feedback: dict = field(default_factory=dict)
    stop_evidence: dict = field(default_factory=dict)
    protection_reason: str | None = None
    # Facts sharing a lifecycle state remain distinct: local admission is not
    # an official preset ACK, and requesting FINISH is not measured motion.
    fact: str = "status"
    finish_decision_seq: int | None = None
    finish_request_id: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "status", ExecutionStatus(self.status))
        object.__setattr__(self, "behavior", Behavior(self.behavior))
        require(all(identifier(x) for x in (self.run_id, self.session_id, self.execution_id, self.request_id, self.skill_pack_id)), "Invalid event identity")
        require(self.backend in ("sim", "robot", "offline_fixture"), "Invalid execution backend")
        require(all(type(x) is int and x >= 0 for x in (self.event_seq, self.decision_seq)) and finite(self.received_monotonic) and self.received_monotonic >= 0, "Invalid event clock")
        require(all(isinstance(x, dict) for x in (self.mc_request, self.raw_feedback, self.stop_evidence)), "Event evidence must be structured")
        require(self.protection_reason is None or identifier(self.protection_reason), "Invalid protection reason")
        require(self.fact in ("status", "skill_admitted", "preset_ack", "finish_requested"), "Invalid execution fact")
        require((self.finish_decision_seq is None and self.finish_request_id is None) or
                (type(self.finish_decision_seq) is int and self.finish_decision_seq >= 0 and identifier(self.finish_request_id)), "Invalid finish attribution")
        digest(self.to_dict())  # Reject non-serializable or nonfinite nested evidence.


@dataclass(frozen=True)
class Admission:
    accepted: bool
    reason: str
    decision: Decision
    active_execution_id: str | None


class DecisionGate:
    """Host-side identity/freshness/serial-skill gate, with no task-to-action rule.

    observe() is called on receipt; the model cannot renew observation time.
    An admitted START reserves the slot until its attributed execution event.
    No method emits an MC command. Safety protection belongs to the executor.
    """
    def __init__(self, run_id, session_id, policy_id, checkpoint_id, input_map_id, *, max_age_s=30.):
        require(all(identifier(x) for x in (run_id, session_id, policy_id, checkpoint_id, input_map_id)), "Invalid gate identity")
        require(finite(max_age_s) and max_age_s > 0, "Invalid decision age budget")
        self.run_id, self.session_id = run_id, session_id
        self.identities = (policy_id, checkpoint_id, input_map_id)
        self.max_age_s = max_age_s
        self.observation = None
        self.last_decision_seq = -1
        self.last_event_seq = -1
        self.active = None
        self._admitted = {}
        self._used_executions = set()
        self._event_time = -1.
        self.blocked_reason = None

    def observe(self, observation):
        require(isinstance(observation, Observation) and (observation.run_id, observation.session_id) == (self.run_id, self.session_id), "Cross-run observation")
        if self.observation is not None:
            require(observation.observation_seq > self.observation.observation_seq
                    and observation.received_monotonic >= self.observation.received_monotonic, "Reordered observation")
            require(observation.request_id != self.observation.request_id or observation.task == self.observation.task, "Task changed without new request")
        self.observation = observation

    def admit(self, decision, now):
        require(isinstance(decision, Decision) and finite(now), "Invalid decision admission")
        reason = self._reason(decision, now)
        if reason == "accepted":
            self.last_decision_seq = decision.decision_seq
            origin = self.active["decision"].decision_seq if self.active else decision.decision_seq
            self._admitted[decision.decision_seq] = (decision, origin)
            if self.active is None:
                self.active = {"decision": decision, "execution_id": None, "status": None,
                               "preset_ack": False, "finish_requests": set()}
        return Admission(reason == "accepted", reason, decision, self.active["execution_id"] if self.active else None)

    def _reason(self, d, now):
        if (d.run_id, d.session_id) != (self.run_id, self.session_id): return "cross_run"
        if (d.policy_id, d.checkpoint_id, d.input_map_id) != self.identities: return "policy_identity"
        if d.decision_seq <= self.last_decision_seq: return "duplicate_or_reordered_decision"
        o = self.observation
        if o is None or d.observation_seq != o.observation_seq: return "superseded_observation"
        if d.request_id != o.request_id: return "superseded_request"
        if not 0 <= now-o.received_monotonic <= self.max_age_s: return "expired_observation"
        if self.blocked_reason is not None: return "execution_protection_latched"
        if d.behavior not in o.capabilities: return "unavailable_skill"
        if d.behavior == Behavior.HOLD:
            if d.phase != Phase.FINISH: return "invalid_hold_phase"
            # HOLD/FINISH requests safe completion/cancellation of an in-flight
            # behavior; it does not declare that robot motion has already ended.
            return "accepted"
        if d.phase == Phase.START:
            return "skill_already_active" if self.active else "accepted"
        if self.active is None: return "no_active_skill"
        if self.active["decision"].behavior != d.behavior: return "active_skill_mismatch"
        if self.active["execution_id"] is None: return "awaiting_execution_acceptance"
        return "accepted"

    def execution_event(self, event):
        require(isinstance(event, ExecutionEvent) and (event.run_id, event.session_id) == (self.run_id, self.session_id), "Cross-run event")
        require(event.event_seq > self.last_event_seq and event.received_monotonic >= self._event_time, "Duplicate/reordered event")
        admitted = self._admitted.get(event.decision_seq)
        require(admitted is not None, "Unattributed execution event")
        d, origin = admitted
        require(d.request_id == event.request_id, "Unattributed execution request")
        require(event.behavior == d.behavior or (d.behavior == Behavior.HOLD and self.active is not None
                and event.behavior == self.active["decision"].behavior), "Execution behavior differs")
        if event.finish_decision_seq is not None:
            finish = self._admitted.get(event.finish_decision_seq)
            require(finish is not None and finish[1] == origin and finish[0].phase == Phase.FINISH
                    and finish[0].request_id == event.finish_request_id, "Unattributed finish decision")
        if self.active is not None:
            require(origin == self.active["decision"].decision_seq, "Event belongs to another active decision")
            current = self.active["execution_id"]
            require(current in (None, event.execution_id), "Concurrent execution event")
            require(current is not None or event.execution_id not in self._used_executions, "Reused execution ID")
            old = self.active["status"]
            transitions = {None: {ExecutionStatus.ACCEPTED, ExecutionStatus.REJECTED, ExecutionStatus.FAILED, ExecutionStatus.UNKNOWN},
                ExecutionStatus.ACCEPTED: {ExecutionStatus.STARTED, ExecutionStatus.CANCELLED, ExecutionStatus.REJECTED, ExecutionStatus.FAILED, ExecutionStatus.UNKNOWN},
                ExecutionStatus.STARTED: {ExecutionStatus.RUNNING, *TERMINAL},
                ExecutionStatus.RUNNING: {ExecutionStatus.RUNNING, *TERMINAL},
                ExecutionStatus.UNKNOWN: {ExecutionStatus.UNKNOWN, ExecutionStatus.FAILED}}
            if event.fact == "preset_ack":
                require(d.behavior == Behavior.WAVE_RIGHT and old == event.status == ExecutionStatus.ACCEPTED
                        and not self.active["preset_ack"], "Duplicate or invalid official preset ACK")
                require(type(event.raw_feedback.get("task_id")) is int and event.raw_feedback["task_id"] > 0,
                        "Official preset ACK needs task attribution")
            elif event.fact == "finish_requested":
                require(event.finish_decision_seq is not None and old == event.status
                        and event.finish_decision_seq not in self.active["finish_requests"], "Duplicate or invalid finish fact")
            else:
                require(event.fact != "skill_admitted" or (old is None and event.status == ExecutionStatus.ACCEPTED), "Invalid local admission fact")
                require(event.status in transitions.get(old, set()), "Invalid execution transition; ACK is not completion")
            if current is None:
                self._used_executions.add(event.execution_id)
            self.active["execution_id"] = event.execution_id
            if event.fact == "preset_ack":
                self.active["preset_ack"] = True
            if event.fact == "finish_requested":
                self.active["finish_requests"].add(event.finish_decision_seq)
            if event.status in (ExecutionStatus.UNKNOWN, ExecutionStatus.FAILED) or event.protection_reason:
                self.blocked_reason = event.protection_reason or str(event.status)
            if event.status in TERMINAL and self.blocked_reason is None:
                self.active = None
            else:
                self.active["status"] = event.status
        else:
            require(origin is None and d.behavior == Behavior.HOLD and event.status in TERMINAL, "Event without active skill")
            require(event.execution_id not in self._used_executions, "Reused execution ID")
            self._used_executions.add(event.execution_id)
        self.last_event_seq, self._event_time = event.event_seq, event.received_monotonic
