"""Shared observation semantics; no teacher, action selection, brain, or device IO."""
from copy import deepcopy
import math

from .x2_behavior import Task, digest, finite, require
from .x2_skill_profile import (candidate_skill_profile, SIM_FIXED_ADVANCE_PROFILE_ID,
    frozen_full_skill_profile)

HEADING_OFFSETS = {"FACE_LEFT": .45, "FACE_RIGHT": -.45}


def interaction_profile(*, wave_enabled=False):
    payload = {"schema": "malecns_lab.x2_interaction_profile.v1",
        "skill_profile_id": candidate_skill_profile(wave_enabled=wave_enabled)["profile_id"],
        "capabilities": ["HOLD", "ADVANCE", "TURN_LEFT", "TURN_RIGHT"],
        "wave_enabled": False, "relative_heading_offsets_rad": dict(HEADING_OFFSETS),
        "other_task_heading_error_rad": 0.,
        "state_reset": "explicit session start only; never reset on request change",
        "sampling": "inference followed by gate-only execution polling until completion or a new request",
        "qualification_only": True}
    if wave_enabled:
        payload.update(wave_enabled=True, capabilities=[*payload["capabilities"], "WAVE_RIGHT"],
                       wave_contract=candidate_skill_profile(wave_enabled=True)["wave_contract"])
    return {**payload, "profile_id": digest(payload)}


def validate_interaction_profile(value):
    require(value in (interaction_profile(), interaction_profile(wave_enabled=True), fixed_advance_interaction_profile(), frozen_history_interaction_profile(), frozen_full_interaction_profile()), "Interaction profile differs from the deployed simulation profile")
    return deepcopy(value)


def fixed_advance_interaction_profile():
    """Keep every sensory/session field; change only the execution binding."""
    payload = interaction_profile()
    payload.pop("profile_id")
    payload["skill_profile_id"] = SIM_FIXED_ADVANCE_PROFILE_ID
    return {**payload, "profile_id": digest(payload)}


def frozen_history_interaction_profile():
    """Fixed SIM execution with the trained B request/semantic-clock contract."""
    payload = fixed_advance_interaction_profile()
    payload.pop("profile_id")
    payload.update(state_reset="each new request restores the same frozen 600-step initial brain and zero readout history",
        sampling="consume first request input and semantic-clock changes; otherwise hold the previous neural readout without advancing brain or GRU")
    return {**payload, "profile_id": digest(payload)}


def frozen_full_interaction_profile():
    """Same frozen B sensory/history rules with real full-action capability."""
    payload = frozen_history_interaction_profile()
    payload.pop('profile_id')
    execution = frozen_full_skill_profile()
    payload.update(skill_profile_id=execution['profile_id'], wave_enabled=True,
        capabilities=['HOLD', 'ADVANCE', 'TURN_LEFT', 'TURN_RIGHT', 'WAVE_RIGHT'],
        wave_contract=execution['wave_contract'],
        request_limits=dict(max_motion_starts=4, max_decisions=32))
    return {**payload, 'profile_id': digest(payload)}


def profile_capabilities(profile):
    return tuple(validate_interaction_profile(profile)["capabilities"])


def wrap_heading(value):
    require(finite(value), "Finite measured heading required")
    return math.atan2(math.sin(value), math.cos(value))


def relative_heading_goal(task, origin_yaw_rad):
    """Bind a relative request to its measured start; never choose an action."""
    task = Task(task)
    require(finite(origin_yaw_rad), "Finite measured heading origin required")
    if task not in HEADING_OFFSETS:
        return None
    offset = HEADING_OFFSETS[task]
    return {"relative_offset_rad": offset, "origin_yaw_rad": origin_yaw_rad,
            "target_yaw_rad": wrap_heading(origin_yaw_rad + offset)}


def relative_heading_error(task, goal, current_yaw_rad):
    task = Task(task)
    require(finite(current_yaw_rad), "Finite current heading required")
    if task not in HEADING_OFFSETS:
        require(goal is None, "Non-heading task must not carry a heading goal")
        return 0.
    require(isinstance(goal, dict), "Relative heading goal required")
    expected = relative_heading_goal(task, goal.get("origin_yaw_rad"))
    require(all(finite(goal.get(key)) and abs(goal[key]-value) <= 1e-8
                for key, value in expected.items()), "Relative heading goal changed")
    return wrap_heading(goal["target_yaw_rad"] - current_yaw_rad)
