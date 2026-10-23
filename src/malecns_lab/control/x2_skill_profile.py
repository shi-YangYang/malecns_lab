"""Fixed simulation qualification profile, independent of brain/task selection."""
import json

from .x2_skills import SKILL_PACK_ID, validate_command_durations

PROFILE_ID = "malecns.x2.sim.advance1-turn2.v1"


def candidate_skill_profile():
    # Single intensity=1 probes only; this is not a calibration of other
    # intensities or evidence of reaching the relative heading target.
    return dict(profile_id=PROFILE_ID, skill_pack_id=SKILL_PACK_ID,
        command_seconds_by_behavior=dict(ADVANCE=1., TURN_LEFT=2., TURN_RIGHT=2.),
        hold_mode="standing_confirmation", wave_enabled=False, qualification_only=True)


def validate_candidate_skill_profile(value):
    """Reject any different registration before transport can publish commands."""
    if not isinstance(value, dict):
        raise ValueError("Missing candidate skill profile")
    validate_command_durations(value.get("command_seconds_by_behavior"))
    expected = candidate_skill_profile()
    if json.dumps(value, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError("Candidate skill profile differs from the fixed simulation configuration")
    return expected
