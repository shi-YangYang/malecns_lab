"""Regression for the real startup tick payload; no ROS or simulation IO."""
import io
import json

from malecns_lab.control.x2_skills import SkillExecutor
from malecns_lab.experiments.x2 import _EvidenceLog
from malecns_lab.experiments.x2_skills import _event_record


def test_real_tick_retains_source_time_and_recorder_receipt_time():
    commands = []
    core = SkillExecutor(commands.append, lambda command: None, clock=lambda: 3.)
    core.motion.arm()
    payload = core.tick()
    stream = io.StringIO()
    evidence = _EvidenceLog(stream)

    event = _event_record("tick", "startup", payload,
                          received_monotonic=3.125, wall_time=1000.)
    assert evidence.record(event)
    recorded = json.loads(stream.getvalue())
    assert recorded["monotonic"] == payload["monotonic"] == 3.
    assert recorded["received_monotonic"] == 3.125
    assert recorded["wall_time"] == 1000.
    assert recorded["kind"] == "tick"
    assert recorded["phase"] == "startup"
    assert {key: recorded[key] for key in payload} == json.loads(json.dumps(payload))
    assert "received_monotonic" not in payload
    assert core.command is None
    assert core.events == []
    assert commands == [(0., 0., 0.)]
