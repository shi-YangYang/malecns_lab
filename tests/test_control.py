import json

from malecns_lab.config import load_config
from malecns_lab.control import FlightCommand, MotorDecoder
from malecns_lab.control.safety import SafetyGovernor, Telemetry


def test_decoder_settles_then_decodes():
    cfg = load_config()
    dec = MotorDecoder(cfg)
    rest = {"DNg02_L": 35, "DNg02_R": 35, "DNp03_L": 0, "DNp03_R": 0, "DNp01_L": 0, "DNp01_R": 0}
    for _ in range(40):
        c = dec.update(rest, 0.05)
    assert abs(c.throttle) < 0.05
    for _ in range(20):
        c = dec.update({**rest, "DNg02_L": 70, "DNg02_R": 70}, 0.05)
    assert c.throttle > 0.3
    for _ in range(20):
        c = dec.update({**rest, "DNg02_R": 60, "DNg02_L": 20}, 0.05)
    assert c.yaw > 0.3


def test_escape_reflex():
    cfg = load_config()
    dec = MotorDecoder(cfg)
    rest = {"DNg02_L": 35, "DNg02_R": 35, "DNp01_L": 0, "DNp01_R": 0}
    for _ in range(40):
        dec.update(rest, 0.05)
    c = None
    for _ in range(4):
        c = dec.update({**rest, "DNp01_L": 80}, 0.05)
    assert c.escape and c.throttle > 0


def test_readout_file(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"baseline": {"A": 10}, "axes": {"throttle": {"gain": 1.0, "terms": {"A": 0.1}}}}))
    cfg = load_config(overrides={"decoder": {"readout_file": str(p), "smoothing": 1.0}})
    dec = MotorDecoder(cfg)
    c = dec.update({"A": 15}, 0.05)
    assert abs(c.throttle - 0.5) < 1e-6


def test_safety_limits():
    cfg = load_config()
    s = SafetyGovernor(cfg)
    c = s.filter(FlightCommand(throttle=1.0, yaw=1.0), Telemetry(t=0, alt_m=1.0, flying=True), dt=10)
    assert c.throttle <= cfg["safety"]["max_throttle"] + 1e-9
    assert c.yaw <= cfg["safety"]["max_yaw"] + 1e-9
    c = s.filter(FlightCommand(throttle=0.5), Telemetry(t=1, alt_m=5.0, flying=True), dt=10)
    assert c.throttle < 0
    c = s.filter(FlightCommand(throttle=-0.5), Telemetry(t=2, alt_m=0.1, flying=True), dt=10)
    assert c.throttle == 0
    s.filter(FlightCommand(), Telemetry(t=3, alt_m=1.0, battery_pct=5, flying=True), dt=0.05)
    assert s.land_requested


def test_slew_rate():
    s = SafetyGovernor(load_config())
    c = s.filter(FlightCommand(yaw=0.6), Telemetry(t=0, alt_m=1, flying=True), dt=0.05)
    assert c.yaw <= 2.5 * 0.05 + 1e-9


def test_brain_timeout_hovers():
    s = SafetyGovernor(load_config())
    c = s.filter(FlightCommand(throttle=0.5), Telemetry(t=0, alt_m=1, flying=True), dt=0.05, brain_age_s=2.0)
    assert c.throttle == 0 and "timeout" in c.note
