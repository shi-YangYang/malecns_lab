import pytest

from cnsbridge.brain import Brain, build_minifly
from cnsbridge.calibrate import fit_readout, record_responses
from cnsbridge.cli import build_parser, main
from cnsbridge.config import load_config


def test_calibration_recovers_signs():
    cfg = load_config()
    b = Brain(build_minifly(), cfg)
    outs, X, Y = record_responses(b, cfg, settle_ms=200, measure_ms=600, repeats=1)
    res = fit_readout(outs, X, Y)
    yaw = res["axes"]["yaw"]["terms"]
    thr = res["axes"]["throttle"]["terms"]
    assert yaw["DNg02_R"] > 0 > yaw["DNg02_L"]
    assert thr["DNg02_L"] + thr["DNg02_R"] > 0


def test_cli_bench(capsys):
    assert main(["bench", "--ms", "200"]) == 0
    assert "brain time" in capsys.readouterr().out


def test_cli_help_lists_offline_tools(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "cnsbridge" in output
    assert "{download,build-brain,inspect,calibrate,bench}" in output


@pytest.mark.parametrize("command", ["demo", "swarm", "fly"])
def test_cli_rejects_removed_legacy_commands(command):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args([command])
    assert exc.value.code == 2
