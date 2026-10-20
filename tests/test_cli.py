import pytest

from malecns_lab.cli import build_parser, main


def test_cli_bench_uses_real_graph(capsys, malecns_path):
    assert main(["bench", "--ms", "5"]) == 0
    output = capsys.readouterr().out
    assert "brain time" in output
    assert "166,700" in output


def test_cli_help_lists_offline_tools(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "malecns_lab" in output
    assert "{download,build-brain,inspect,calibrate,bench}" in output


@pytest.mark.parametrize("command", ["demo", "swarm", "fly"])
def test_cli_rejects_removed_legacy_commands(command):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args([command])
    assert exc.value.code == 2
