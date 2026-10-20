"""Real child-process orchestration with an explicitly injected five-node fixture.

Only fullgraph/old-parent admission is replaced. Production job claiming, locks,
resource guards, GPU calculations, archive readers and evidence seals run intact.
"""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from malecns_lab.learning import train_gpu
from malecns_lab.learning.resource_guard import supervise as actual_supervise
from malecns_lab.learning.training_contract import PARAMETERS, read_json, read_arrays
from malecns_lab.learning.gpu_evidence import read_genesis, read_gate, scan_run
from malecns_lab.learning.train_snn import workspace_lock
from malecns_lab.brain.metadata import canonical_hash
from test_gpu_recovery import make_policy, run_identity


def dataset():
    from test_neural_policy import frame
    return SimpleNamespace(sequences=[SimpleNamespace(scene={"scene_id": "fixture"},
        frames=np.stack([frame(red=(i % 2 == 0)) for i in range(320)]),
        time_s=np.arange(320, dtype=np.float64) * .1, valid=np.ones(320, bool),
        teacher_target=np.tile(np.array([[-.2, -.1, .3], [-.1, .1, -.3]], np.float64), (160, 1)))])


def test_initialize_gate_two_new_process_windows_and_terminal_tampering(tmp_path, monkeypatch):
    if not torch.backends.mps.is_available():
        pytest.skip("Actual MPS required")
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    torch.set_num_threads(4)
    with workspace_lock():
        policy, cpu, _ = make_policy(frame_ms=4)
        with torch.no_grad():
            policy.readout.bias[0] = 10
        params = {k: dict(policy.named_parameters())[n].detach().cpu().numpy().copy() for k, n in PARAMETERS.items()}
        run = run_identity(policy)
    run.update(schema="explicit_numerical_process_fixture", scenes=[{"scene_id": "fixture", "frame_count": 320}],
               graph={"path": "explicit-five-node-fixture-only"},
               parent={"checkpoint_id": "1" * 64, "prefix_sha256": "2" * 64})
    parent = {"checkpoint_id": "1" * 64, "parameters": params}
    inputs = tmp_path / "injected-fixture"
    inputs.mkdir()
    (inputs / "run.json").write_text(json.dumps(run))
    (inputs / "cpu.json").write_text(json.dumps(cpu.identity))
    np.savez(inputs / "parent.npz", **params)
    del policy, cpu
    data = dataset()
    monkeypatch.setattr(train_gpu, "register_run", lambda *_: run)
    monkeypatch.setattr(train_gpu, "validate_run", lambda _: (data, parent, read_json(inputs / "cpu.json")))
    # Child patching is confined to this explicit test-generated script, never a
    # selectable production fallback. The actual worker still claims its job.
    script = tmp_path / "fixture_child.py"
    script.write_text('''import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from malecns_lab.learning.training_contract import read_json, read_arrays
import malecns_lab.learning.gpu_contract as contract
import malecns_lab.learning.gpu_policy as policies
from test_gpu_recovery_orchestration import dataset
from test_gpu_recovery import make_policy
root=Path(sys.argv[1])
run=read_json(root/'run.json')
parent={'checkpoint_id':'1'*64,'parameters':read_arrays(root/'parent.npz')}
contract.validate_run=lambda value: (dataset(),parent,read_json(root/'cpu.json'))
def factory(path,cpu,*,credit_horizon):
    assert path=='explicit-five-node-fixture-only'
    return make_policy(credit_horizon,frame_ms=4)[0]
policies.GPUPolicy.from_reference=staticmethod(factory)
from malecns_lab.learning.gpu_worker import run_worker
result=run_worker(Path(sys.argv[2]))
raise SystemExit(0 if result['status']=='completed' else 1)
''')
    env_path = os.pathsep.join((str(Path.cwd() / "src"), str(Path.cwd() / "tests")))
    monkeypatch.setenv("PYTHONPATH", env_path)
    def supervise(command, directory, **kwargs):
        return actual_supervise([sys.executable, str(script), str(inputs), str(directory)], directory, **kwargs)
    monkeypatch.setattr(train_gpu, "supervise", supervise)
    output = tmp_path / "run"
    created = train_gpu.run_job(directory=output, kind="initialize", parent_run="explicit-fixture", credit_horizon=100, theta_norm_limit=1000)
    assert created["status"] == "completed", created
    gene = read_genesis(output, run, parent)
    gate = train_gpu.run_job(directory=output, kind="gate")
    assert gate["status"] == "completed", gate
    read_gate(output, run, data, gene)
    for _ in range(2):
        result = train_gpu.run_job(directory=output, kind="train", windows=1)
        assert result["status"] == "completed", result
    chain = scan_run(output, run, data, gene)
    assert chain["completed_windows"] == chain["optimizer_steps"] == 2
    assert len(chain["segments"]) == 2
    first = read_json(output / "segments/000001/registration.json")
    second = read_json(output / "segments/000002/registration.json")
    assert first["parent_checkpoint_id"] == gene["checkpoint_id"]
    assert second["parent_checkpoint_id"] == chain["windows"][0]["checkpoint_id"]
    assert (output / "segments/000001/worker.claim").read_text() != (output / "segments/000002/worker.claim").read_text()
    assert all(not np.count_nonzero(v) for v in read_genesis(output, run, parent)["moments"].values())
    # The gate is disposable; the first real window inherits the untouched genesis.
    saved = read_arrays(output / "segments/000001/windows/000001/evidence.npz")
    for key, value in params.items():
        np.testing.assert_array_equal(saved[f"before_{key}"], value)
    terminal = output / "gate/worker_result.json"
    changed = read_json(terminal)
    changed["completed_windows"] = 99
    terminal.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Terminal evidence changed"):
        read_gate(output, run, data, gene)
    print(json.dumps({"three_stage_child_pipeline": True, "separate_training_processes": 2,
                      "genesis_disposable_gate_exact": True, "forged_terminal_rejected": True}))
