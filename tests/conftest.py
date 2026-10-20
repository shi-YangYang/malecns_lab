"""Optional local real-connectome assets for integration checks, never fallback."""

from pathlib import Path

import pytest

from malecns_lab.brain import load_connectome
from malecns_lab.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def malecns_path():
    path = ROOT / "data" / "malecns_brain.npz"
    if not path.is_file():
        pytest.skip("Real MaleCNS asset is not installed; no substitute graph is used")
    return path


@pytest.fixture(scope="session")
def malecns_graph(malecns_path):
    return load_connectome(malecns_path)


@pytest.fixture
def malecns_config(malecns_path):
    return load_config(ROOT / "configs" / "malecns.yaml")
