"""Small atomic JSON writes shared by the parent and isolated worker."""

import json
import os
import tempfile
from pathlib import Path

DISPLAY_ASSETS = ("catalog.json", "atlas.json", "run.json", "stimulus.png")
EVIDENCE_FILES = ("manifest.json", "records.npz", "inputs.npz", "neuron_totals.npz")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=".writing-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
