"""Replay a ground video through a frozen network and inspect simulated activity.

This command never connects to a drone and does not read desired-action labels.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import shutil
from pathlib import Path


def analyze(session, out_dir, *, config=None, brain=None, stimuli=None, max_neurons=128, warmup_s=2.0, progress=None):
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"Output directory already exists: {out_dir}")
    try:
        import cv2  # noqa: F401 — validate optional dependencies before allocation
        import matplotlib  # noqa: F401
    except ImportError as exc:
        raise ValueError("Install offline analysis dependencies: pip install -e '.[analysis]'") from exc
    from cnsbridge.config import load_config
    from cnsbridge.paths import source_project_root

    from .collector import collect
    from .reporting import write_report

    project = source_project_root()
    cfg = load_config(config or project / "configs" / "malecns.yaml")
    source = brain or cfg["brain"]["source"]
    if str(source).lower() not in ("minifly", "synthetic", "mini"):
        source = str((project / source).resolve()) if not Path(source).is_absolute() else str(source)
    cfg["brain"]["source"] = source
    out_dir.mkdir(parents=True, exist_ok=False)
    try:
        arrays, metadata = collect(Path(session), cfg, stimuli_path=stimuli, max_neurons=max_neurons,
                                   warmup_s=warmup_s, progress=progress)
        metadata["software"] = {"opencv": cv2.__version__, "matplotlib": matplotlib.__version__,
                                **{name: importlib.metadata.version(name) for name in ("numpy", "scipy", "PyYAML")}}
        return write_report(out_dir, arrays, metadata)
    except BaseException:
        # This directory was exclusively created by this run, never pre-existing.
        shutil.rmtree(out_dir)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path, help="Existing ground-recording session directory (not a bare video)")
    parser.add_argument("--out-dir", type=Path, required=True, help="New output directory; never overwrites")
    parser.add_argument("--stimuli", type=Path, help="Optional start_s,end_s,stimulus CSV in AVI playback seconds")
    parser.add_argument("--config", type=Path, help="Config YAML; defaults to configs/malecns.yaml in the source checkout")
    parser.add_argument("--brain", help="Override network source; minifly is only a synthetic software check")
    parser.add_argument("--max-neurons", type=int, default=128, help="Bounded neuron sample, 1–4096 (default: 128)")
    parser.add_argument("--warmup-s", type=float, default=2.0, help="Exclude initial playback seconds from comparisons only (default: 2)")
    args = parser.parse_args(argv)
    try:
        report = analyze(args.session, args.out_dir, config=args.config, brain=args.brain, stimuli=args.stimuli,
                         max_neurons=args.max_neurons, warmup_s=args.warmup_s,
                         progress=lambda done, total: print(f"REPLAY | {done}/{total} frames", flush=True))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"ANALYSIS STOPPED: {exc}\n")
    print(f"SIMULATED ACTIVITY | {report['frame_count']} frames | {report['neural_provenance']['network']['name']}")
    print(f"REPORT: {(args.out_dir / 'index.html').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
