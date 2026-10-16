"""CNSBridge offline connectome tools."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from . import __version__


def _cfg(args):
    from .config import load_config
    from .paths import source_project_root

    over = {}
    if getattr(args, "brain", None):
        over.setdefault("brain", {})["source"] = args.brain
    return load_config(getattr(args, "config", None) or source_project_root() / "configs" / "malecns.yaml", over)


def _brain(cfg):
    from .brain import Brain, load_connectome

    c = load_connectome(cfg["brain"]["source"])
    print(c.summary())
    return Brain(c, cfg)


def cmd_download(args) -> int:
    from .data import download_malecns
    from .paths import project_asset_path

    download_malecns(project_asset_path(args.dir))
    print("next: cnsbridge build-brain")
    return 0


def cmd_build(args) -> int:
    from .brain import Brain, build_malecns
    from .config import load_config
    from .paths import project_asset_path

    cfg = load_config(args.config)
    c = build_malecns(project_asset_path(args.data_dir), min_synapses=args.min_synapses)
    print(c.summary())
    b = Brain(c, cfg)  # resolves groups and warns about empty ones
    for name in list(b.input_specs) + list(b.output_specs):
        print(f"  {name:10s} {c.group(name).size:6d} neurons")
    if args.core_hops:
        c = c.sensorimotor_core(args.core_hops, args.max_neurons)
        print("sensorimotor core:", c.summary())
    out = c.save(project_asset_path(args.out))
    print(f"saved -> {out}\ninspect it: cnsbridge inspect --brain {out}")
    return 0


def cmd_inspect(args) -> int:
    cfg = _cfg(args)
    brain = _brain(cfg)
    c = brain.connectome
    print("\ninput groups:")
    for k in brain.input_specs:
        print(f"  {k:10s} {c.group(k).size:6d}  types={brain.input_specs[k].types} side={brain.input_specs[k].side}")
    print("output groups:")
    for k in brain.output_specs:
        print(f"  {k:10s} {c.group(k).size:6d}  {brain.output_specs[k].note}")
    print("\npoke test (500 ms each):")
    brain.tick({}, 500)
    for g in [k for k in brain.input_specs if c.group(k).size]:
        r = brain.stimulate(g, args.hz, 500)
        print(f"  {g:10s} -> " + "  ".join(f"{k}={r[k]:5.1f}" for k in brain.output_specs))
    return 0


def cmd_calibrate(args) -> int:
    from .control.calibration import calibrate
    from .paths import project_asset_path

    cfg = _cfg(args)
    brain = _brain(cfg)
    calibrate(brain, cfg, project_asset_path(args.out))
    return 0


def cmd_bench(args) -> int:
    import time

    cfg = _cfg(args)
    brain = _brain(cfg)
    rng = np.random.default_rng(0)
    inputs = {k: rng.random(brain.connectome.group(k).size) * 60 for k in brain.input_specs}
    brain.tick(inputs, 100)
    t0 = time.perf_counter()
    brain.tick(inputs, args.ms)
    wall = time.perf_counter() - t0
    st = brain.net.stats
    print(f"{args.ms:.0f} ms of brain time in {wall * 1000:.0f} ms wall -> {args.ms / 1000 / wall:.2f}x real time")
    print(f"dt={brain.net.p.dt} ms, spikes so far {st.spikes:,}, synaptic events {st.synaptic_events:,}")
    return 0


# ---------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cnsbridge", description="Build, inspect and benchmark connectome models for CNSBridge.")
    p.add_argument("--version", action="version", version=f"cnsbridge {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--config", help="YAML file overriding defaults")
        sp.add_argument("--brain", help="Path to a real connectome .npz; defaults to the MaleCNS asset")

    sp = sub.add_parser("download", help="download connectome data")
    sp.add_argument("dataset", choices=["malecns"])
    sp.add_argument("--dir", default="data/malecns_v1")
    sp.set_defaults(func=cmd_download)

    sp = sub.add_parser("build-brain", help="build a signed connectome .npz from MaleCNS v1.0")
    sp.add_argument("--data-dir", default="data/malecns_v1")
    sp.add_argument("--config")
    sp.add_argument("--min-synapses", type=int, default=3)
    sp.add_argument("--core-hops", type=int, default=0, help="keep only neurons within N synapses of inputs and outputs")
    sp.add_argument("--max-neurons", type=int)
    sp.add_argument("--out", default="data/malecns_brain.npz")
    sp.set_defaults(func=cmd_build)

    sp = sub.add_parser("inspect", help="list input/output groups and poke each input")
    common(sp)
    sp.add_argument("--hz", type=float, default=100)
    sp.set_defaults(func=cmd_inspect)

    sp = sub.add_parser("calibrate", help="fit the descending-neuron read-out for this brain")
    common(sp)
    sp.add_argument("--out", default="readout_malecns.json")
    sp.set_defaults(func=cmd_calibrate)

    sp = sub.add_parser("bench", help="measure simulation speed")
    common(sp)
    sp.add_argument("--ms", type=float, default=1000)
    sp.set_defaults(func=cmd_bench)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
