# src/strataframe/steps/step4_run.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from strataframe.pipelines.step4_type_columns import IntervalStatsConfig, Step4Config, run_step4_type_columns
from strataframe.rgt.chronostrat import ChronostratConfig
from strataframe.rgt.monotonic import MonotonicConfig
from strataframe.rgt.wavelet_tops import CwtConfig


def _p(s: str) -> Path:
    return Path(s).expanduser().resolve()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Step 4: build per-cell type strat columns.")

    p.add_argument("--framework-nodes-csv", required=True)
    p.add_argument("--framework-edges-csv", required=True)
    p.add_argument("--rep-arrays-npz", required=True)
    p.add_argument("--rgt-shifts-npz", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--overwrite", action="store_true")

    # Kernel controls
    p.add_argument("--kernel-radius", type=int, default=1)
    p.add_argument("--kernel-radius-max", type=int, default=3)
    p.add_argument("--min-wells-per-cell", type=int, default=5)
    p.add_argument("--max-wells-per-cell", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)

    # Chronostrat config
    p.add_argument("--n-rgt", type=int, default=800)
    p.add_argument("--rgt-pad-frac", type=float, default=0.02)
    p.add_argument("--monotonic-mode", type=str, default="nondecreasing")

    # CWT config
    p.add_argument("--cwt-widths", type=str, default="2,4,6,8,12,16,24,32")
    p.add_argument("--cwt-snap-window", type=int, default=25)
    p.add_argument("--cwt-include-endpoints", action="store_true")
    p.add_argument("--cwt-no-include-endpoints", dest="cwt_include_endpoints", action="store_false")
    p.set_defaults(cwt_include_endpoints=True)

    # Interval stats config
    p.add_argument("--ntg-cutoff", type=float, default=0.40)
    p.add_argument("--min-finite", type=int, default=20)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    widths = [int(x.strip()) for x in str(args.cwt_widths).split(",") if x.strip() != ""]
    chron = ChronostratConfig(
        n_rgt=int(args.n_rgt),
        rgt_pad_frac=float(args.rgt_pad_frac),
        monotonic=MonotonicConfig(mode=str(args.monotonic_mode)),
    )
    cwt = CwtConfig(
        widths=tuple(widths),
        snap_window=int(args.cwt_snap_window),
        include_endpoints=bool(args.cwt_include_endpoints),
    )
    interval = IntervalStatsConfig(ntg_cutoff=float(args.ntg_cutoff), min_finite=int(args.min_finite))

    cfg = Step4Config(
        framework_nodes_csv=_p(args.framework_nodes_csv),
        framework_edges_csv=_p(args.framework_edges_csv),
        rep_arrays_npz=_p(args.rep_arrays_npz),
        rgt_shifts_npz=_p(args.rgt_shifts_npz),
        kernel_radius=int(args.kernel_radius),
        kernel_radius_max=int(args.kernel_radius_max),
        min_wells_per_cell=int(args.min_wells_per_cell),
        max_wells_per_cell=int(args.max_wells_per_cell),
        seed=int(args.seed),
        chronostrat=chron,
        cwt=cwt,
        interval_stats=interval,
    )

    run_step4_type_columns(out_dir=_p(args.out_dir), cfg=cfg, overwrite=bool(args.overwrite))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
