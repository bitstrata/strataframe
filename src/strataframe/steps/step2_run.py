# src/strataframe/steps/step2_run.py
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

from strataframe.pipelines.step2_reps import Step2RepsConfig, run_step2_reps
from strataframe.pipelines.step2_typewells_grid import Step2TypeWellsGridConfig, run_step2_typewells_grid
from strataframe.typewell.local_typewell import TypeWellConfig


# Back-compat: preserve the name run_step2 used by earlier orchestrators.
def run_step2(
    *,
    well_to_cell_csv: Path,
    las_root: Path,
    out_dir: Path,
    cfg: Step2RepsConfig,
    dry_run: bool = False,
    manifest_json: Optional[Path] = None,
    force_rebuild_well_to_cell: bool = False,
):
    return run_step2_reps(
        well_to_cell_csv=well_to_cell_csv,
        las_root=las_root,
        out_dir=out_dir,
        cfg=cfg,
        dry_run=dry_run,
        manifest_json=manifest_json,
        force_rebuild_well_to_cell=force_rebuild_well_to_cell,
    )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Step 2: representative selection and/or typewell selection.")
    sub = p.add_subparsers(dest="mode", required=True)

    # -------------------------------------------------------------------------
    # reps (rep selection + optional local typewells/placement)
    # -------------------------------------------------------------------------
    pr = sub.add_parser("reps", help="Select representatives; optionally build local typewells + placements.")
    pr.add_argument("--well-to-cell-csv", type=Path, required=True)
    pr.add_argument("--manifest-json", type=Path, default=None)
    pr.add_argument("--force-rebuild-well-to-cell", action="store_true")
    pr.add_argument("--las-root", type=Path, default=Path("data/las"))
    pr.add_argument("--out", type=Path, required=True)

    pr.add_argument("--n-rep", type=int, default=1000)
    pr.add_argument("--quota-mode", choices=["equal", "proportional"], default="equal")
    pr.add_argument("--q-min", type=int, default=5)
    pr.add_argument("--candidate-method", choices=["farthest", "random"], default="farthest")
    pr.add_argument("--seed", type=int, default=42)
    pr.add_argument("--max-candidates-per-cell", type=int, default=25)
    pr.add_argument("--no-require-gr", action="store_true")
    pr.add_argument("--grid-km", type=float, default=10.0)

    pr.add_argument("--no-typewells", action="store_true")
    pr.add_argument("--dry-run", action="store_true")
    pr.add_argument("--max-las-mb", type=int, default=512)
    pr.add_argument("--max-las-curves", type=int, default=0)
    pr.add_argument("--typewell-max-cells", type=int, default=0)
    pr.add_argument("--typewell-max-kernel-wells", type=int, default=200)
    pr.add_argument("--typewell-gc-every", type=int, default=10)
    pr.add_argument("--typewell-max-rows", type=int, default=0)
    pr.add_argument("--typewell-subprocess", action="store_true")
    pr.add_argument("--typewell-subprocess-min-mb", type=int, default=0)
    pr.add_argument("--typewell-worker-timeout", type=int, default=0)
    pr.add_argument("--typewell-worker-mem-mb", type=int, default=0)
    pr.add_argument("--typewell-gr-cache-dir", type=Path, default=None)
    pr.add_argument("--typewell-no-gr-cache", action="store_true")
    pr.add_argument("--no-resume", action="store_true")

    # -------------------------------------------------------------------------
    # typewells-grid (grid typewell selection)
    # -------------------------------------------------------------------------
    pg = sub.add_parser("typewells-grid", help="Select typewells on a grid kernel using LAS + well_to_cell.")
    pg.add_argument("--well-to-cell-csv", type=Path, required=True)
    pg.add_argument("--manifest-json", type=Path, default=None)
    pg.add_argument("--force-rebuild-well-to-cell", action="store_true")
    pg.add_argument("--las-root", type=Path, default=Path("data/las"))
    pg.add_argument("--out", type=Path, required=True)
    pg.add_argument("--dry-run", action="store_true")

    pg.add_argument("--grid-km", type=float, default=10.0)
    pg.add_argument("--kernel-radius", type=int, default=1)
    pg.add_argument("--kernel-radius-max", type=int, default=3)
    pg.add_argument("--n-min-postqc", type=int, default=12)
    pg.add_argument("--max-candidates-per-kernel", type=int, default=80)
    pg.add_argument("--fallback-nearest-n", type=int, default=40)
    pg.add_argument("--fallback-r-max-km", type=float, default=50.0)

    # Correct, single-destination boolean flag pair (default: True)
    wg = pg.add_mutually_exclusive_group()
    wg.add_argument(
        "--use-distance-weights",
        dest="use_distance_weights",
        action="store_true",
        default=True,
        help="Enable distance weighting (default).",
    )
    wg.add_argument(
        "--no-distance-weights",
        dest="use_distance_weights",
        action="store_false",
        help="Disable distance weighting.",
    )

    pg.add_argument("--sigma-km", type=float, default=20.0)
    pg.add_argument("--min-finite-frac", type=float, default=0.20)
    pg.add_argument("--min-thickness", type=float, default=1.0)
    pg.add_argument("--zmax-feature", type=float, default=3.5)
    pg.add_argument("--zmax-shape", type=float, default=3.5)
    pg.add_argument("--n-lowres", type=int, default=128)
    pg.add_argument("--p-lo", type=float, default=1.0)
    pg.add_argument("--p-hi", type=float, default=99.0)
    pg.add_argument("--min-finite-raw", type=int, default=50)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    if args.mode == "reps":
        tw_cfg = TypeWellConfig()
        if int(args.typewell_max_rows) >= 0:
            tw_cfg = replace(tw_cfg, max_las_rows=int(args.typewell_max_rows))
        if (
            bool(args.typewell_subprocess)
            or int(args.typewell_subprocess_min_mb) > 0
            or int(args.typewell_worker_timeout) != 0
            or int(args.typewell_worker_mem_mb) != 0
        ):
            tw_cfg = replace(
                tw_cfg,
                use_subprocess=bool(args.typewell_subprocess),
                subprocess_min_las_mb=int(args.typewell_subprocess_min_mb),
                worker_timeout_sec=int(args.typewell_worker_timeout),
                worker_mem_mb=int(args.typewell_worker_mem_mb),
            )

        if bool(args.typewell_no_gr_cache):
            cache_dir = ""
        elif args.typewell_gr_cache_dir is not None and str(args.typewell_gr_cache_dir).strip():
            cache_dir = str(args.typewell_gr_cache_dir)
        else:
            cache_dir = str(Path(args.out) / "gr_cache")
        tw_cfg = replace(tw_cfg, gr_cache_dir=str(cache_dir), gr_cache_read=True, gr_cache_write=True)

        cfg = Step2RepsConfig(
            n_rep=int(args.n_rep),
            quota_mode=str(args.quota_mode),
            q_min=int(args.q_min),
            candidate_method=str(args.candidate_method),
            seed=int(args.seed),
            max_candidates_per_cell=int(args.max_candidates_per_cell),
            require_gr=not bool(args.no_require_gr),
            grid_km=float(args.grid_km),
            build_typewells=not bool(args.no_typewells),
            typewell=tw_cfg,
            max_las_mb=int(args.max_las_mb),
            max_las_curves=int(args.max_las_curves),
            typewell_max_cells=int(args.typewell_max_cells),
            typewell_max_kernel_wells=int(args.typewell_max_kernel_wells),
            typewell_gc_every=int(args.typewell_gc_every),
            resume=not bool(args.no_resume),
        )
        run_step2_reps(
            well_to_cell_csv=Path(args.well_to_cell_csv),
            manifest_json=Path(args.manifest_json) if args.manifest_json else None,
            force_rebuild_well_to_cell=bool(args.force_rebuild_well_to_cell),
            las_root=Path(args.las_root),
            out_dir=Path(args.out),
            cfg=cfg,
            dry_run=bool(args.dry_run),
        )
        return 0

    if args.mode == "typewells-grid":
        cfg = Step2TypeWellsGridConfig(
            grid_km=float(args.grid_km),
            kernel_radius=int(args.kernel_radius),
            kernel_radius_max=int(args.kernel_radius_max),
            n_min_postqc=int(args.n_min_postqc),
            max_candidates_per_kernel=int(args.max_candidates_per_kernel),
            fallback_nearest_n=int(args.fallback_nearest_n),
            fallback_r_max_km=float(args.fallback_r_max_km),
            use_distance_weights=bool(args.use_distance_weights),
            sigma_km=float(args.sigma_km),
            min_finite_frac=float(args.min_finite_frac),
            min_thickness=float(args.min_thickness),
            zmax_feature=float(args.zmax_feature),
            zmax_shape=float(args.zmax_shape),
            n_lowres=int(args.n_lowres),
            p_lo=float(args.p_lo),
            p_hi=float(args.p_hi),
            min_finite_raw=int(args.min_finite_raw),
        )
        run_step2_typewells_grid(
            well_to_cell_csv=Path(args.well_to_cell_csv),
            manifest_json=Path(args.manifest_json) if args.manifest_json else None,
            force_rebuild_well_to_cell=bool(args.force_rebuild_well_to_cell),
            las_root=Path(args.las_root),
            out_dir=Path(args.out),
            cfg=cfg,
            dry_run=bool(args.dry_run),
        )
        return 0

    raise SystemExit(f"Unknown mode: {args.mode!r}")


if __name__ == "__main__":
    raise SystemExit(main())
