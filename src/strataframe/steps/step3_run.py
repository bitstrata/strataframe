# scripts/step3_run.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from strataframe.pipelines.step3_run_correlation import Step3Config, run_step3
from strataframe.pipelines.step3f_products import Step3ProductsConfig, run_step3_products


def _p(s: str) -> Path:
    return Path(s).expanduser().resolve()


def main() -> None:
    ap = argparse.ArgumentParser(description="Run strataframe Step 3 (type-well correlation + RGT + products).")

    ap.add_argument("--reps-csv", required=True, help="Step2 reps CSV (must include rep_id, lat, lon, bin_id/h3_cell).")
    ap.add_argument("--las-root", required=True, help="Root folder containing LAS files (for resolving relpaths).")
    ap.add_argument("--out-dir", required=True, help="Output directory for Step 3 artifacts.")
    ap.add_argument("--wells-gr-parquet", default="", help="Optional step0 wells_gr.parquet used to resolve LAS paths.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite cached Step 3 outputs.")
    ap.add_argument("--mode", type=str, default="full", choices=["prep", "full"], help="Run mode: prep (no DTW/RGT) or full.")

    # Candidate graph parameters (mapped to Step3Config cg_* fields)
    ap.add_argument("--cg-k-max", type=int, default=12)
    ap.add_argument("--cg-r-max-km", type=float, default=5.0)
    ap.add_argument("--cg-use-quadrants", action="store_true")
    ap.add_argument("--cg-no-quadrants", dest="cg_use_quadrants", action="store_false")
    ap.set_defaults(cg_use_quadrants=True)
    ap.add_argument("--cg-ensure-one-nn", action="store_true")
    ap.add_argument("--cg-no-ensure-one-nn", dest="cg_ensure_one_nn", action="store_false")
    ap.set_defaults(cg_ensure_one_nn=True)

    # Products
    ap.add_argument("--no-products", action="store_true", help="If set, do not generate chronostrat/tops products.")
    ap.add_argument("--tops-levels", default="0,2,4", help="Comma-separated hierarchy levels to export to CSV.")
    ap.add_argument("--tops-prefix", default="TOP")

    args = ap.parse_args()

    out_dir = _p(args.out_dir)

    cfg = Step3Config(
        reps_csv=_p(args.reps_csv),
        las_root=_p(args.las_root),
        wells_gr_parquet=_p(args.wells_gr_parquet) if args.wells_gr_parquet else None,
        mode=str(args.mode),
        cg_k_max=int(args.cg_k_max),
        cg_r_max_km=float(args.cg_r_max_km),
        cg_use_quadrants=bool(args.cg_use_quadrants),
        cg_ensure_one_nn=bool(args.cg_ensure_one_nn),
    )

    diag: Dict[str, Any] = run_step3(out_dir=out_dir, cfg=cfg, overwrite=bool(args.overwrite))

    products_manifest = None
    if str(args.mode).lower() == "prep":
        args.no_products = True

    if not bool(args.no_products):
        levels = [int(x.strip()) for x in str(args.tops_levels).split(",") if x.strip() != ""]
        pcfg = Step3ProductsConfig(export_levels=tuple(levels), tops_prefix=str(args.tops_prefix))

        products_manifest = run_step3_products(
            out_dir=out_dir,
            framework_nodes_csv=out_dir / "framework_nodes.csv",
            framework_edges_csv=out_dir / "framework_edges.csv",
            rep_arrays_npz=out_dir / "rep_arrays.npz",
            rgt_shifts_npz=out_dir / "rgt_shifts_resampled.npz",
            cfg=pcfg,
        )

    manifest = {"step3": diag, "products": products_manifest}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["step3"].get("counts", {}), indent=2))


if __name__ == "__main__":
    main()
