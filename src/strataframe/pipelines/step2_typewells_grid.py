# src/strataframe/pipelines/step2_typewells_grid.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from strataframe.contracts.step2_outputs import REP_CSV_HEADER
from strataframe.io.csv import write_csv
from strataframe.spatial.make_well_to_cell import load_or_build_well_to_cell_rows
from strataframe.typewell.grid_typewells import TypeWellConfig as GridTypeWellConfig
from strataframe.typewell.grid_typewells import select_type_wells_grid


@dataclass(frozen=True)
class Step2TypeWellsGridConfig:
    # Spatial (grid)
    grid_km: float = 10.0
    kernel_radius: int = 1
    kernel_radius_max: int = 3
    n_min_postqc: int = 12
    max_candidates_per_kernel: int = 80

    # Fallback search
    fallback_nearest_n: int = 40
    fallback_r_max_km: float = 50.0

    # Weighting
    use_distance_weights: bool = True
    sigma_km: float = 20.0

    # QC / features
    min_finite_frac: float = 0.20
    min_thickness: float = 1.0
    zmax_feature: float = 3.5
    zmax_shape: float = 3.5

    # GR downsample / percentile shaping
    n_lowres: int = 128
    p_lo: float = 1.0
    p_hi: float = 99.0
    min_finite_raw: int = 50


def _validate(cfg: Step2TypeWellsGridConfig) -> Step2TypeWellsGridConfig:
    grid_km = float(cfg.grid_km) if float(cfg.grid_km) > 0 else 10.0

    r0 = int(cfg.kernel_radius)
    rmax = int(cfg.kernel_radius_max)
    r0 = max(0, r0)
    rmax = max(r0, rmax)

    n_min_postqc = max(1, int(cfg.n_min_postqc))
    max_candidates = max(1, int(cfg.max_candidates_per_kernel))

    fallback_nearest_n = max(1, int(cfg.fallback_nearest_n))
    fallback_r_max_km = max(0.0, float(cfg.fallback_r_max_km))

    sigma_km = float(cfg.sigma_km)
    if sigma_km <= 0:
        sigma_km = 20.0

    min_finite_frac = float(cfg.min_finite_frac)
    min_finite_frac = min(1.0, max(0.0, min_finite_frac))

    min_thickness = max(0.0, float(cfg.min_thickness))

    zmax_feature = float(cfg.zmax_feature) if float(cfg.zmax_feature) > 0 else 3.5
    zmax_shape = float(cfg.zmax_shape) if float(cfg.zmax_shape) > 0 else 3.5

    n_lowres = int(cfg.n_lowres)
    if n_lowres < 8:
        n_lowres = 128

    p_lo = float(cfg.p_lo)
    p_hi = float(cfg.p_hi)
    if not (0.0 <= p_lo < p_hi <= 100.0):
        p_lo, p_hi = 1.0, 99.0

    min_finite_raw = max(1, int(cfg.min_finite_raw))

    return Step2TypeWellsGridConfig(
        grid_km=grid_km,
        kernel_radius=r0,
        kernel_radius_max=rmax,
        n_min_postqc=n_min_postqc,
        max_candidates_per_kernel=max_candidates,
        fallback_nearest_n=fallback_nearest_n,
        fallback_r_max_km=fallback_r_max_km,
        use_distance_weights=bool(cfg.use_distance_weights),
        sigma_km=sigma_km,
        min_finite_frac=min_finite_frac,
        min_thickness=min_thickness,
        zmax_feature=zmax_feature,
        zmax_shape=zmax_shape,
        n_lowres=n_lowres,
        p_lo=p_lo,
        p_hi=p_hi,
        min_finite_raw=min_finite_raw,
    )


def _with_gr_cfg(
    tw_cfg: GridTypeWellConfig,
    *,
    n_lowres: int,
    p_lo: float,
    p_hi: float,
    min_finite_raw: int,
) -> GridTypeWellConfig:
    """
    Update nested gr_cfg if present. Defensive if config schema evolves.
    """
    if not hasattr(tw_cfg, "gr_cfg"):
        return tw_cfg
    gr_cfg = getattr(tw_cfg, "gr_cfg", None)
    if gr_cfg is None or not hasattr(gr_cfg, "__class__"):
        return tw_cfg

    try:
        new_gr = gr_cfg.__class__(
            n_lowres=int(n_lowres),
            p_lo=float(p_lo),
            p_hi=float(p_hi),
            min_finite_raw=int(min_finite_raw),
        )
    except Exception:
        return tw_cfg

    try:
        return tw_cfg.__class__(**{**tw_cfg.__dict__, "gr_cfg": new_gr})
    except Exception:
        return tw_cfg


def run_step2_typewells_grid(
    *,
    well_to_cell_csv: Path,
    las_root: Path,
    out_dir: Path,
    cfg: Step2TypeWellsGridConfig,
    dry_run: bool = False,
    manifest_json: Optional[Path] = None,
    force_rebuild_well_to_cell: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg2 = _validate(cfg)

    rows, w2c_diag = load_or_build_well_to_cell_rows(
        well_to_cell_csv=Path(well_to_cell_csv),
        manifest_json=Path(manifest_json) if manifest_json else None,
        grid_km=float(cfg2.grid_km),
        force_rebuild=bool(force_rebuild_well_to_cell),
        dry_run=bool(dry_run),
    )

    if not rows:
        diag: Dict[str, Any] = {
            "error": "no_rows_in_well_to_cell",
            "well_to_cell": w2c_diag,
            "inputs": {
                "well_to_cell_csv": str(well_to_cell_csv),
                "manifest_json": str(manifest_json) if manifest_json else "",
                "las_root": str(las_root),
            },
            "params": cfg2.__dict__,
        }
        if not dry_run:
            (out_dir / "typewells_diagnostics.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
        return [], diag

    tw_cfg = GridTypeWellConfig(
        grid_km=float(cfg2.grid_km),
        kernel_radius=int(cfg2.kernel_radius),
        kernel_radius_max=int(cfg2.kernel_radius_max),
        n_min_postqc=int(cfg2.n_min_postqc),
        max_candidates_per_kernel=int(cfg2.max_candidates_per_kernel),
        fallback_nearest_n=int(cfg2.fallback_nearest_n),
        fallback_r_max_km=float(cfg2.fallback_r_max_km),
        use_distance_weights=bool(cfg2.use_distance_weights),
        sigma_km=float(cfg2.sigma_km),
        min_finite_frac=float(cfg2.min_finite_frac),
        min_thickness=float(cfg2.min_thickness),
        zmax_feature=float(cfg2.zmax_feature),
        zmax_shape=float(cfg2.zmax_shape),
    )
    tw_cfg = _with_gr_cfg(
        tw_cfg,
        n_lowres=int(cfg2.n_lowres),
        p_lo=float(cfg2.p_lo),
        p_hi=float(cfg2.p_hi),
        min_finite_raw=int(cfg2.min_finite_raw),
    )

    reps, diag0 = select_type_wells_grid(rows, las_root=Path(las_root), cfg=tw_cfg)
    diag: Dict[str, Any] = dict(diag0) if isinstance(diag0, dict) else {}
    diag["well_to_cell"] = w2c_diag
    diag["inputs"] = {
        "well_to_cell_csv": str(well_to_cell_csv),
        "manifest_json": str(manifest_json) if manifest_json else "",
        "las_root": str(las_root),
        "out_dir": str(out_dir),
    }
    diag["params"] = cfg2.__dict__

    if not dry_run:
        write_csv(out_dir / "representatives.csv", REP_CSV_HEADER, reps)
        (out_dir / "typewells_diagnostics.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")

    return reps, diag
