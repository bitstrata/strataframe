# src/strataframe/steps/step2_select_reps.py
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from strataframe.io.csv import read_csv_rows, write_csv
from strataframe.graph.select_representatives import RepSelectConfig, select_representatives
from strataframe.typewell.local_typewell import TypeWellConfig, build_cell_typewell, place_rep_against_template


# =============================================================================
# Step-2 config
# =============================================================================

@dataclass(frozen=True)
class Step2Config:
    # --- Representative selection objective ---
    n_rep: int = 1000
    quota_mode: str = "equal"          # "equal" | "proportional"
    q_min: int = 5                     # used only for proportional
    candidate_method: str = "farthest" # "farthest" | "random"
    seed: int = 42

    max_candidates_per_cell: int = 25
    require_gr: bool = True

    # --- Binning scheme (compat only; Step2 consumes Step1 outputs) ---
    bin_scheme: str = "grid"           # "grid" | "h3" (legacy)
    grid_km: float = 10.0              # informational only in this pipeline path

    # --- Typewell / placement ---
    build_typewells: bool = True
    typewell: TypeWellConfig = TypeWellConfig()


# =============================================================================
# Contracts
# =============================================================================

WELL_TO_CELL_HEADER = [
    "url", "kgs_id", "api", "api_num_nodash", "operator", "lease",
    "lat", "lon", "h3_cell", "h3_res",
    "bin_scheme", "grid_m", "proj_method", "proj_epsg", "x_m", "y_m", "grid_ix", "grid_iy", "grid_cell",
]

REP_CSV_HEADER = [
    "rep_id", "h3_cell", "h3_res", "score",
    "url", "kgs_id", "api", "api_num_nodash", "operator", "lease", "lat", "lon",
    "las_path",
    "picked_gr", "picked_por", "picked_den", "picked_neu", "picked_pe", "picked_dt",
]

CELL_TYPEWELL_HEADER = [
    "cell_id", "grid_ix", "grid_iy",
    "n_kernel", "n_read", "n_used",
    "template_path", "type_url", "type_las_path",
    "thickness_median", "ntg_median", "iqr_median",
    "error",
]

REP_PLACEMENT_HEADER = [
    "rep_id", "cell_id", "url", "las_path",
    "status",
    "thickness", "expected_thickness",
    "z_top", "z_base",
    "placed_z_top", "placed_z_base",
    "short_flag",
    "missing_top_flag", "missing_base_flag", "condensed_flag",
    "match_j_start", "match_j_end",
    "match_s_start", "match_s_end",
    "match_span", "warp_ratio",
    "dtw_cost_per_step",
    "error",
]


# =============================================================================
# Helpers
# =============================================================================

_GCELL_RE = re.compile(r"^g(-?\d+)_(-?\d+)$", re.IGNORECASE)
_GRID_RE = re.compile(r"^GRID_(-?\d+)_(-?\d+)$", re.IGNORECASE)
_RC_RE = re.compile(r"r(\d+)_c(\d+)", re.IGNORECASE)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _as_str(v: Any) -> str:
    return "" if v is None else str(v)


def _parse_cell_xy(cell_id: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Supported:
      - g<col>_<row>      (your Step1: g10_6 => col=10,row=6)
      - GRID_<col>_<row>
      - r00010_c00006 or r10_c6 (interprets r=row, c=col)
    Returns (col, row)
    """
    s = (cell_id or "").strip()
    if not s:
        return None, None

    m = _GCELL_RE.match(s)
    if m:
        return int(m.group(1)), int(m.group(2))

    m = _GRID_RE.match(s)
    if m:
        return int(m.group(1)), int(m.group(2))

    m = _RC_RE.search(s)
    if m:
        row = int(m.group(1))
        col = int(m.group(2))
        return col, row

    return None, None


def _format_cell_id(col: int, row: int, *, style_hint: str) -> str:
    s = (style_hint or "").strip()
    if s.upper().startswith("GRID_"):
        return f"GRID_{int(col)}_{int(row)}"
    # default to Step1 "g{col}_{row}"
    return f"g{int(col)}_{int(row)}"


def _get_cell_id(r: Dict[str, str]) -> str:
    for k in ("grid_cell", "h3_cell", "bin_id", "cell_id"):
        v = (r.get(k) or "").strip()
        if v:
            return v
    return ""


def _normalize_w2c_rows(rows: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """
    Ensure every row has h3_cell set (Step2/3 compatibility),
    pulling from bin_id/cell_id if needed.
    """
    out: List[Dict[str, str]] = []
    fixed_h3 = 0
    missing_cell = 0

    for r in rows:
        rr = dict(r)

        cell = (rr.get("h3_cell") or "").strip()
        if not cell:
            cell = (rr.get("bin_id") or "").strip() or (rr.get("cell_id") or "").strip()
            if cell:
                rr["h3_cell"] = cell
                fixed_h3 += 1

        if not cell:
            missing_cell += 1
            continue

        if not (rr.get("grid_cell") or "").strip():
            rr["grid_cell"] = rr["h3_cell"]

        out.append(rr)

    diag = {
        "rows_in": int(len(rows)),
        "rows_out": int(len(out)),
        "fixed_h3_cell": int(fixed_h3),
        "dropped_missing_cell": int(missing_cell),
    }
    return out, diag


def _empty_placement(rep: Dict[str, Any], cell_id: str, status: str, error: str) -> Dict[str, Any]:
    """
    Build a placement row that satisfies REP_PLACEMENT_HEADER without crashing downstream.
    """
    row: Dict[str, Any] = {k: "" for k in REP_PLACEMENT_HEADER}
    row["rep_id"] = _as_str(rep.get("rep_id", "")).strip()
    row["cell_id"] = _as_str(cell_id).strip()
    row["url"] = _as_str(rep.get("url", "")).strip()
    row["las_path"] = _as_str(rep.get("las_path", "")).strip()
    row["status"] = str(status)
    row["error"] = str(error)[:500]
    return row


# =============================================================================
# Step 2 orchestration
# =============================================================================

def run_step2(
    *,
    well_to_cell_csv: Path,
    las_root: Path,
    out_dir: Path,
    cfg: Step2Config,
    dry_run: bool = False,
    manifest_json: Optional[Path] = None,
    force_rebuild_well_to_cell: bool = False,  # kept for signature compatibility
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)

    if not well_to_cell_csv.exists():
        raise FileNotFoundError(
            f"Missing well_to_cell.csv: {well_to_cell_csv}. "
            "Step 2 in this pipeline consumes Step 1 outputs; run Step 1 first."
        )

    # Load + normalize well_to_cell
    rows_raw = read_csv_rows(well_to_cell_csv)
    rows, w2c_diag = _normalize_w2c_rows(rows_raw)

    rep_cfg = RepSelectConfig(
        reps_per_cell=2,
        max_candidates_per_cell=int(cfg.max_candidates_per_cell),
        require_gr=bool(cfg.require_gr),
    )

    reps, diag = select_representatives(
        rows,
        las_root=las_root,
        cfg=rep_cfg,
        n_rep_target=int(cfg.n_rep),
        quota_mode=str(cfg.quota_mode),
        q_min=int(cfg.q_min),
        candidate_method=str(cfg.candidate_method),
        seed=int(cfg.seed),
        out_dir=None if dry_run else out_dir,
    )

    diag = dict(diag or {})
    diag["well_to_cell"] = {"path": str(well_to_cell_csv), "normalized": w2c_diag}
    if manifest_json is not None:
        diag["step1_manifest"] = str(manifest_json)

    if not dry_run:
        write_csv(out_dir / "representatives.csv", REP_CSV_HEADER, reps)
        (out_dir / "rep_selection_diagnostics.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")

    # --- Typewells + placement ---
    if bool(cfg.build_typewells) and not dry_run and reps:
        templates_dir = out_dir / "cell_templates"
        templates_dir.mkdir(parents=True, exist_ok=True)

        if manifest_json is not None and manifest_json.exists():
            try:
                (out_dir / "grid_meta.json").write_text(
                    json.dumps(_read_json(manifest_json), indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass

        # Index rows by cell id
        by_cell: Dict[str, List[Dict[str, str]]] = {}
        for r in rows:
            cid = _get_cell_id(r)
            if cid:
                by_cell.setdefault(cid, []).append(r)

        rep_cells = sorted({str(r.get("h3_cell", "")).strip() for r in reps if str(r.get("h3_cell", "")).strip()})

        cell_meta_list: List[Dict[str, Any]] = []
        cell_meta_by_id: Dict[str, Dict[str, Any]] = {}

        tw_cfg = cfg.typewell

        for cell_id in rep_cells:
            col0, row0 = _parse_cell_xy(cell_id)
            if col0 is None or row0 is None:
                col0, row0 = 0, 0

            # build kernel with adaptive expansion
            kernel_rows: List[Dict[str, str]] = []
            r0 = int(getattr(tw_cfg, "kernel_radius", 1))
            rmax = int(getattr(tw_cfg, "kernel_radius_max", max(2, r0)))

            r_use = r0
            style_hint = cell_id

            while True:
                kernel_rows = []
                for dc in range(-r_use, r_use + 1):
                    for dr in range(-r_use, r_use + 1):
                        cid = _format_cell_id(col0 + dc, row0 + dr, style_hint=style_hint)
                        kernel_rows.extend(by_cell.get(cid, []))
                if len(kernel_rows) >= int(getattr(tw_cfg, "min_kernel_wells", 1)) or r_use >= rmax:
                    break
                r_use += 1

            meta = build_cell_typewell(
                cell_id=cell_id,
                cell_ix=int(col0),
                cell_iy=int(row0),
                kernel_rows=kernel_rows,
                las_root=las_root,
                cfg=tw_cfg,
                templates_dir=templates_dir,
                seed=int(cfg.seed),
            )

            meta.setdefault("cell_id", cell_id)
            meta.setdefault("grid_ix", str(int(col0)))
            meta.setdefault("grid_iy", str(int(row0)))
            meta.setdefault("error", "")
            meta.setdefault("template_path", "")
            cell_meta_list.append(meta)
            cell_meta_by_id[cell_id] = meta

        write_csv(out_dir / "cell_typewells.csv", CELL_TYPEWELL_HEADER, cell_meta_list)

        # Place reps robustly (never crash if template missing)
        placements: List[Dict[str, Any]] = []
        for rep in reps:
            cell_id = str(rep.get("h3_cell", "")).strip()
            cm = cell_meta_by_id.get(cell_id, {})
            tpl = str(cm.get("template_path", "") or "").strip()

            if not tpl or not Path(tpl).is_file():
                err = (cm.get("error") or "").strip() or "no_template"
                placements.append(_empty_placement(rep, cell_id, status="NO_TEMPLATE", error=err))
                continue

            try:
                p = place_rep_against_template(
                    rep_row=rep,
                    cell_meta=cm,
                    las_root=las_root,
                    cfg=tw_cfg,
                )
                # ensure all keys exist
                p.setdefault("error", "")
                p.setdefault("match_span", "")
                p.setdefault("warp_ratio", "")
                placements.append(p)
            except Exception as e:
                placements.append(_empty_placement(rep, cell_id, status="PLACE_FAIL", error=f"{type(e).__name__}: {e}"))

        write_csv(out_dir / "rep_placement.csv", REP_PLACEMENT_HEADER, placements)

    return reps, diag


# =============================================================================
# CLI
# =============================================================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Step 2: select reps; build local typewells + placement (robust).")
    p.add_argument("--well-to-cell-csv", type=Path, required=True, help="Path to Step1 well_to_cell.csv")
    p.add_argument("--manifest-json", type=Path, default=None, help="Optional Step1 manifest.json (for grid_meta.json copy)")
    p.add_argument("--force-rebuild-well-to-cell", action="store_true", help="Kept for CLI compatibility (ignored)")

    p.add_argument("--las-root", type=Path, default=Path("data/las"), help="Directory containing LAS files")
    p.add_argument("--out", type=Path, required=True, help="Output directory")

    p.add_argument("--n-rep", type=int, default=1000)
    p.add_argument("--quota-mode", choices=["equal", "proportional"], default="equal")
    p.add_argument("--q-min", type=int, default=5)
    p.add_argument("--candidate-method", choices=["farthest", "random"], default="farthest")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-candidates-per-cell", type=int, default=25)
    p.add_argument("--no-require-gr", action="store_true")

    p.add_argument("--no-typewells", action="store_true", help="Disable template build + placement.")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    cfg = Step2Config(
        n_rep=int(args.n_rep),
        quota_mode=str(args.quota_mode),
        q_min=int(args.q_min),
        candidate_method=str(args.candidate_method),
        seed=int(args.seed),
        max_candidates_per_cell=int(args.max_candidates_per_cell),
        require_gr=not bool(args.no_require_gr),
        build_typewells=not bool(args.no_typewells),
        typewell=TypeWellConfig(),
    )

    run_step2(
        well_to_cell_csv=Path(args.well_to_cell_csv),
        manifest_json=Path(args.manifest_json) if args.manifest_json else None,
        force_rebuild_well_to_cell=bool(args.force_rebuild_well_to_cell),
        las_root=Path(args.las_root),
        out_dir=Path(args.out),
        cfg=cfg,
        dry_run=bool(args.dry_run),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
