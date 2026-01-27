# src/strataframe/spatial/make_well_to_cell.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from strataframe.io.csv import read_csv_rows, write_csv

# -----------------------------------------------------------------------------
# Output contract: well_to_cell.csv (GRID workflow only)
# -----------------------------------------------------------------------------
WELL_TO_CELL_HEADER = [
    "url",
    "kgs_id",
    "api",
    "api_num_nodash",
    "operator",
    "lease",
    "lat",
    "lon",
    "cell_id",
    "cell_tag",
    "grid_km",
    "grid_m",
    "proj_method",
    "proj_epsg",
    "x_m",
    "y_m",
    "grid_ix",
    "grid_iy",
    "grid_cell",
]

_G_CELL_RE = re.compile(r"^g([+-]?\d+)_([+-]?\d+)$")
_IJ_RE = re.compile(r"^([+-]?\d+)[,_]([+-]?\d+)$")


def parse_grid_cell_id(cell_id: str) -> Optional[Tuple[int, int]]:
    """
    Parse grid cell IDs. Strictly supports:
      - g{ix}_{iy}
      - {ix}_{iy} or {ix},{iy}

    Returns None if not parseable (prevents silent (0,0) fallbacks).
    """
    s = (cell_id or "").strip()
    m = _G_CELL_RE.match(s)
    if m:
        return int(m.group(1)), int(m.group(2))
    m2 = _IJ_RE.match(s)
    if m2:
        return int(m2.group(1)), int(m2.group(2))
    return None


def grid_cell_id_from_ij(ix: int, iy: int) -> str:
    # Must match grid id convention used in Step1 binning outputs.
    return f"g{int(ix)}_{int(iy)}"


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _try_manifest_output_path(manifest_json: Path, keys: Sequence[str]) -> Optional[Path]:
    """
    Attempt to read a path from manifest_json["outputs"][key] for any key in keys.
    Resolves relative paths against manifest parent.
    """
    try:
        man = _read_json(manifest_json)
    except Exception:
        return None
    if not isinstance(man, dict):
        return None
    out = man.get("outputs", {})
    if not isinstance(out, dict):
        return None
    for k in keys:
        v = out.get(k)
        if not isinstance(v, str):
            continue
        s = v.strip()
        if not s:
            continue
        p = Path(s)
        if not p.is_absolute():
            p = (manifest_json.parent / p).resolve()
        if p.exists():
            return p
    return None


def _infer_bins_csv(*, well_to_cell_csv: Path, manifest_json: Optional[Path]) -> Optional[Path]:
    """
    Best-effort bins.csv inference.
    Order:
      1) Step1 manifest outputs (bins_csv / bins / bins_path)
      2) sibling bins.csv beside well_to_cell.csv
      3) search upward for common run layouts (01_bins/bins.csv)
    """
    sibling = well_to_cell_csv.parent / "bins.csv"
    if sibling.exists():
        return sibling

    if manifest_json is not None and manifest_json.exists():
        p = _try_manifest_output_path(manifest_json, keys=("bins_csv", "bins", "bins_path"))
        if p is not None:
            return p

    for parent in list(well_to_cell_csv.parents)[:6]:
        cand1 = parent / "bins.csv"
        if cand1.exists():
            return cand1
        cand2 = parent / "01_bins" / "bins.csv"
        if cand2.exists():
            return cand2

    return None


def _infer_grid_km_from_step1_manifest(manifest_json: Optional[Path]) -> Optional[float]:
    """
    Best-effort: extract chosen grid cell size (km) from Step1 manifest.
    """
    if manifest_json is None or not manifest_json.exists():
        return None
    try:
        man = _read_json(manifest_json)
    except Exception:
        return None
    if not isinstance(man, dict):
        return None

    try_paths = [
        ("params", "grid_cell_km"),
        ("diagnostics", "chosen_grid_cell_km"),
        ("diagnostics", "grid", "cell_km"),
    ]
    for path in try_paths:
        cur: Any = man
        ok = True
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                ok = False
                break
            cur = cur.get(k)
        if not ok:
            continue
        try:
            if cur is None:
                continue
            v = float(cur)
            if np.isfinite(v) and v > 0:
                return float(v)
        except Exception:
            continue
    return None


def _infer_step0_wells_gr(*, well_to_cell_csv: Path, manifest_json: Optional[Path]) -> Optional[Path]:
    """
    Best-effort wells_gr.parquet inference.
    Order:
      1) manifest outputs (wells_gr / wells_gr_parquet / step0_wells_gr)
      2) search upward for common run layouts (00_wells_gr/wells_gr.parquet)
      3) sibling wells_gr.parquet
    """
    if manifest_json is not None and manifest_json.exists():
        p = _try_manifest_output_path(manifest_json, keys=("wells_gr", "wells_gr_parquet", "step0_wells_gr"))
        if p is not None:
            return p

    for parent in list(well_to_cell_csv.parents)[:7]:
        cand = parent / "00_wells_gr" / "wells_gr.parquet"
        if cand.exists():
            return cand
        cand2 = parent / "wells_gr.parquet"
        if cand2.exists():
            return cand2

    return None


def _read_parquet_any(path: Path):
    """
    Read parquet from:
      - a single .parquet file, OR
      - a directory dataset (e.g. wells_gr.parquet/part-*.parquet).
    Returns a pandas.DataFrame.
    """
    import pandas as pd  # type: ignore

    p = Path(path)
    if p.is_dir():
        import pyarrow.dataset as ds  # type: ignore

        return ds.dataset(str(p), format="parquet").to_table().to_pandas()
    return pd.read_parquet(p)


def build_well_to_cell_rows_grid(
    *,
    well_to_cell_csv: Path,
    manifest_json: Optional[Path],
    grid_km_hint: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Build well_to_cell rows using:
      - Step1 bins.csv (well_id -> bin_id), where bin_id is the grid cell ID
      - Step0 wells_gr.parquet for url/lat/lon + identifiers

    Grid only. No H3/legacy behavior.
    """
    bins_csv = _infer_bins_csv(well_to_cell_csv=Path(well_to_cell_csv), manifest_json=manifest_json)
    if bins_csv is None or not bins_csv.exists():
        return [], {
            "error": "bins_csv_not_found",
            "well_to_cell_csv": str(well_to_cell_csv),
            "manifest_json": str(manifest_json) if manifest_json else "",
        }

    wells_gr = _infer_step0_wells_gr(well_to_cell_csv=Path(well_to_cell_csv), manifest_json=manifest_json)
    if wells_gr is None or not wells_gr.exists():
        return [], {
            "error": "step0_wells_gr_not_found",
            "bins_csv": str(bins_csv),
            "well_to_cell_csv": str(well_to_cell_csv),
            "manifest_json": str(manifest_json) if manifest_json else "",
        }

    grid_km = _infer_grid_km_from_step1_manifest(manifest_json)
    if grid_km is None:
        grid_km = float(grid_km_hint)
    if not (np.isfinite(grid_km) and grid_km > 0):
        grid_km = 10.0

    grid_m = float(grid_km) * 1000.0
    cell_tag = f"grid:{float(grid_km):g}km"

    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        return [], {"error": f"pandas_required: {type(e).__name__}: {e}"}

    b = pd.read_csv(bins_csv)

    # Accept case variants
    cols_b = {c.lower(): c for c in b.columns}
    wcol = cols_b.get("well_id")
    ccol = cols_b.get("bin_id")
    if wcol is None or ccol is None:
        return [], {"error": "bins_csv_missing_columns", "found": list(b.columns), "bins_csv": str(bins_csv)}

    b = b.rename(columns={wcol: "well_id", ccol: "bin_id"})
    b["well_id"] = b["well_id"].astype(str)
    b["bin_id"] = b["bin_id"].astype(str).str.strip()

    n_bins0 = int(len(b))
    b = b.drop_duplicates(["well_id", "bin_id"]).reset_index(drop=True)

    g = _read_parquet_any(wells_gr)
    if "well_id" not in g.columns:
        # case-insensitive fallback
        cols_g0 = {c.lower(): c for c in g.columns}
        w = cols_g0.get("well_id")
        if w is None:
            return [], {"error": "wells_gr_missing_well_id", "found": list(g.columns), "wells_gr": str(wells_gr)}
        g = g.rename(columns={w: "well_id"})

    g["well_id"] = g["well_id"].astype(str)
    cols_g = {c.lower(): c for c in g.columns}

    # lat/lon tolerant
    latc = cols_g.get("lat") or cols_g.get("latitude")
    lonc = cols_g.get("lon") or cols_g.get("longitude")
    if latc is None or lonc is None:
        return [], {"error": "wells_gr_missing_latlon", "found": list(g.columns), "wells_gr": str(wells_gr)}

    g_lat = pd.to_numeric(g[latc], errors="coerce")
    g_lon = pd.to_numeric(g[lonc], errors="coerce")
    g = g.assign(lat=g_lat, lon=g_lon).dropna(subset=["lat", "lon"]).copy()

    # Prefer url; otherwise fall back to basename of las_path if available
    urlc = cols_g.get("url")
    laspc = cols_g.get("las_path")

    if urlc is None:
        if laspc is not None:
            g["url"] = g[laspc].astype(str).map(lambda x: Path(x).name if str(x).strip() else "")
        else:
            g["url"] = ""
    else:
        g["url"] = g[urlc].astype(str)

    # Optional identifiers if present
    for want in ["kgs_id", "api", "api_num_nodash", "operator", "lease"]:
        c = cols_g.get(want)
        if c is None:
            g[want] = ""
        else:
            g[want] = g[c].astype(str)

    g2 = g[["well_id", "url", "kgs_id", "api", "api_num_nodash", "operator", "lease", "lat", "lon"]].copy()
    j = b.merge(g2, on="well_id", how="left")
    ok = j["lat"].notna() & j["lon"].notna()
    n_missing_join = int((~ok).sum())
    j = j[ok].copy()

    # bins.csv bin_id becomes the cell_id
    j["cell_id"] = j["bin_id"].astype(str).str.strip()
    j["cell_tag"] = cell_tag
    j["grid_km"] = f"{float(grid_km):g}"
    j["grid_m"] = f"{float(grid_m):.3f}"

    # Optional fields not computed here
    j["proj_method"] = ""
    j["proj_epsg"] = ""
    j["x_m"] = ""
    j["y_m"] = ""

    ij = j["cell_id"].map(lambda s: parse_grid_cell_id(str(s)))
    j["grid_ix"] = ij.map(lambda t: str(int(t[0])) if t is not None else "")
    j["grid_iy"] = ij.map(lambda t: str(int(t[1])) if t is not None else "")
    j["grid_cell"] = j["cell_id"]

    out = j[WELL_TO_CELL_HEADER].copy()
    out["lat"] = out["lat"].map(lambda x: f"{float(x):.8f}" if x == x else "")
    out["lon"] = out["lon"].map(lambda x: f"{float(x):.8f}" if x == x else "")

    rows_out = out.to_dict(orient="records")

    diag = {
        "bins_csv": str(bins_csv),
        "wells_gr": str(wells_gr),
        "grid_km": float(grid_km),
        "grid_m": float(grid_m),
        "cell_tag": cell_tag,
        "bins_rows_raw": int(n_bins0),
        "bins_rows_dedup": int(len(b)),
        "unique_wells_in_bins": int(b["well_id"].nunique()),
        "rows_written": int(len(rows_out)),
        "rows_missing_join": int(n_missing_join),
    }
    return rows_out, diag


def ensure_well_to_cell_csv(
    *,
    well_to_cell_csv: Path,
    manifest_json: Optional[Path],
    grid_km: float,
    force_rebuild: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    """
    Ensure well_to_cell.csv exists for the GRID workflow.

    If missing (or rebuild forced), it is rebuilt from:
      - Step1 bins.csv
      - Step0 wells_gr.parquet

    Grid only. No H3 behavior.
    """
    well_to_cell_csv = Path(well_to_cell_csv)
    if not force_rebuild and well_to_cell_csv.exists():
        return {"built": False, "path": str(well_to_cell_csv)}

    rows, diag = build_well_to_cell_rows_grid(
        well_to_cell_csv=well_to_cell_csv,
        manifest_json=manifest_json,
        grid_km_hint=float(grid_km),
    )

    if not rows:
        return {"built": False, "path": str(well_to_cell_csv), "diagnostics": diag}

    if not dry_run:
        well_to_cell_csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(well_to_cell_csv, WELL_TO_CELL_HEADER, rows)

    return {"built": True, "path": str(well_to_cell_csv), "diagnostics": diag}


def load_or_build_well_to_cell_rows(
    *,
    well_to_cell_csv: Path,
    manifest_json: Optional[Path],
    grid_km: float,
    force_rebuild: bool,
    dry_run: bool,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """
    Dry-run-safe loader:
      - If CSV exists and not force_rebuild: read from file.
      - Else: build from bins.csv + wells_gr; write unless dry_run; return rows.

    Returns:
      rows (list of dict[str,str]),
      diag (includes built/source + diagnostics if built)
    """
    well_to_cell_csv = Path(well_to_cell_csv)

    if (not force_rebuild) and well_to_cell_csv.exists():
        return read_csv_rows(well_to_cell_csv), {"built": False, "path": str(well_to_cell_csv), "source": "file"}

    rows, diag = build_well_to_cell_rows_grid(
        well_to_cell_csv=well_to_cell_csv,
        manifest_json=manifest_json,
        grid_km_hint=float(grid_km),
    )
    if not rows:
        return [], {"built": False, "path": str(well_to_cell_csv), "source": "build_failed", "diagnostics": diag}

    if not dry_run:
        well_to_cell_csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(well_to_cell_csv, WELL_TO_CELL_HEADER, rows)
        return read_csv_rows(well_to_cell_csv), {"built": True, "path": str(well_to_cell_csv), "source": "rebuilt_file", "diagnostics": diag}

    rows_str: List[Dict[str, str]] = [{k: ("" if v is None else str(v)) for k, v in r.items()} for r in rows]
    return rows_str, {"built": True, "path": str(well_to_cell_csv), "source": "in_memory", "diagnostics": diag}
