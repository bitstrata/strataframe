# src/strataframe/viz/step1_plot_grid_only.py
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# -----------------------------
# Parsing / numeric helpers
# -----------------------------

_INT_RE = re.compile(r"(-?\d+)")
_G_GID_RE = re.compile(r"(?i)^\s*g\s*(-?\d+)\s*[_:,;\s]\s*(-?\d+)\s*$")
_RC_RE = re.compile(r"(?i)(?:^|[^a-z])(?:r|row)\s*[:=_-]?\s*(-?\d+).*?(?:c|col)\s*[:=_-]?\s*(-?\d+)")
_CR_RE = re.compile(r"(?i)(?:^|[^a-z])(?:c|col)\s*[:=_-]?\s*(-?\d+).*?(?:r|row)\s*[:=_-]?\s*(-?\d+)")


def _to_float(x: str) -> Optional[float]:
    try:
        v = float((x or "").strip())
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _to_int(x: str) -> Optional[int]:
    try:
        return int((x or "").strip())
    except Exception:
        try:
            return int(float((x or "").strip()))
        except Exception:
            return None


def _parse_row_col_from_id(s: str) -> Optional[Tuple[int, int]]:
    """
    Return (row, col).

    Supports:
      - g<col>_<row> (preferred) => (row, col)
      - r<row>_c<col> etc
      - c<col>_r<row> etc
      - fallback: first two integers; if starts with g assume (col,row), else (row,col)
    """
    t = (s or "").strip()
    if not t:
        return None

    mg = _G_GID_RE.match(t)
    if mg:
        try:
            col = int(mg.group(1))
            row = int(mg.group(2))
            return row, col
        except Exception:
            return None

    m = _RC_RE.search(t)
    if m:
        try:
            return int(m.group(1)), int(m.group(2))
        except Exception:
            pass

    m = _CR_RE.search(t)
    if m:
        try:
            return int(m.group(2)), int(m.group(1))
        except Exception:
            pass

    nums = _INT_RE.findall(t)
    if len(nums) < 2:
        return None
    try:
        a = int(nums[0])
        b = int(nums[1])
    except Exception:
        return None

    if t.lstrip().lower().startswith("g"):
        return b, a
    return a, b


# -----------------------------
# Manifest helpers
# -----------------------------

def _load_manifest(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _grid_from_manifest(manifest: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Expected:
      manifest["diagnostics"]["grid"] = {
        "crs_epsg": 5070,
        "cell_m": ...,
        "origin_x_m": ...,
        "origin_y_m": ...,
        "ix_min": ..., "ix_max": ..., "iy_min": ..., "iy_max": ...,
      }
    """
    if not manifest:
        return {}
    diag = manifest.get("diagnostics", {})
    if not isinstance(diag, dict):
        return {}
    grid = diag.get("grid", {})
    return grid if isinstance(grid, dict) else {}


def _get_grid_params(grid: Dict[str, Any]) -> Tuple[
    Optional[int],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[int],
]:
    """
    Returns:
      epsg, cell_m, origin_x_m, origin_y_m, ix_min, ix_max, iy_min, iy_max
    """
    def _f(k: str) -> Optional[float]:
        v = grid.get(k, None)
        try:
            if v is None:
                return None
            fv = float(v)
            return fv if math.isfinite(fv) else None
        except Exception:
            return None

    def _i(k: str) -> Optional[int]:
        v = grid.get(k, None)
        try:
            if v is None:
                return None
            return int(v)
        except Exception:
            try:
                return int(float(v))
            except Exception:
                return None

    epsg = _i("crs_epsg") or _i("epsg") or _i("proj_epsg")
    cell_m = _f("cell_m") or _f("chosen_cell_m")
    origin_x_m = _f("origin_x_m")
    origin_y_m = _f("origin_y_m")
    ix_min = _i("ix_min")
    ix_max = _i("ix_max")
    iy_min = _i("iy_min")
    iy_max = _i("iy_max")
    return epsg, cell_m, origin_x_m, origin_y_m, ix_min, ix_max, iy_min, iy_max


# -----------------------------
# Read + normalize well_to_cell
# -----------------------------

def _read_rows(path: Path, *, swap_rc: bool) -> List[Dict[str, Any]]:
    """
    Accepts either:
      - explicit cell_row/cell_col/cell_id, OR
      - parseable cell_id/bin_id like g10_6, r10_c6, etc.

    Convention:
      g<col>_<row> => cell_col=col, cell_row=row
      returned internally as cell_row/cell_col.
    """
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            well_id = (r.get("well_id", "") or "").strip()
            if not well_id:
                continue

            cell_id = (r.get("cell_id", "") or "").strip()
            if not cell_id:
                cell_id = (r.get("bin_id", "") or "").strip()

            rr = _to_int(r.get("cell_row", "") or r.get("grid_iy", "") or "")
            cc = _to_int(r.get("cell_col", "") or r.get("grid_ix", "") or "")

            if rr is None or cc is None:
                src = cell_id
                ij = _parse_row_col_from_id(src)
                if ij is None:
                    continue
                row, col = ij
                rr, cc = row, col

            if swap_rc:
                rr, cc = cc, rr

            if not cell_id:
                cell_id = f"g{cc}_{rr}"

            lat = _to_float(r.get("lat", "") or r.get("Latitude", "") or "")
            lon = _to_float(r.get("lon", "") or r.get("Longitude", "") or "")

            out.append(
                {
                    "well_id": well_id,
                    "cell_id": cell_id,
                    "cell_row": int(rr),
                    "cell_col": int(cc),
                    "lat": float(lat) if lat is not None else None,
                    "lon": float(lon) if lon is not None else None,
                }
            )
    return out


# -----------------------------
# Geometry helpers (projected CRS)
# -----------------------------

def _cell_ring_xy_km(
    *,
    origin_x_m: float,
    origin_y_m: float,
    cell_m: float,
    row: int,
    col: int,
) -> List[Tuple[float, float]]:
    """
    Rectangle ring in *km* for cell at (row, col):
      x = origin_x_m + col * cell_m
      y = origin_y_m + row * cell_m
    """
    x_min_m = float(origin_x_m + float(col) * float(cell_m))
    x_max_m = float(origin_x_m + (float(col) + 1.0) * float(cell_m))
    y_min_m = float(origin_y_m + float(row) * float(cell_m))
    y_max_m = float(origin_y_m + (float(row) + 1.0) * float(cell_m))

    x_min = x_min_m / 1000.0
    x_max = x_max_m / 1000.0
    y_min = y_min_m / 1000.0
    y_max = y_max_m / 1000.0

    return [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max), (x_min, y_min)]


# -----------------------------
# Main
# -----------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Plot grid cells only (Step1 grid debug).")
    ap.add_argument("--well-to-cell", type=str, required=True, help="Path to well_to_cell.csv")
    ap.add_argument("--manifest", type=str, default="", help="Optional manifest.json (strongly recommended)")
    ap.add_argument("--out", type=str, default="", help="Output PNG path (default: alongside well_to_cell.csv)")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--pad-cells", type=int, default=2, help="Pad extent by N cells on each side")
    ap.add_argument("--swap-rc", action="store_true", help="Swap row/col mapping (debug)")
    ap.add_argument("--hide-empty", action="store_true", help="Do not draw empty cells (occupied only)")
    ap.add_argument(
        "--space",
        choices=["proj", "index"],
        default="proj",
        help="proj=use manifest origin/cell_m (recommended); index=unit squares in row/col space",
    )
    ap.add_argument("--cell-m", type=float, default=float("nan"), help="Override cell size in meters (proj mode)")
    ap.add_argument("--origin-x-m", type=float, default=float("nan"), help="Override origin_x_m (proj mode)")
    ap.add_argument("--origin-y-m", type=float, default=float("nan"), help="Override origin_y_m (proj mode)")
    ap.add_argument("--extent-from-manifest", action="store_true", help="In proj mode, prefer ix/iy extents from manifest")
    ap.add_argument("--show", action="store_true", help="Show interactive window (requires GUI backend)")
    args = ap.parse_args(argv)

    # Lazy matplotlib import with headless-safe backend selection
    try:
        import matplotlib  # type: ignore

        out_requested = bool(str(args.out).strip())
        if out_requested and not args.show:
            try:
                matplotlib.use("Agg")
            except Exception:
                pass

        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib.collections import PatchCollection  # type: ignore
        from matplotlib.patches import Polygon  # type: ignore
    except Exception as e:
        print("ERROR: matplotlib is required for this visualization.")
        print("Install with: pip install matplotlib")
        print(f"Details: {type(e).__name__}: {e}")
        return 2

    w2c = Path(args.well_to_cell)
    if not w2c.exists():
        raise SystemExit(f"Not found: {w2c}")

    out_png = Path(args.out) if str(args.out).strip() else (w2c.parent / "grid_only.png")

    manifest_path = Path(args.manifest) if str(args.manifest).strip() else (w2c.parent / "manifest.json")
    manifest = _load_manifest(manifest_path)
    grid = _grid_from_manifest(manifest)
    epsg, cell_m, origin_x_m, origin_y_m, ix_min_m, ix_max_m, iy_min_m, iy_max_m = _get_grid_params(grid)

    # CLI overrides (proj mode)
    if math.isfinite(float(args.cell_m)) and float(args.cell_m) > 0:
        cell_m = float(args.cell_m)
    if math.isfinite(float(args.origin_x_m)):
        origin_x_m = float(args.origin_x_m)
    if math.isfinite(float(args.origin_y_m)):
        origin_y_m = float(args.origin_y_m)

    rows = _read_rows(w2c, swap_rc=bool(args.swap_rc))
    if not rows:
        raise SystemExit(
            "No usable rows found. Need well_id and either:\n"
            "  - cell_row/cell_col, or\n"
            "  - parseable cell_id/bin_id like g10_6."
        )

    # Unique wells per cell
    cell_to_wells: Dict[Tuple[int, int], set[str]] = {}
    for r in rows:
        rr = int(r["cell_row"])
        cc = int(r["cell_col"])
        cell_to_wells.setdefault((rr, cc), set()).add(str(r["well_id"]))

    occupied = sorted(cell_to_wells.keys())
    counts = {k: int(len(cell_to_wells[k])) for k in occupied}

    r_vals = [k[0] for k in occupied]
    c_vals = [k[1] for k in occupied]
    rmin_obs, rmax_obs = int(min(r_vals)), int(max(r_vals))
    cmin_obs, cmax_obs = int(min(c_vals)), int(max(c_vals))

    pad = int(args.pad_cells)

    if (
        args.space == "proj"
        and args.extent_from_manifest
        and (ix_min_m is not None)
        and (ix_max_m is not None)
        and (iy_min_m is not None)
        and (iy_max_m is not None)
    ):
        cmin, cmax = int(ix_min_m), int(ix_max_m)
        rmin, rmax = int(iy_min_m), int(iy_max_m)
    else:
        cmin, cmax = cmin_obs, cmax_obs
        rmin, rmax = rmin_obs, rmax_obs

    cmin2, cmax2 = cmin - pad, cmax + pad
    rmin2, rmax2 = rmin - pad, rmax + pad

    if args.space == "proj":
        if cell_m is None or origin_x_m is None or origin_y_m is None:
            raise SystemExit(
                "proj mode requires manifest diagnostics.grid with origin_x_m/origin_y_m + cell_m.\n"
                "Either provide --manifest (preferred) or switch to --space index.\n"
                "You can also override with --cell-m --origin-x-m --origin-y-m."
            )

    patches: List[Any] = []
    vals: List[float] = []

    for rr in range(rmin2, rmax2 + 1):
        for cc in range(cmin2, cmax2 + 1):
            n = float(counts.get((rr, cc), 0))
            if args.hide_empty and n <= 0:
                continue

            if args.space == "index":
                ring = [(cc, rr), (cc + 1, rr), (cc + 1, rr + 1), (cc, rr + 1), (cc, rr)]
            else:
                ring = _cell_ring_xy_km(
                    origin_x_m=float(origin_x_m),
                    origin_y_m=float(origin_y_m),
                    cell_m=float(cell_m),
                    row=int(rr),
                    col=int(cc),
                )

            patches.append(Polygon(ring, closed=True))
            vals.append(n)

    fig, ax = plt.subplots(figsize=(11, 8))
    pc = PatchCollection(patches, linewidths=0.25, alpha=0.95)
    pc.set_array(np.asarray(vals, dtype=float))
    ax.add_collection(pc)

    cb = fig.colorbar(pc, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("unique wells per cell")

    if args.space == "index":
        ax.set_title("Grid only (index space)")
        ax.set_xlabel("cell_col")
        ax.set_ylabel("cell_row")
    else:
        cell_km = float(cell_m) / 1000.0
        ax.set_title(f"Grid only (projected; EPSG:{epsg if epsg is not None else 'unknown'}) | cell={cell_km:.3f} km")
        ax.set_xlabel("X (km, projected)")
        ax.set_ylabel("Y (km, projected)")

    ax.set_aspect("equal", adjustable="datalim")
    ax.autoscale()
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=int(args.dpi))

    if bool(args.show):
        plt.show()

    plt.close(fig)

    n_rows = (rmax2 - rmin2 + 1)
    n_cols = (cmax2 - cmin2 + 1)
    print(f"Wrote: {out_png}")
    print(f"Rows read: {len(rows)} | unique wells: {len({r['well_id'] for r in rows})}")
    print(f"Occupied cells: {len(occupied)} | extent rows={n_rows} cols={n_cols} (pad={pad})")
    if args.space == "proj":
        print(f"EPSG: {epsg} | cell_m={float(cell_m):.3f} | origin_x_m={float(origin_x_m):.3f} | origin_y_m={float(origin_y_m):.3f}")
        if args.extent_from_manifest:
            print("Extent source: manifest (if present) else observed")
    if bool(args.swap_rc):
        print("NOTE: swap_rc=True")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
