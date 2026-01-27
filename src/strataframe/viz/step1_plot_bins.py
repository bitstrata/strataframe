# src/strataframe/viz/step1_plot_bins.py
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
# IO helpers
# -----------------------------

def _read_bins_meta(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            bin_id = (r.get("bin_id", "") or "").strip()
            if not bin_id:
                continue

            def _to_int(x: str, d: int = 0) -> int:
                try:
                    return int(float((x or "").strip() or d))
                except Exception:
                    return d

            def _to_float(x: str, d: float = float("nan")) -> float:
                try:
                    return float((x or "").strip() or d)
                except Exception:
                    return d

            rows.append(
                {
                    "bin_id": bin_id,
                    "n_wells": _to_int(r.get("n_wells", "0"), 0),
                    "centroid_lat": _to_float(r.get("centroid_lat", "nan")),
                    "centroid_lon": _to_float(r.get("centroid_lon", "nan")),
                    "radius_km": _to_float(r.get("radius_km", "nan")),  # may be missing/NaN
                    "_raw": dict(r),
                }
            )
    return rows


def _load_manifest(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _load_bin_sources_from_manifest(manifest: Optional[Dict[str, Any]]) -> Optional[Dict[str, List[str]]]:
    """
    Expected shape (H3 legacy; also usable for grid merges):
      manifest["diagnostics"]["bin_sources"] = {
        bin_id: {"source_cells":[...], ...}, ...
      }
    """
    if not manifest:
        return None
    diag = manifest.get("diagnostics", {}) if isinstance(manifest, dict) else {}
    src = diag.get("bin_sources", {}) if isinstance(diag, dict) else {}
    out: Dict[str, List[str]] = {}

    if not isinstance(src, dict):
        return None

    for bin_id, payload in src.items():
        if not isinstance(payload, dict):
            continue
        cells = payload.get("source_cells", [])
        if isinstance(cells, list) and cells:
            out[str(bin_id)] = [str(c) for c in cells if str(c).strip()]
    return out or None


def _read_cell_rowcol_map(path: Path) -> Dict[str, Tuple[int, int]]:
    """
    Read Step1 well_to_cell.csv and build mapping:
      cell_id (or bin_id / h3_cell) -> (row, col)

    Supports columns commonly seen across iterations:
      - cell_id, cell_row, cell_col
      - bin_id, cell_row, cell_col
      - h3_cell, cell_row, cell_col (rare)
      - grid_ix/grid_iy or cell_col/cell_row

    Returns empty dict if file missing/unreadable.
    """
    if not path.exists():
        return {}

    out: Dict[str, Tuple[int, int]] = {}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            cid = (r.get("cell_id", "") or "").strip()
            if not cid:
                cid = (r.get("bin_id", "") or "").strip()
            if not cid:
                cid = (r.get("h3_cell", "") or "").strip()
            if not cid:
                continue

            rr = (r.get("cell_row", "") or r.get("grid_iy", "") or "").strip()
            cc = (r.get("cell_col", "") or r.get("grid_ix", "") or "").strip()
            if not rr or not cc:
                continue

            try:
                row = int(float(rr))
                col = int(float(cc))
            except Exception:
                continue

            # first win is fine; many rows share same cell
            if cid not in out:
                out[cid] = (row, col)

    return out


# -----------------------------
# Projection helpers (local XY in km)
# -----------------------------

def _lonlat_to_xy_km(lon: float, lat: float, lat0: float) -> Tuple[float, float]:
    # simple local projection: x ~ lon*cos(lat0), y ~ lat
    deg_to_km = 111.32
    x = float(lon) * math.cos(math.radians(lat0)) * deg_to_km
    y = float(lat) * deg_to_km
    return x, y


# -----------------------------
# Method detection
# -----------------------------

_HEX_RE = re.compile(r"^[0-9a-fA-F]{10,20}$")


def _looks_like_h3_cell(s: str) -> bool:
    t = (s or "").strip()
    return bool(t and _HEX_RE.match(t))


def _detect_method(*, manifest: Optional[Dict[str, Any]], meta: List[Dict[str, Any]], forced: str) -> str:
    f = (forced or "").strip().lower()
    if f in ("h3", "grid", "auto"):
        if f != "auto":
            return f
    else:
        f = "auto"

    # Prefer manifest params/diagnostics if present
    if manifest:
        params = manifest.get("params", {}) if isinstance(manifest.get("params", {}), dict) else {}
        diag = manifest.get("diagnostics", {}) if isinstance(manifest.get("diagnostics", {}), dict) else {}

        for d in (params, diag):
            m = str(d.get("method", "") or "").strip().lower()
            if m in ("h3", "grid"):
                return m

    # Fallback heuristic: bin_id looks like H3 hex
    if not meta:
        return "grid"
    n = len(meta)
    n_h3 = sum(1 for r in meta if _looks_like_h3_cell(str(r.get("bin_id", ""))))
    return "h3" if n_h3 >= max(1, int(0.8 * n)) else "grid"


# -----------------------------
# Grid cell parsing + geometry (projected grid preferred)
# -----------------------------

# Common grid cell id patterns:
#   g<col>_<row>   (preferred convention across recent scripts)
#   r<row>_c<col>
#   row12_col34
# We return (row, col).
_G_GID_RE = re.compile(r"(?i)^\s*g\s*(-?\d+)\s*[_:,;\s]\s*(-?\d+)\s*$")
_RC_RE = re.compile(r"(?i)(?:^|[^a-z])(?:r|row)\s*[:=_-]?\s*(-?\d+).*?(?:c|col)\s*[:=_-]?\s*(-?\d+)")
_CR_RE = re.compile(r"(?i)(?:^|[^a-z])(?:c|col)\s*[:=_-]?\s*(-?\d+).*?(?:r|row)\s*[:=_-]?\s*(-?\d+)")
_INT_RE = re.compile(r"(-?\d+)")


def _parse_grid_row_col(cell_id: str) -> Optional[Tuple[int, int]]:
    """
    Parse (row, col) from a grid cell id.

    Supports:
      - g<col>_<row> (preferred) => returns (row, col)
      - r12_c34 / row12_col34 / r:12 c:34
      - c34_r12 (swapped)
      - fallback: first two integers; assumes g-style (col,row) if string starts with 'g', else (row,col)
    """
    s = (cell_id or "").strip()
    if not s:
        return None

    mg = _G_GID_RE.match(s)
    if mg:
        try:
            col = int(mg.group(1))
            row = int(mg.group(2))
            return row, col
        except Exception:
            return None

    m = _RC_RE.search(s)
    if m:
        try:
            return int(m.group(1)), int(m.group(2))
        except Exception:
            pass

    m = _CR_RE.search(s)
    if m:
        try:
            # c.. then r.. => return (row, col)
            return int(m.group(2)), int(m.group(1))
        except Exception:
            pass

    nums = _INT_RE.findall(s)
    if len(nums) >= 2:
        try:
            a = int(nums[0])
            b = int(nums[1])
        except Exception:
            return None
        # Heuristic: g-like ids are (col,row)
        if s.lstrip().lower().startswith("g"):
            return b, a
        return a, b

    return None


def _grid_from_manifest(manifest: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Expected (projected grid binning):
      manifest["diagnostics"]["grid"] = {
        "crs_epsg": 5070,
        "cell_m": ...,
        "origin_x_m": ...,
        "origin_y_m": ...,
        ...
      }
    """
    if not manifest:
        return {}
    diag = manifest.get("diagnostics", {})
    if not isinstance(diag, dict):
        return {}
    grid = diag.get("grid", {})
    return grid if isinstance(grid, dict) else {}


def _get_grid_params(grid: Dict[str, Any]) -> Tuple[Optional[int], Optional[float], Optional[float], Optional[float]]:
    """
    Returns: epsg, cell_m, origin_x_m, origin_y_m
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
    return epsg, cell_m, origin_x_m, origin_y_m


def _infer_cell_km_from_manifest_or_spacing(
    centers_xy: List[Tuple[float, float]],
    *,
    manifest: Optional[Dict[str, Any]],
    user_cell_km: Optional[float],
) -> Optional[float]:
    # 1) CLI override
    if user_cell_km is not None and math.isfinite(float(user_cell_km)) and float(user_cell_km) > 0:
        return float(user_cell_km)

    # 2) manifest params/diagnostics
    if manifest:
        params = manifest.get("params", {}) if isinstance(manifest.get("params", {}), dict) else {}
        for k in ("grid_cell_km", "cell_km", "grid_cell_size_km"):
            v = params.get(k, None)
            try:
                if v is None:
                    continue
                fv = float(v)
                if math.isfinite(fv) and fv > 0:
                    return fv
            except Exception:
                pass

        diag = manifest.get("diagnostics", {}) if isinstance(manifest.get("diagnostics", {}), dict) else {}
        grid = diag.get("grid", {}) if isinstance(diag.get("grid", {}), dict) else {}
        for k in ("grid_cell_km", "cell_km", "grid_cell_size_km"):
            v = grid.get(k, None)
            try:
                if v is None:
                    continue
                fv = float(v)
                if math.isfinite(fv) and fv > 0:
                    return fv
            except Exception:
                pass

        # common: cell_m (convert to km)
        try:
            cell_m = grid.get("cell_m", None) if isinstance(grid, dict) else None
            if cell_m is not None:
                fm = float(cell_m)
                if math.isfinite(fm) and fm > 0:
                    return fm / 1000.0
        except Exception:
            pass

    # 3) infer from centroid NN spacing
    if len(centers_xy) < 2:
        return None

    pts = np.asarray(centers_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return None

    dmins: List[float] = []
    for i in range(pts.shape[0]):
        dx = pts[:, 0] - pts[i, 0]
        dy = pts[:, 1] - pts[i, 1]
        d = np.sqrt(dx * dx + dy * dy)
        d[i] = np.inf
        m = float(np.min(d))
        if math.isfinite(m) and m > 0:
            dmins.append(m)

    if not dmins:
        return None

    return float(np.median(np.asarray(dmins, dtype=float)))


def _grid_cell_ring_lonlat_from_manifest(
    *,
    row: int,
    col: int,
    epsg: int,
    cell_m: float,
    origin_x_m: float,
    origin_y_m: float,
) -> Optional[List[Tuple[float, float]]]:
    """
    Build (lon,lat) ring for a projected grid cell using manifest origin/cell_m.
    Requires pyproj. Returns None if pyproj unavailable.
    """
    try:
        import pyproj  # type: ignore
    except Exception:
        return None

    tf = pyproj.Transformer.from_crs(int(epsg), 4326, always_xy=True)

    x0 = float(origin_x_m + float(col) * float(cell_m))
    x1 = float(origin_x_m + (float(col) + 1.0) * float(cell_m))
    y0 = float(origin_y_m + float(row) * float(cell_m))
    y1 = float(origin_y_m + (float(row) + 1.0) * float(cell_m))

    xs = [x0, x1, x1, x0, x0]
    ys = [y0, y0, y1, y1, y0]

    lons, lats = tf.transform(xs, ys)
    ring = [(float(lon), float(lat)) for lon, lat in zip(lons, lats)]
    return ring


def _rect_ring_xy_from_center(x: float, y: float, *, cell_km: float) -> List[Tuple[float, float]]:
    h = 0.5 * float(cell_km)
    return [
        (x - h, y - h),
        (x + h, y - h),
        (x + h, y + h),
        (x - h, y + h),
        (x - h, y - h),
    ]


def _circle_xy(x: float, y: float, r_km: float, n: int = 64) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    rr = float(r_km)
    for k in range(n + 1):
        t = (2.0 * math.pi) * (float(k) / float(n))
        out.append((x + rr * math.cos(t), y + rr * math.sin(t)))
    return out


# -----------------------------
# Main
# -----------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Plot Step 1 bins as a map (grid-first; retains H3 support).")
    ap.add_argument("--bins-meta", type=str, required=True, help="Path to bins_meta.csv")
    ap.add_argument("--manifest", type=str, default="", help="Path to manifest.json (optional; improves method detection / merges)")
    ap.add_argument("--out", type=str, default="", help="Output PNG path (default: alongside bins_meta.csv)")
    ap.add_argument("--geojson-out", type=str, default="", help="Optional: write bins_cells.geojson for QGIS")

    ap.add_argument("--title", type=str, default="Step 1 — Spatial Bins", help="Plot title")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--cmap", type=str, default="", help="Optional matplotlib colormap name (default: matplotlib default)")
    ap.add_argument("--label-min-wells", type=int, default=0, help="Label bins with n_wells if >= this value (0 disables)")

    # Execution
    ap.add_argument("--show", action="store_true", help="Show interactive window (requires GUI backend)")

    # Method control
    ap.add_argument("--method", type=str, default="auto", help="grid | auto | h3 (default auto)")

    # Grid supports
    ap.add_argument("--grid-cell-km", type=float, default=float("nan"), help="Grid cell size (km). Optional; prefers manifest.")
    ap.add_argument(
        "--well-to-cell",
        type=str,
        default="",
        help="Optional path to Step1 well_to_cell.csv; if omitted, tries alongside bins_meta.csv",
    )

    # Optional overlays (best-effort; safe to ignore if deps missing)
    ap.add_argument("--kansas-outline", action="store_true", help="Attempt to draw Kansas state outline (cartopy, optional)")
    ap.add_argument("--draw-radius", action="store_true", help="Draw per-bin radius circles using radius_km (if present)")

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

    bins_meta_path = Path(args.bins_meta)
    if not bins_meta_path.exists():
        raise SystemExit(f"Not found: {bins_meta_path}")

    out_png = Path(args.out) if str(args.out).strip() else (bins_meta_path.parent / "bins_map.png")
    manifest_path = Path(args.manifest) if str(args.manifest).strip() else (bins_meta_path.parent / "manifest.json")
    manifest = _load_manifest(manifest_path)

    meta = _read_bins_meta(bins_meta_path)
    if not meta:
        raise SystemExit("bins_meta.csv appears empty (no rows with bin_id).")

    method = _detect_method(manifest=manifest, meta=meta, forced=str(args.method))
    bin_sources = _load_bin_sources_from_manifest(manifest)

    # lat0 for local projection (mean centroid lat)
    lat_vals = [float(r["centroid_lat"]) for r in meta if math.isfinite(float(r["centroid_lat"]))]
    if not lat_vals:
        raise SystemExit("No finite centroid_lat values found in bins_meta.csv.")
    lat0 = float(sum(lat_vals) / float(len(lat_vals)))

    # Plot scaffolding
    patches: List[Any] = []
    vals: List[float] = []
    geo_features: List[Dict[str, Any]] = []

    # Centroid scatter / labels / radius circles
    cx: List[float] = []
    cy: List[float] = []
    cs: List[float] = []
    clabels: List[Tuple[float, float, str]] = []
    radius_rings: List[List[Tuple[float, float]]] = []

    # Kansas outline (optional)
    kansas_geom = None
    if bool(args.kansas_outline):
        try:
            import cartopy.io.shapereader as shpreader  # type: ignore

            shp = shpreader.natural_earth(
                resolution="10m",
                category="cultural",
                name="admin_1_states_provinces",
            )
            reader = shpreader.Reader(shp)
            for rec in reader.records():
                attrs = rec.attributes or {}
                name = (attrs.get("name") or attrs.get("name_en") or "").strip()
                admin = (attrs.get("admin") or "").strip()
                if name == "Kansas" and (admin == "" or admin == "United States of America"):
                    kansas_geom = rec.geometry
                    break
        except Exception:
            kansas_geom = None

    # -------------------------
    # H3 mode
    # -------------------------
    if method == "h3":
        from strataframe.spatial.h3_utils import cell_to_boundary, require_h3  # type: ignore

        require_h3()

        # Build patches per (bin -> source cells) so merged bins render properly.
        for r in meta:
            bin_id = str(r["bin_id"])
            n_wells = float(r["n_wells"])
            cells = bin_sources.get(bin_id, [bin_id]) if bin_sources else [bin_id]

            for cell in cells:
                boundary_latlon = cell_to_boundary(cell)  # [(lat, lon), ...]
                ring_xy = [_lonlat_to_xy_km(lon, lat, lat0) for (lat, lon) in boundary_latlon]
                if ring_xy and ring_xy[0] != ring_xy[-1]:
                    ring_xy.append(ring_xy[0])

                patches.append(Polygon(ring_xy, closed=True))
                vals.append(n_wells)

                # GeoJSON (lon,lat)
                ring_ll = [(float(lon), float(lat)) for (lat, lon) in boundary_latlon]
                if ring_ll and ring_ll[0] != ring_ll[-1]:
                    ring_ll.append(ring_ll[0])

                if str(args.geojson_out).strip():
                    geo_features.append(
                        {
                            "type": "Feature",
                            "properties": {
                                "bin_id": bin_id,
                                "h3_cell": str(cell),
                                "n_wells": int(n_wells),
                                "method": "h3",
                            },
                            "geometry": {"type": "Polygon", "coordinates": [[list(p) for p in ring_ll]]},
                        }
                    )

        for r in meta:
            lat = float(r["centroid_lat"])
            lon = float(r["centroid_lon"])
            if not (math.isfinite(lat) and math.isfinite(lon)):
                continue
            x, y = _lonlat_to_xy_km(lon, lat, lat0)
            cx.append(x)
            cy.append(y)
            cs.append(12.0 + 6.0 * math.sqrt(max(0.0, float(r["n_wells"]))))
            if int(args.label_min_wells) > 0 and int(r["n_wells"]) >= int(args.label_min_wells):
                clabels.append((x, y, str(int(r["n_wells"]))))

            if bool(args.draw_radius):
                rk = float(r.get("radius_km", float("nan")))
                if math.isfinite(rk) and rk > 0:
                    radius_rings.append(_circle_xy(x, y, rk, n=72))

    # -------------------------
    # Grid mode (default)
    # -------------------------
    else:
        # Optional well_to_cell mapping (preferred for correct grid geometry)
        if str(args.well_to_cell).strip():
            w2c_path = Path(args.well_to_cell)
        else:
            w2c_path = bins_meta_path.parent / "well_to_cell.csv"
        cell_rowcol = _read_cell_rowcol_map(w2c_path) if w2c_path.exists() else {}

        centers_xy: List[Tuple[float, float]] = []
        for r in meta:
            lat = float(r["centroid_lat"])
            lon = float(r["centroid_lon"])
            if not (math.isfinite(lat) and math.isfinite(lon)):
                continue
            centers_xy.append(_lonlat_to_xy_km(lon, lat, lat0))

        user_cell_km = None
        try:
            if math.isfinite(float(args.grid_cell_km)) and float(args.grid_cell_km) > 0:
                user_cell_km = float(args.grid_cell_km)
        except Exception:
            user_cell_km = None

        cell_km = _infer_cell_km_from_manifest_or_spacing(centers_xy, manifest=manifest, user_cell_km=user_cell_km)
        if cell_km is None or not math.isfinite(cell_km) or cell_km <= 0:
            raise SystemExit(
                "Grid plotting requires a grid cell size. "
                "Provide --grid-cell-km or ensure manifest.json includes diagnostics.grid.cell_m (or params.grid_cell_km)."
            )

        # Prefer exact projected grid geometry if available
        grid = _grid_from_manifest(manifest)
        epsg, cell_m, origin_x_m, origin_y_m = _get_grid_params(grid)
        have_proj_grid = (
            epsg is not None
            and cell_m is not None
            and origin_x_m is not None
            and origin_y_m is not None
        )

        # Centroid scatter / labels / radius
        for r in meta:
            lat = float(r["centroid_lat"])
            lon = float(r["centroid_lon"])
            if not (math.isfinite(lat) and math.isfinite(lon)):
                continue
            x, y = _lonlat_to_xy_km(lon, lat, lat0)
            cx.append(x)
            cy.append(y)
            cs.append(12.0 + 6.0 * math.sqrt(max(0.0, float(r["n_wells"]))))
            if int(args.label_min_wells) > 0 and int(r["n_wells"]) >= int(args.label_min_wells):
                clabels.append((x, y, str(int(r["n_wells"]))))

            if bool(args.draw_radius):
                rk = float(r.get("radius_km", float("nan")))
                if math.isfinite(rk) and rk > 0:
                    radius_rings.append(_circle_xy(x, y, rk, n=72))

        # Draw rectangles per bin (and per contributing source cell if available)
        for r in meta:
            bin_id = str(r["bin_id"])
            n_wells = float(r["n_wells"])
            cells = bin_sources.get(bin_id, [bin_id]) if bin_sources else [bin_id]

            drew_any = False
            for cell in cells:
                # row/col from well_to_cell mapping preferred; else parse
                rc = cell_rowcol.get(str(cell)) if cell_rowcol else None
                if rc is None:
                    rc = _parse_grid_row_col(str(cell))
                if rc is None:
                    continue

                row, col = rc

                ring_ll: Optional[List[Tuple[float, float]]] = None
                if have_proj_grid:
                    ring_ll = _grid_cell_ring_lonlat_from_manifest(
                        row=int(row),
                        col=int(col),
                        epsg=int(epsg),
                        cell_m=float(cell_m),
                        origin_x_m=float(origin_x_m),
                        origin_y_m=float(origin_y_m),
                    )

                if ring_ll is not None:
                    # Plot in local XY km using transformed lon/lat corners (accurate geometry)
                    ring_xy = [_lonlat_to_xy_km(lon, lat, lat0) for (lon, lat) in ring_ll]
                else:
                    # Fallback: build in local XY using cell_km around inferred center position
                    # If centroid exists for this bin, anchor the single-cell case to it; otherwise skip.
                    # For merged bins without manifest geometry, this is approximate but consistent.
                    # (Still better than silently omitting.)
                    # We place the square centered at the bin centroid if available.
                    latc = float(r.get("centroid_lat", float("nan")))
                    lonc = float(r.get("centroid_lon", float("nan")))
                    if not (math.isfinite(latc) and math.isfinite(lonc)):
                        continue
                    xc, yc = _lonlat_to_xy_km(lonc, latc, lat0)
                    ring_xy = _rect_ring_xy_from_center(xc, yc, cell_km=float(cell_km))
                    ring_ll = None  # unknown / approximate

                patches.append(plt_polygon := None)  # placeholder to keep structure identical in branches
                # Replace placeholder with actual Polygon now that we have ring_xy
                from matplotlib.patches import Polygon  # type: ignore

                patches[-1] = Polygon(ring_xy, closed=True)
                vals.append(n_wells)

                if str(args.geojson_out).strip():
                    if ring_ll is None:
                        # Approximate GeoJSON via ring_xy inversion is not reliable; still export centroid square in lon/lat.
                        # Use bin centroid with cell_km as a visual proxy.
                        latc = float(r.get("centroid_lat", float("nan")))
                        lonc = float(r.get("centroid_lon", float("nan")))
                        if math.isfinite(latc) and math.isfinite(lonc):
                            # Make a small local square in lon/lat space (approx, but usable for quick QGIS sanity checks)
                            # Convert km to degrees at lat0
                            dlat = (0.5 * float(cell_km)) / 111.32
                            dlon = (0.5 * float(cell_km)) / (111.32 * max(1e-12, math.cos(math.radians(lat0))))
                            ring_ll2 = [
                                (lonc - dlon, latc - dlat),
                                (lonc + dlon, latc - dlat),
                                (lonc + dlon, latc + dlat),
                                (lonc - dlon, latc + dlat),
                                (lonc - dlon, latc - dlat),
                            ]
                        else:
                            ring_ll2 = []
                    else:
                        ring_ll2 = ring_ll

                    if ring_ll2:
                        geo_features.append(
                            {
                                "type": "Feature",
                                "properties": {
                                    "bin_id": bin_id,
                                    "grid_cell": str(cell),
                                    "n_wells": int(n_wells),
                                    "method": "grid",
                                },
                                "geometry": {"type": "Polygon", "coordinates": [[list(p) for p in ring_ll2]]},
                            }
                        )

                drew_any = True

            if drew_any:
                continue

            # If we can't parse row/col at all, fall back to a centroid square (still conveys counts and locations)
            lat = float(r.get("centroid_lat", float("nan")))
            lon = float(r.get("centroid_lon", float("nan")))
            if not (math.isfinite(lat) and math.isfinite(lon)):
                continue
            x, y = _lonlat_to_xy_km(lon, lat, lat0)
            ring_xy = _rect_ring_xy_from_center(x, y, cell_km=float(cell_km))

            from matplotlib.patches import Polygon  # type: ignore
            patches.append(Polygon(ring_xy, closed=True))
            vals.append(n_wells)

            if str(args.geojson_out).strip():
                # Approximate lon/lat square around centroid
                dlat = (0.5 * float(cell_km)) / 111.32
                dlon = (0.5 * float(cell_km)) / (111.32 * max(1e-12, math.cos(math.radians(lat0))))
                ring_ll3 = [
                    (lon - dlon, lat - dlat),
                    (lon + dlon, lat - dlat),
                    (lon + dlon, lat + dlat),
                    (lon - dlon, lat + dlat),
                    (lon - dlon, lat - dlat),
                ]
                geo_features.append(
                    {
                        "type": "Feature",
                        "properties": {"bin_id": bin_id, "grid_cell": "", "n_wells": int(n_wells), "method": "grid"},
                        "geometry": {"type": "Polygon", "coordinates": [[list(p) for p in ring_ll3]]},
                    }
                )

    # -------------------------
    # Plot
    # -------------------------
    fig, ax = plt.subplots(figsize=(11, 8))

    # Optional Kansas outline on local XY axes (best-effort, approximate): convert outline vertices to local xy
    if kansas_geom is not None:
        try:
            # Kansan geom is in lon/lat; draw exterior rings only
            geoms = []
            if hasattr(kansas_geom, "geoms"):
                geoms = list(kansas_geom.geoms)
            else:
                geoms = [kansas_geom]

            for g in geoms:
                if not hasattr(g, "exterior"):
                    continue
                xs_ll, ys_ll = g.exterior.xy
                ring_xy = [_lonlat_to_xy_km(float(lon), float(lat), lat0) for lon, lat in zip(xs_ll, ys_ll)]
                if ring_xy:
                    ax.plot([p[0] for p in ring_xy], [p[1] for p in ring_xy], linewidth=1.1, zorder=1)
        except Exception:
            pass

    pc = PatchCollection(patches, linewidths=0.6, alpha=0.65)
    pc.set_array(np.asarray(vals, dtype=float))
    if str(args.cmap).strip():
        try:
            pc.set_cmap(str(args.cmap).strip())
        except Exception:
            pass
    ax.add_collection(pc)

    # centroid scatter
    ax.scatter(cx, cy, s=cs, linewidths=0.6, zorder=3)

    # optional radius rings
    if radius_rings:
        for ring in radius_rings:
            ax.plot([p[0] for p in ring], [p[1] for p in ring], linewidth=0.6, alpha=0.6, zorder=2)

    cb = fig.colorbar(pc, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("n_wells per bin")

    title = str(args.title)
    if title.strip() == "Step 1 — Spatial Bins":
        title = f"Step 1 — Spatial Bins ({method})"
    ax.set_title(title)

    ax.set_xlabel("X (km, local projection)")
    ax.set_ylabel("Y (km, local projection)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.autoscale()

    for x, y, txt in clabels:
        ax.text(x, y, txt, fontsize=8, ha="center", va="center", zorder=4)

    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=int(args.dpi))

    if bool(args.show):
        plt.show()

    plt.close(fig)

    if str(args.geojson_out).strip():
        gj_path = Path(args.geojson_out)
        gj_path.parent.mkdir(parents=True, exist_ok=True)
        gj = {"type": "FeatureCollection", "features": geo_features}
        gj_path.write_text(json.dumps(gj), encoding="utf-8")
        print(f"Wrote: {gj_path}")

    print(f"Wrote: {out_png}")

    n_bins = len({r["bin_id"] for r in meta})
    n_cells_drawn = len(geo_features) if str(args.geojson_out).strip() else len(patches)
    print(f"Method: {method} | Bins: {n_bins} | Cells drawn: {n_cells_drawn} | manifest merges: {'yes' if bin_sources else 'no'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
