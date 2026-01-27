# src/strataframe/viz/step0_kansas_map.py
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# NOTE:
# - matplotlib is intentionally NOT imported at module import time to avoid hard failures
#   in minimal environments. It is required at runtime (main()).


# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------

def _read_parquet_any(path: Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    if p.is_dir():
        # Prefer pyarrow.dataset when available for partitioned datasets
        try:
            import pyarrow.dataset as ds  # type: ignore
            return ds.dataset(str(p), format="parquet").to_table().to_pandas()
        except Exception:
            parts = sorted([x for x in p.glob("*.parquet") if x.is_file()])
            if not parts:
                raise ValueError(f"No parquet part files found in directory: {p}")
            return pd.concat([pd.read_parquet(x) for x in parts], ignore_index=True)

    return pd.read_parquet(p)


def _load_wells_gr(path: Path) -> pd.DataFrame:
    df = _read_parquet_any(path) if (path.is_dir() or path.suffix.lower() == ".parquet") else pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    lat_col = cols.get("lat")
    lon_col = cols.get("lon")
    if lat_col is None or lon_col is None:
        raise ValueError(f"wells_gr missing lat/lon columns. Found: {list(df.columns)}")

    out = pd.DataFrame(
        {
            "lat": pd.to_numeric(df[lat_col], errors="coerce"),
            "lon": pd.to_numeric(df[lon_col], errors="coerce"),
        }
    )
    out = out[np.isfinite(out["lat"].values) & np.isfinite(out["lon"].values)].copy()
    return out


def _load_reps_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    lat_col = cols.get("lat")
    lon_col = cols.get("lon")
    if lat_col is None or lon_col is None:
        raise ValueError(f"representatives.csv missing lat/lon columns. Found: {list(df.columns)}")

    out = pd.DataFrame(
        {
            "lat": pd.to_numeric(df[lat_col], errors="coerce"),
            "lon": pd.to_numeric(df[lon_col], errors="coerce"),
        }
    )
    out = out[np.isfinite(out["lat"].values) & np.isfinite(out["lon"].values)].copy()
    return out


def _load_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        return {"_raw": obj}
    return obj


# -----------------------------------------------------------------------------
# Kansas outline + extent
# -----------------------------------------------------------------------------

def _get_kansas_geom():
    """
    Returns (kansas_geom, cartopy.crs module) or (None, None) if cartopy unavailable.
    """
    try:
        import cartopy.crs as ccrs  # type: ignore
        import cartopy.io.shapereader as shpreader  # type: ignore
    except Exception:
        return None, None

    shp = shpreader.natural_earth(
        resolution="10m",
        category="cultural",
        name="admin_1_states_provinces",
    )
    reader = shpreader.Reader(shp)

    kansas_geom = None
    for rec in reader.records():
        attrs = rec.attributes or {}
        name = (attrs.get("name") or attrs.get("name_en") or "").strip()
        admin = (attrs.get("admin") or "").strip()
        if name == "Kansas" and (admin == "" or admin == "United States of America"):
            kansas_geom = rec.geometry
            break

    return kansas_geom, ccrs


def _kansas_bbox_fallback() -> Tuple[float, float, float, float]:
    # Rough KS bounds
    return (-102.2, -94.4, 36.8, 40.2)  # lon_min, lon_max, lat_min, lat_max


def _extent_from_geom(geom, pad_frac: float = 0.03) -> Tuple[float, float, float, float]:
    minx, miny, maxx, maxy = geom.bounds
    dx = maxx - minx
    dy = maxy - miny
    return (minx - dx * pad_frac, maxx + dx * pad_frac, miny - dy * pad_frac, maxy + dy * pad_frac)


# -----------------------------------------------------------------------------
# Grid extraction (legacy Step 1 grid support)
# -----------------------------------------------------------------------------

_GCELL_RE = re.compile(r"^g(-?\d+)_(-?\d+)$")


def _to_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return int(x)
        return int(float(x))
    except Exception:
        return None


def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _extract_grid_params_from_manifest(obj: Any) -> Dict[str, Any]:
    """
    Best-effort recursive scan for a dict containing:
      - cell_m (or grid_m / cell_size_m)
      - origin_x_m, origin_y_m (or origin: {x0_m, y0_m})
      - proj_epsg (or epsg / crs_epsg)
      - optional ix_min/ix_max/iy_min/iy_max
    Returns {} if not found.

    This is "legacy grid" support; for H3-based Step 1, this typically returns {}.
    """
    best: Dict[str, Any] = {}

    def consider(d: Dict[str, Any]) -> None:
        nonlocal best

        cell_m = _to_float(d.get("cell_m", None))
        if cell_m is None:
            cell_m = _to_float(d.get("grid_m", None))
        if cell_m is None:
            cell_m = _to_float(d.get("cell_size_m", None))

        ox = _to_float(d.get("origin_x_m", None))
        oy = _to_float(d.get("origin_y_m", None))

        origin = d.get("origin", None)
        if isinstance(origin, dict):
            if ox is None:
                ox = _to_float(origin.get("x0_m", None))
            if oy is None:
                oy = _to_float(origin.get("y0_m", None))

        epsg = _to_int(d.get("proj_epsg", None))
        if epsg is None:
            epsg = _to_int(d.get("epsg", None))
        if epsg is None:
            epsg = _to_int(d.get("crs_epsg", None))

        if cell_m is None or ox is None or oy is None or epsg is None:
            return

        ix_min = _to_int(d.get("ix_min", None))
        ix_max = _to_int(d.get("ix_max", None))
        iy_min = _to_int(d.get("iy_min", None))
        iy_max = _to_int(d.get("iy_max", None))

        cand: Dict[str, Any] = {
            "cell_m": float(cell_m),
            "origin_x_m": float(ox),
            "origin_y_m": float(oy),
            "epsg": int(epsg),
        }
        if ix_min is not None and ix_max is not None and iy_min is not None and iy_max is not None:
            cand.update({"ix_min": int(ix_min), "ix_max": int(ix_max), "iy_min": int(iy_min), "iy_max": int(iy_max)})

        # prefer candidates with explicit extents
        if not best:
            best = cand
            return
        best_has_ext = all(k in best for k in ("ix_min", "ix_max", "iy_min", "iy_max"))
        cand_has_ext = all(k in cand for k in ("ix_min", "ix_max", "iy_min", "iy_max"))
        if cand_has_ext and not best_has_ext:
            best = cand

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            consider(x)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(obj)
    return best


def _extent_from_well_to_cell_csv(path: Path) -> Optional[Tuple[int, int, int, int]]:
    """
    Return (ix_min, ix_max, iy_min, iy_max) best-effort.

    Supports:
      - cell_col/cell_row
      - grid_ix/grid_iy
      - parse from h3_cell/bin_id/cell_id like g10_6
    """
    p = Path(path)
    if not p.exists():
        return None

    df = pd.read_csv(p)
    cols = {c.lower(): c for c in df.columns}

    def get_int_series(name: str) -> Optional[pd.Series]:
        c = cols.get(name.lower())
        if c is None:
            return None
        s = pd.to_numeric(df[c], errors="coerce")
        return s

    col_s = get_int_series("cell_col")
    row_s = get_int_series("cell_row")
    if col_s is None or row_s is None:
        col_s = get_int_series("grid_ix")
        row_s = get_int_series("grid_iy")

    if col_s is not None and row_s is not None and np.isfinite(col_s.values).any() and np.isfinite(row_s.values).any():
        ix_min = int(np.nanmin(col_s.values))
        ix_max = int(np.nanmax(col_s.values))
        iy_min = int(np.nanmin(row_s.values))
        iy_max = int(np.nanmax(row_s.values))
        return ix_min, ix_max, iy_min, iy_max

    # Parse from cell id strings
    id_col = cols.get("h3_cell") or cols.get("bin_id") or cols.get("cell_id") or cols.get("grid_cell")
    if id_col is None:
        return None

    xs: list[int] = []
    ys: list[int] = []
    for v in df[id_col].astype(str).fillna("").values.tolist():
        s = str(v).strip()
        m = _GCELL_RE.match(s)
        if not m:
            continue
        xs.append(int(m.group(1)))
        ys.append(int(m.group(2)))

    if not xs:
        return None

    return int(min(xs)), int(max(xs)), int(min(ys)), int(max(ys)))


def _auto_step1_paths(*, reps_csv: Optional[Path], wells_gr_path: Optional[Path]) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Try to infer (Step 1 manifest, well_to_cell) from either:
      - runs/<run>/02_reps/representatives.csv
      - runs/<run>/00_wells_gr/wells_gr.parquet

    Expected structure:
      runs/<run>/01_bins/manifest.json
      runs/<run>/01_bins/well_to_cell.csv
    """
    candidates: list[Path] = []
    if reps_csv is not None:
        try:
            candidates.append(Path(reps_csv).resolve())
        except Exception:
            pass
    if wells_gr_path is not None:
        try:
            candidates.append(Path(wells_gr_path).resolve())
        except Exception:
            pass

    for p in candidates:
        # If reps_csv -> .../runA/02_reps/representatives.csv
        # If wells_gr -> .../runA/00_wells_gr/wells_gr.parquet
        try:
            run_dir = p.parent.parent
            bins_dir = run_dir / "01_bins"
            man = bins_dir / "manifest.json"
            w2c = bins_dir / "well_to_cell.csv"
            return (man if man.exists() else None, w2c if w2c.exists() else None)
        except Exception:
            continue

    return None, None


# -----------------------------------------------------------------------------
# Grid drawing (optional)
# -----------------------------------------------------------------------------

def _draw_grid_cartopy(
    ax,
    *,
    cell_m: float,
    origin_x_m: float,
    origin_y_m: float,
    epsg: int,
    ix_min: int,
    ix_max: int,
    iy_min: int,
    iy_max: int,
    alpha: float,
    lw: float,
):
    import cartopy.crs as ccrs  # type: ignore

    # cartopy supports EPSG via ccrs.epsg() in most builds
    try:
        crs_grid = ccrs.epsg(int(epsg))
    except Exception:
        crs_grid = None

    if crs_grid is None:
        # Fallback: try pyproj to transform to lon/lat and plot without a transform
        try:
            import pyproj  # type: ignore
        except Exception:
            return False

        tf = pyproj.Transformer.from_crs(int(epsg), 4326, always_xy=True)

        def plot_line(xs_m, ys_m):
            lons, lats = tf.transform(xs_m, ys_m)
            ax.plot(lons, lats, linewidth=lw, alpha=alpha, zorder=2)

        # sample density handles curvature under projection inversion
        n = 50
        x_start = origin_x_m + float(ix_min) * cell_m
        x_end = origin_x_m + float(ix_max + 1) * cell_m
        y_start = origin_y_m + float(iy_min) * cell_m
        y_end = origin_y_m + float(iy_max + 1) * cell_m

        ys = np.linspace(y_start, y_end, n)
        xs = np.linspace(x_start, x_end, n)

        for col in range(ix_min, ix_max + 2):
            x = origin_x_m + float(col) * cell_m
            plot_line(np.full_like(ys, x, dtype="float64"), ys)

        for row in range(iy_min, iy_max + 2):
            y = origin_y_m + float(row) * cell_m
            plot_line(xs, np.full_like(xs, y, dtype="float64"))

        return True

    # Native cartopy transform path (preferred)
    x_start = origin_x_m + float(ix_min) * cell_m
    x_end = origin_x_m + float(ix_max + 1) * cell_m
    y_start = origin_y_m + float(iy_min) * cell_m
    y_end = origin_y_m + float(iy_max + 1) * cell_m

    for col in range(ix_min, ix_max + 2):
        x = origin_x_m + float(col) * cell_m
        ax.plot([x, x], [y_start, y_end], transform=crs_grid, linewidth=lw, alpha=alpha, zorder=2)

    for row in range(iy_min, iy_max + 2):
        y = origin_y_m + float(row) * cell_m
        ax.plot([x_start, x_end], [y, y], transform=crs_grid, linewidth=lw, alpha=alpha, zorder=2)

    return True


def _draw_grid_plain_axes(
    ax,
    *,
    cell_m: float,
    origin_x_m: float,
    origin_y_m: float,
    epsg: int,
    ix_min: int,
    ix_max: int,
    iy_min: int,
    iy_max: int,
    alpha: float,
    lw: float,
):
    try:
        import pyproj  # type: ignore
    except Exception:
        return False

    tf = pyproj.Transformer.from_crs(int(epsg), 4326, always_xy=True)

    def plot_line(xs_m, ys_m):
        lons, lats = tf.transform(xs_m, ys_m)
        ax.plot(lons, lats, linewidth=lw, alpha=alpha, zorder=2)

    n = 50
    x_start = origin_x_m + float(ix_min) * cell_m
    x_end = origin_x_m + float(ix_max + 1) * cell_m
    y_start = origin_y_m + float(iy_min) * cell_m
    y_end = origin_y_m + float(iy_max + 1) * cell_m

    ys = np.linspace(y_start, y_end, n)
    xs = np.linspace(x_start, x_end, n)

    for col in range(ix_min, ix_max + 2):
        x = origin_x_m + float(col) * cell_m
        plot_line(np.full_like(ys, x, dtype="float64"), ys)

    for row in range(iy_min, iy_max + 2):
        y = origin_y_m + float(row) * cell_m
        plot_line(xs, np.full_like(xs, y, dtype="float64"))

    return True


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Kansas map: usable GR wells (open) and optional representatives (filled). Optional legacy grid overlay if Step1 grid params are available."
    )

    # Core inputs
    ap.add_argument("--wells-gr", type=str, required=True, help="Path to step0 wells_gr.parquet (file or dir) or csv")

    # Optional overlays (kept compatible with older and newer usage)
    ap.add_argument("--reps-csv", type=str, default="", help="Optional path to step2 representatives.csv (if provided, reps are plotted filled)")
    ap.add_argument("--ks-manifest", type=str, default="", help="Optional Kansas LAS manifest/list path (for provenance only; not required)")

    # Outputs / display
    ap.add_argument("--out-png", type=str, default="", help="Optional output PNG path")
    ap.add_argument("--dpi", type=int, default=220, help="PNG dpi (if --out-png is used)")
    ap.add_argument("--show", action="store_true", help="Show interactive window (requires a GUI backend)")

    # Optional Step1 inputs for legacy grid overlay (auto-detected from run folder if omitted)
    ap.add_argument("--step1-manifest", type=str, default="", help="Path to Step 1 manifest.json (optional; auto if blank)")
    ap.add_argument("--well-to-cell", type=str, default="", help="Path to Step 1 well_to_cell.csv (optional; auto if blank)")

    # Grid styling + padding
    ap.add_argument("--grid-alpha", type=float, default=0.25, help="Grid line alpha (0-1)")
    ap.add_argument("--grid-lw", type=float, default=0.7, help="Grid line width")
    ap.add_argument("--grid-pad-cells", type=int, default=0, help="Pad grid extent by N cells beyond manifest/observed extent")

    args = ap.parse_args()

    # Matplotlib import (lazy) with backend selection for headless PNG-only use
    try:
        import matplotlib  # type: ignore

        # If user is only writing PNG and not explicitly showing, force a non-interactive backend
        if args.out_png and not args.show:
            try:
                matplotlib.use("Agg")  # must be set before importing pyplot
            except Exception:
                pass

        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        print("ERROR: matplotlib is required for this visualization.")
        print("Install with: pip install matplotlib")
        print(f"Details: {type(e).__name__}: {e}")
        return 2

    wells_gr_path = Path(args.wells_gr)
    df_gr = _load_wells_gr(wells_gr_path)

    reps_csv = Path(args.reps_csv) if str(args.reps_csv).strip() else None
    df_reps: Optional[pd.DataFrame] = None
    if reps_csv is not None:
        if not reps_csv.exists():
            raise FileNotFoundError(reps_csv)
        df_reps = _load_reps_csv(reps_csv)

    # Resolve Step1 paths (auto if not provided)
    man_path = Path(args.step1_manifest) if str(args.step1_manifest).strip() else None
    w2c_path = Path(args.well_to_cell) if str(args.well_to_cell).strip() else None

    if (man_path is None or not man_path.exists()) or (w2c_path is None or not w2c_path.exists()):
        auto_man, auto_w2c = _auto_step1_paths(reps_csv=reps_csv, wells_gr_path=wells_gr_path)
        if man_path is None or not (man_path and man_path.exists()):
            man_path = auto_man
        if w2c_path is None or not (w2c_path and w2c_path.exists()):
            w2c_path = auto_w2c

    # Grid params (legacy)
    grid_params: Dict[str, Any] = {}
    if man_path is not None and man_path.exists():
        man = _load_json(man_path)
        grid_params = _extract_grid_params_from_manifest(man) or {}

    extent = None
    if w2c_path is not None and w2c_path.exists():
        extent = _extent_from_well_to_cell_csv(w2c_path)

    if grid_params and not all(k in grid_params for k in ("ix_min", "ix_max", "iy_min", "iy_max")):
        if extent is not None:
            ix_min, ix_max, iy_min, iy_max = extent
            grid_params.update({"ix_min": ix_min, "ix_max": ix_max, "iy_min": iy_min, "iy_max": iy_max})

    pad = int(args.grid_pad_cells)
    if grid_params and all(k in grid_params for k in ("ix_min", "ix_max", "iy_min", "iy_max")) and pad != 0:
        grid_params["ix_min"] = int(grid_params["ix_min"]) - pad
        grid_params["ix_max"] = int(grid_params["ix_max"]) + pad
        grid_params["iy_min"] = int(grid_params["iy_min"]) - pad
        grid_params["iy_max"] = int(grid_params["iy_max"]) + pad

    geom, _ccrs_mod = _get_kansas_geom()

    # Title / labels
    base_title = "Kansas wells: usable GR (open)"
    if df_reps is not None:
        base_title += " + representatives (filled)"
    base_title += " — StrataFrame"

    if geom is not None:
        import cartopy.crs as ccrs2  # type: ignore

        fig = plt.figure(figsize=(11, 8))
        ax = plt.axes(projection=ccrs2.PlateCarree())

        ax.add_geometries([geom], crs=ccrs2.PlateCarree(), facecolor="none", edgecolor="black", linewidth=1.2, zorder=3)

        extent_ll = _extent_from_geom(geom, pad_frac=0.03)
        ax.set_extent([extent_ll[0], extent_ll[1], extent_ll[2], extent_ll[3]], crs=ccrs2.PlateCarree())

        # Optional grid overlay (legacy)
        drew_grid = False
        if grid_params and all(
            k in grid_params for k in ("cell_m", "origin_x_m", "origin_y_m", "epsg", "ix_min", "ix_max", "iy_min", "iy_max")
        ):
            drew_grid = _draw_grid_cartopy(
                ax,
                cell_m=float(grid_params["cell_m"]),
                origin_x_m=float(grid_params["origin_x_m"]),
                origin_y_m=float(grid_params["origin_y_m"]),
                epsg=int(grid_params["epsg"]),
                ix_min=int(grid_params["ix_min"]),
                ix_max=int(grid_params["ix_max"]),
                iy_min=int(grid_params["iy_min"]),
                iy_max=int(grid_params["iy_max"]),
                alpha=float(args.grid_alpha),
                lw=float(args.grid_lw),
            )

        if grid_params and not drew_grid:
            print("NOTE: grid parameters found but grid could not be drawn (cartopy EPSG support missing and pyproj not available).")
        elif not grid_params:
            # Expected for H3-based Step 1
            pass

        # All GR wells: open circles
        ax.scatter(
            df_gr["lon"].values,
            df_gr["lat"].values,
            s=14,
            facecolors="none",
            edgecolors="black",
            linewidths=0.5,
            alpha=0.5,
            transform=ccrs2.PlateCarree(),
            label=f"Usable GR wells (n={len(df_gr)})",
            zorder=4,
        )

        # Selected reps: filled circles (optional)
        if df_reps is not None and len(df_reps) > 0:
            ax.scatter(
                df_reps["lon"].values,
                df_reps["lat"].values,
                s=22,
                alpha=0.55,
                edgecolors="black",
                linewidths=0.05,
                transform=ccrs2.PlateCarree(),
                label=f"Selected reps (n={len(df_reps)})",
                zorder=5,
            )

        ax.set_title(base_title)
        ax.legend(loc="lower left", frameon=True)

    else:
        # Fallback: bbox only (no cartopy)
        lon_min, lon_max, lat_min, lat_max = _kansas_bbox_fallback()

        fig, ax = plt.subplots(figsize=(11, 8))

        ax.plot([lon_min, lon_max, lon_max, lon_min, lon_min], [lat_min, lat_min, lat_max, lat_max, lat_min], linewidth=1.2, zorder=3)

        # Optional grid overlay via pyproj (legacy)
        drew_grid = False
        if grid_params and all(
            k in grid_params for k in ("cell_m", "origin_x_m", "origin_y_m", "epsg", "ix_min", "ix_max", "iy_min", "iy_max")
        ):
            drew_grid = _draw_grid_plain_axes(
                ax,
                cell_m=float(grid_params["cell_m"]),
                origin_x_m=float(grid_params["origin_x_m"]),
                origin_y_m=float(grid_params["origin_y_m"]),
                epsg=int(grid_params["epsg"]),
                ix_min=int(grid_params["ix_min"]),
                ix_max=int(grid_params["ix_max"]),
                iy_min=int(grid_params["iy_min"]),
                iy_max=int(grid_params["iy_max"]),
                alpha=float(args.grid_alpha),
                lw=float(args.grid_lw),
            )

        if grid_params and not drew_grid:
            print("NOTE: grid parameters found but grid could not be drawn (pyproj not available).")
        elif not grid_params:
            pass

        ax.scatter(
            df_gr["lon"].values,
            df_gr["lat"].values,
            s=14,
            facecolors="none",
            edgecolors="black",
            linewidths=0.5,
            alpha=0.5,
            label=f"Usable GR wells (n={len(df_gr)})",
            zorder=4,
        )

        if df_reps is not None and len(df_reps) > 0:
            ax.scatter(
                df_reps["lon"].values,
                df_reps["lat"].values,
                s=22,
                alpha=0.55,
                edgecolors="black",
                linewidths=0.2,
                label=f"Selected reps (n={len(df_reps)})",
                zorder=5,
            )

        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(base_title)
        ax.legend(loc="lower left", frameon=True)

    fig.tight_layout()

    if args.out_png:
        out = Path(args.out_png)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=int(args.dpi))
        print(f"Wrote: {out}")

    # Show if explicitly requested, or if no output target is provided
    if args.show or not args.out_png:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
