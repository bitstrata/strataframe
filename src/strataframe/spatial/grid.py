# src/strataframe/spatial/grid.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# consistent with your geodesy constant
EARTH_RADIUS_KM: float = 6371.0088


def _to_xy_km(lat: np.ndarray, lon: np.ndarray, *, lat0: float, lon0: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert lat/lon (deg) to local x/y (km) using equirectangular approximation
    around reference (lat0,lon0). Suitable for regional extents.
    """
    lat = np.asarray(lat, dtype="float64")
    lon = np.asarray(lon, dtype="float64")

    phi = np.deg2rad(lat)
    phi0 = np.deg2rad(float(lat0))
    lam = np.deg2rad(lon)
    lam0 = np.deg2rad(float(lon0))

    x = EARTH_RADIUS_KM * (lam - lam0) * np.cos(phi0)
    y = EARTH_RADIUS_KM * (phi - phi0)
    return x, y


def _to_latlon(x_km: np.ndarray, y_km: np.ndarray, *, lat0: float, lon0: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inverse of _to_xy_km for centroids.
    """
    x_km = np.asarray(x_km, dtype="float64")
    y_km = np.asarray(y_km, dtype="float64")
    phi0 = np.deg2rad(float(lat0))
    lam0 = np.deg2rad(float(lon0))

    phi = (y_km / EARTH_RADIUS_KM) + phi0
    lam = (x_km / (EARTH_RADIUS_KM * np.cos(phi0))) + lam0

    lat = np.rad2deg(phi)
    lon = np.rad2deg(lam)
    return lat, lon


@dataclass(frozen=True)
class GridSpec:
    """
    Defines a rectangular grid over a local x/y (km) plane.

    origin is the southwest corner in the local plane (x_min_km, y_min_km).
    row increases northward, col increases eastward.
    """
    lat0: float
    lon0: float
    x_min_km: float
    y_min_km: float
    cell_km: float

    n_rows: int
    n_cols: int

    def cell_id(self, row: int, col: int) -> str:
        return f"r{int(row):05d}_c{int(col):05d}"

    def xy_centroid_km(self, row: int, col: int) -> Tuple[float, float]:
        x = self.x_min_km + (float(col) + 0.5) * self.cell_km
        y = self.y_min_km + (float(row) + 0.5) * self.cell_km
        return x, y

    def latlon_centroid(self, row: int, col: int) -> Tuple[float, float]:
        x, y = self.xy_centroid_km(row, col)
        lat, lon = _to_latlon(np.array([x]), np.array([y]), lat0=self.lat0, lon0=self.lon0)
        return float(lat[0]), float(lon[0])


def choose_cell_km_for_target_bins(
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    target_bins: int,
    min_bin_size: int = 1,
    max_iter: int = 30,
) -> float:
    """
    Pick cell_km so that occupied cell count ~ target_bins while keeping
    min occupied-cell population >= min_bin_size when possible.

    This is a heuristic, not an optimizer; it is stable and deterministic.
    """
    lat = np.asarray(lat, dtype="float64").reshape(-1)
    lon = np.asarray(lon, dtype="float64").reshape(-1)
    m = np.isfinite(lat) & np.isfinite(lon)
    lat = lat[m]
    lon = lon[m]
    if lat.size < 2:
        return 10.0  # arbitrary fallback

    lat0 = float(np.nanmean(lat))
    lon0 = float(np.nanmean(lon))
    x, y = _to_xy_km(lat, lon, lat0=lat0, lon0=lon0)

    # bounding box area in km^2
    dx = float(np.nanmax(x) - np.nanmin(x))
    dy = float(np.nanmax(y) - np.nanmin(y))
    area = max(1e-6, dx * dy)

    # initial guess: sqrt(area/target_bins)
    cell = float(np.sqrt(area / max(1, int(target_bins))))
    cell = max(0.25, min(cell, 2000.0))

    def eval_cell(cell_km: float) -> Tuple[int, int]:
        col = np.floor((x - np.nanmin(x)) / cell_km).astype(np.int64)
        row = np.floor((y - np.nanmin(y)) / cell_km).astype(np.int64)
        # count occupancy per (row,col)
        key = row.astype(np.int64) * 10_000_000 + col.astype(np.int64)
        _, counts = np.unique(key, return_counts=True)
        n_occ = int(counts.size)
        min_cnt = int(np.min(counts)) if counts.size else 0
        return n_occ, min_cnt

    best = (cell, float("inf"))
    # coarse-to-fine multiplicative search
    for _ in range(int(max_iter)):
        n_occ, min_cnt = eval_cell(cell)
        # objective: match target_bins, penalize min_cnt < min_bin_size
        err = abs(n_occ - int(target_bins))
        if min_cnt < int(min_bin_size):
            err += 10_000 + (int(min_bin_size) - min_cnt) * 100
        if err < best[1]:
            best = (cell, err)

        # update rule
        if min_cnt < int(min_bin_size) or n_occ > int(target_bins):
            cell *= 1.25  # coarsen
        else:
            cell *= 0.85  # refine
        cell = max(0.25, min(cell, 2000.0))

    return float(best[0])


def build_grid_spec(
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    cell_km: Optional[float] = None,
    target_bins: Optional[int] = None,
    min_bin_size: int = 1,
    pad_frac: float = 0.01,
) -> GridSpec:
    """
    Build a grid spec covering the input points.
    If cell_km is None and target_bins is provided, choose cell_km heuristically.
    """
    lat = np.asarray(lat, dtype="float64").reshape(-1)
    lon = np.asarray(lon, dtype="float64").reshape(-1)
    m = np.isfinite(lat) & np.isfinite(lon)
    lat = lat[m]
    lon = lon[m]
    if lat.size < 2:
        raise ValueError("Too few finite lat/lon to build a grid")

    lat0 = float(np.nanmean(lat))
    lon0 = float(np.nanmean(lon))

    x, y = _to_xy_km(lat, lon, lat0=lat0, lon0=lon0)

    if cell_km is None:
        if target_bins is None:
            raise ValueError("Either cell_km or target_bins must be provided")
        cell_km = choose_cell_km_for_target_bins(lat, lon, target_bins=int(target_bins), min_bin_size=int(min_bin_size))

    cell_km = float(cell_km)
    if not np.isfinite(cell_km) or cell_km <= 0:
        raise ValueError(f"Invalid cell_km: {cell_km}")

    x0 = float(np.nanmin(x))
    x1 = float(np.nanmax(x))
    y0 = float(np.nanmin(y))
    y1 = float(np.nanmax(y))

    # optional pad to reduce edge effects
    pad = float(pad_frac)
    dx = max(1e-9, x1 - x0)
    dy = max(1e-9, y1 - y0)
    x0 -= dx * pad
    x1 += dx * pad
    y0 -= dy * pad
    y1 += dy * pad

    n_cols = int(np.ceil((x1 - x0) / cell_km))
    n_rows = int(np.ceil((y1 - y0) / cell_km))
    n_cols = max(1, n_cols)
    n_rows = max(1, n_rows)

    return GridSpec(
        lat0=lat0,
        lon0=lon0,
        x_min_km=x0,
        y_min_km=y0,
        cell_km=cell_km,
        n_rows=n_rows,
        n_cols=n_cols,
    )


def assign_points_to_grid(
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    grid: GridSpec,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (row, col, cell_id) arrays (same length as lat/lon).
    Non-finite points get row/col = -1 and cell_id = "".
    """
    lat = np.asarray(lat, dtype="float64").reshape(-1)
    lon = np.asarray(lon, dtype="float64").reshape(-1)
    n = int(lat.size)
    row = np.full((n,), -1, dtype=np.int64)
    col = np.full((n,), -1, dtype=np.int64)
    cell_id = np.full((n,), "", dtype=object)

    m = np.isfinite(lat) & np.isfinite(lon)
    if not np.any(m):
        return row, col, cell_id.astype(str)

    x, y = _to_xy_km(lat[m], lon[m], lat0=grid.lat0, lon0=grid.lon0)
    cc = np.floor((x - grid.x_min_km) / grid.cell_km).astype(np.int64)
    rr = np.floor((y - grid.y_min_km) / grid.cell_km).astype(np.int64)

    # clamp to grid bounds (defensive)
    cc = np.clip(cc, 0, grid.n_cols - 1)
    rr = np.clip(rr, 0, grid.n_rows - 1)

    row[m] = rr
    col[m] = cc
    for i0, (r, c) in zip(np.where(m)[0].tolist(), zip(rr.tolist(), cc.tolist())):
        cell_id[i0] = grid.cell_id(r, c)

    return row, col, cell_id.astype(str)


def neighbors_3x3(row: int, col: int, *, grid: GridSpec) -> List[Tuple[int, int]]:
    """
    Return (r,c) for the 3×3 neighborhood around (row,col), cropped at boundaries.
    """
    out: List[Tuple[int, int]] = []
    for dr in (-1, 0, 1):
        rr = int(row) + dr
        if rr < 0 or rr >= grid.n_rows:
            continue
        for dc in (-1, 0, 1):
            cc = int(col) + dc
            if cc < 0 or cc >= grid.n_cols:
                continue
            out.append((rr, cc))
    return out


# -----------------------------------------------------------------------------
# Compatibility layer for Step 2 (grid assignment in meters + g<ix>_<iy> ids)
# -----------------------------------------------------------------------------

from dataclasses import dataclass

try:
    from pyproj import Transformer  # type: ignore
except Exception:  # pragma: no cover
    Transformer = None  # type: ignore


DEFAULT_GRID_PROJ_EPSG = 5070  # NAD83 / Conus Albers (good for Kansas)


@dataclass(frozen=True)
class GridAssignMeta:
    proj_method: str
    proj_epsg: int
    cell_m: float
    x0_m: float
    y0_m: float
    pad_frac: float
    lat0: float
    lon0: float
    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float
    ix_min: int
    ix_max: int
    iy_min: int
    iy_max: int


def grid_cell_id(ix: int, iy: int) -> str:
    """
    Grid cell id used by your Step1/Step2 grid workflow.
    Convention: g<col>_<row> == g<ix>_<iy>
    """
    return f"g{int(ix)}_{int(iy)}"


def assign_grid_cells(
    *,
    lon: np.ndarray,
    lat: np.ndarray,
    grid_m: float,
    proj_epsg: int = DEFAULT_GRID_PROJ_EPSG,
    pad_frac: float = 0.01,
    origin_x_m: Optional[float] = None,
    origin_y_m: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, GridAssignMeta]:
    """
    Assign (lon,lat) points (EPSG:4326) into a regular grid in *meters*.

    Returns:
      ix, iy: int arrays (col, row)
      x_m, y_m: projected coordinates in meters
      meta: GridAssignMeta with projection + origin used

    Notes:
      - If pyproj is available, uses EPSG:proj_epsg (default 5070).
      - Otherwise falls back to local equirectangular meters (not a true EPSG).
      - origin is snapped to cell size for stability.
    """
    lat = np.asarray(lat, dtype="float64").reshape(-1)
    lon = np.asarray(lon, dtype="float64").reshape(-1)
    n = int(lat.size)

    ix = np.full((n,), -1, dtype=np.int64)
    iy = np.full((n,), -1, dtype=np.int64)
    x_m = np.full((n,), np.nan, dtype="float64")
    y_m = np.full((n,), np.nan, dtype="float64")

    m = np.isfinite(lat) & np.isfinite(lon)
    if not np.any(m):
        meta = GridAssignMeta(
            proj_method="none",
            proj_epsg=int(proj_epsg),
            cell_m=float(grid_m),
            x0_m=0.0,
            y0_m=0.0,
            pad_frac=float(pad_frac),
            lat0=float("nan"),
            lon0=float("nan"),
            x_min_m=float("nan"),
            x_max_m=float("nan"),
            y_min_m=float("nan"),
            y_max_m=float("nan"),
            ix_min=0,
            ix_max=0,
            iy_min=0,
            iy_max=0,
        )
        return ix, iy, x_m, y_m, meta

    # Reference for fallback
    lat0 = float(np.nanmean(lat[m]))
    lon0 = float(np.nanmean(lon[m]))

    # Project
    if Transformer is not None:
        tr = Transformer.from_crs("EPSG:4326", f"EPSG:{int(proj_epsg)}", always_xy=True)
        xx, yy = tr.transform(lon[m], lat[m])
        xx = np.asarray(xx, dtype="float64")
        yy = np.asarray(yy, dtype="float64")
        proj_method = "pyproj"
        proj_epsg_used = int(proj_epsg)
    else:
        # Fallback: local equirectangular km -> m (not a real EPSG)
        x_km, y_km = _to_xy_km(lat[m], lon[m], lat0=lat0, lon0=lon0)
        xx = np.asarray(x_km, dtype="float64") * 1000.0
        yy = np.asarray(y_km, dtype="float64") * 1000.0
        proj_method = "equirectangular"
        proj_epsg_used = 0

    x_m[m] = xx
    y_m[m] = yy

    cell = float(grid_m)
    if not np.isfinite(cell) or cell <= 0:
        raise ValueError(f"Invalid grid_m: {grid_m}")

    x_min = float(np.nanmin(xx))
    x_max = float(np.nanmax(xx))
    y_min = float(np.nanmin(yy))
    y_max = float(np.nanmax(yy))

    dx = max(1e-9, x_max - x_min)
    dy = max(1e-9, y_max - y_min)
    pad = float(max(dx, dy) * float(pad_frac))

    x_min_p = x_min - pad
    y_min_p = y_min - pad

    # Origin: either provided or snapped down to cell size
    if origin_x_m is None:
        x0 = math.floor(x_min_p / cell) * cell
    else:
        x0 = float(origin_x_m)

    if origin_y_m is None:
        y0 = math.floor(y_min_p / cell) * cell
    else:
        y0 = float(origin_y_m)

    cc = np.floor((xx - x0) / cell).astype(np.int64)
    rr = np.floor((yy - y0) / cell).astype(np.int64)

    ix[m] = cc
    iy[m] = rr

    ix_min = int(np.min(cc)) if cc.size else 0
    ix_max = int(np.max(cc)) if cc.size else 0
    iy_min = int(np.min(rr)) if rr.size else 0
    iy_max = int(np.max(rr)) if rr.size else 0

    meta = GridAssignMeta(
        proj_method=proj_method,
        proj_epsg=proj_epsg_used,
        cell_m=float(cell),
        x0_m=float(x0),
        y0_m=float(y0),
        pad_frac=float(pad_frac),
        lat0=float(lat0),
        lon0=float(lon0),
        x_min_m=float(x_min),
        x_max_m=float(x_max),
        y_min_m=float(y_min),
        y_max_m=float(y_max),
        ix_min=ix_min,
        ix_max=ix_max,
        iy_min=iy_min,
        iy_max=iy_max,
    )

    return ix, iy, x_m, y_m, meta
