# src/strataframe/spatial/grid_index.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    """
    Simple orthogonal grid on lat/lon using a local equirectangular approximation.

    Notes:
    - This is *not* a GIS-grade projection, but is stable and adequate for 10x10 km indexing,
      kernel neighborhoods, and distance weighting.
    - To keep IDs stable across runs, persist origin_lat/origin_lon in diagnostics and reuse them.
    """
    grid_km: float = 10.0
    origin_lat: float = 0.0
    origin_lon: float = 0.0

    # Local scale factors computed at origin_lat
    meters_per_deg_lat: float = 111_132.0

    def meters_per_deg_lon(self) -> float:
        return 111_320.0 * float(np.cos(np.deg2rad(self.origin_lat)))


def infer_grid_origin(lat: np.ndarray, lon: np.ndarray, *, grid_km: float) -> Tuple[float, float]:
    """
    Infer a stable-ish origin from dataset extent by snapping the min lat/lon
    to the grid spacing in degrees (approx).
    """
    lat = np.asarray(lat, dtype="float64")
    lon = np.asarray(lon, dtype="float64")
    m = np.isfinite(lat) & np.isfinite(lon)
    if not np.any(m):
        return 0.0, 0.0

    lat0 = float(np.nanmin(lat[m]))
    lon0 = float(np.nanmin(lon[m]))

    # Convert grid_km to approx degrees at this latitude for snapping.
    # This is only used to pick an origin; subsequent index uses meters.
    deg_lat = (float(grid_km) * 1000.0) / 111_132.0
    deg_lon = (float(grid_km) * 1000.0) / (111_320.0 * float(np.cos(np.deg2rad(lat0))) + 1e-12)

    lat0s = np.floor(lat0 / deg_lat) * deg_lat
    lon0s = np.floor(lon0 / deg_lon) * deg_lon
    return float(lat0s), float(lon0s)


def latlon_to_xy_m(lat: np.ndarray, lon: np.ndarray, *, spec: GridSpec) -> Tuple[np.ndarray, np.ndarray]:
    """
    Equirectangular: x ~ dlon * cos(lat0), y ~ dlat
    """
    lat = np.asarray(lat, dtype="float64")
    lon = np.asarray(lon, dtype="float64")
    x = (lon - float(spec.origin_lon)) * float(spec.meters_per_deg_lon())
    y = (lat - float(spec.origin_lat)) * float(spec.meters_per_deg_lat)
    return x, y


def xy_m_to_ij(x: np.ndarray, y: np.ndarray, *, spec: GridSpec) -> Tuple[np.ndarray, np.ndarray]:
    grid_m = float(spec.grid_km) * 1000.0
    i = np.floor(np.asarray(x, dtype="float64") / grid_m).astype("int64")
    j = np.floor(np.asarray(y, dtype="float64") / grid_m).astype("int64")
    return i, j


def ij_to_cell_id(i: int, j: int) -> str:
    # Sign-safe fixed width for stable lexical sorting.
    return f"g{i:+07d}_{j:+07d}"


def cell_id_to_ij(cell_id: str) -> Tuple[int, int]:
    s = (cell_id or "").strip()
    if not s.startswith("g") or "_" not in s:
        raise ValueError(f"Invalid grid cell id: {cell_id!r}")
    a, b = s[1:].split("_", 1)
    return int(a), int(b)


def kernel_cells_ij(i0: int, j0: int, *, radius: int) -> List[Tuple[int, int]]:
    r = int(max(0, radius))
    out: List[Tuple[int, int]] = []
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            out.append((int(i0 + di), int(j0 + dj)))
    return out


def haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """
    Vector haversine: distance between (lat1,lon1) and arrays (lat2,lon2), in km.
    """
    lat2 = np.asarray(lat2, dtype="float64")
    lon2 = np.asarray(lon2, dtype="float64")
    R = 6371.0088
    phi1 = np.deg2rad(float(lat1))
    lam1 = np.deg2rad(float(lon1))
    phi2 = np.deg2rad(lat2)
    lam2 = np.deg2rad(lon2)

    dphi = phi2 - phi1
    dlam = lam2 - lam1
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    return (2.0 * R * np.arcsin(np.sqrt(a))).astype("float64", copy=False)
