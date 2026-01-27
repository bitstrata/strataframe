# src/strataframe/graph/pair_selection.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

try:
    from scipy.spatial import Delaunay  # type: ignore
except Exception:  # pragma: no cover
    Delaunay = None  # type: ignore

from strataframe.spatial.geodesy import haversine_km


@dataclass(frozen=True)
class PairSelectionConfig:
    # Small threshold for “within cluster” edges
    max_dist_km: float = 15.0
    # Larger threshold for pruning Delaunay-added edges
    delaunay_prune_km: float = 60.0
    use_delaunay: bool = True


def _require_scipy() -> None:
    if Delaunay is None:
        raise RuntimeError("scipy is required for Delaunay. Install with: pip install scipy")


def _project_xy_km(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Simple local projection to km for Kansas-scale extents:
      x ~ lon*cos(lat0), y ~ lat
    """
    lat0 = float(np.nanmean(lats))
    deg_to_km = 111.32
    x = (lons * np.cos(np.deg2rad(lat0))) * deg_to_km
    y = lats * deg_to_km
    return np.vstack([x, y]).T


def _as_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def build_edges_sparse(
    wells: Sequence[object],
    *,
    cfg: PairSelectionConfig,
    lat_key: str = "lat",
    lon_key: str = "lon",
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    Returns list of edges (u, v, attrs), where u/v are well_id.

    Expected per-well attributes:
      - well_id (stringable)
      - lat_key / lon_key (float degrees)

    Notes:
      - Delaunay is used only to add sparse “bridge” edges; failures fall back
        to max-distance edges only (keeps workflow robust).
    """
    ids: List[str] = []
    lats: List[float] = []
    lons: List[float] = []

    for w in wells:
        wid = str(getattr(w, "well_id"))
        lat = _as_float(getattr(w, lat_key, np.nan))
        lon = _as_float(getattr(w, lon_key, np.nan))
        if not (np.isfinite(lat) and np.isfinite(lon)):
            raise ValueError(f"Well {wid}: missing/invalid {lat_key}/{lon_key} (got lat={lat}, lon={lon})")
        ids.append(wid)
        lats.append(float(lat))
        lons.append(float(lon))

    lats_a = np.asarray(lats, dtype="float64")
    lons_a = np.asarray(lons, dtype="float64")
    n = int(len(ids))

    edges: Dict[Tuple[str, str], Dict[str, Any]] = {}

    # 1) Within-cluster edges by max distance (simple O(n^2), OK for reps)
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_km(lats_a[i], lons_a[i], lats_a[j], lons_a[j])
            if d <= float(cfg.max_dist_km):
                edges[(ids[i], ids[j])] = {"dist_km": float(d), "source": "max_dist"}

    # 2) Delaunay to connect clusters, then prune long edges
    if cfg.use_delaunay and n >= 3:
        _require_scipy()
        pts = _project_xy_km(lats_a, lons_a)

        # Delaunay can fail on collinear / duplicate points; guard with try/except.
        try:
            tri = Delaunay(pts)

            # Unique edges from triangles
            for simplex in tri.simplices:
                a, b, c = int(simplex[0]), int(simplex[1]), int(simplex[2])
                for i, j in ((a, b), (b, c), (a, c)):
                    ii, jj = (i, j) if i < j else (j, i)
                    d = haversine_km(lats_a[ii], lons_a[ii], lats_a[jj], lons_a[jj])
                    if d > float(cfg.delaunay_prune_km):
                        continue

                    key = (ids[ii], ids[jj])
                    if key not in edges:
                        edges[key] = {"dist_km": float(d), "source": "delaunay"}
                    else:
                        # Keep min dist if duplicated from different sources
                        edges[key]["dist_km"] = float(min(float(edges[key]["dist_km"]), float(d)))

        except Exception:
            # Robust fallback: keep only max_dist edges
            pass

    return [(u, v, attrs) for (u, v), attrs in sorted(edges.items())]
