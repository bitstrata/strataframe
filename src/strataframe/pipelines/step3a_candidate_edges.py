# src/strataframe/pipelines/step3a_candidate_edges.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from strataframe.spatial.geodesy import haversine_km_vec


@dataclass(frozen=True)
class CandidateGraphConfig:
    """
    Global candidate graph (bins are NOT barriers).

    Algorithm (per node i):
      1) Find nearest neighbour in each cardinal sector (N/E/S/W), any distance.
      2) Find all neighbours within r_max_km; sort ascending by distance; append.
      3) De-dup; truncate to k_max.
      4) (optional) if still empty, ensure_one_nn chooses the global nearest neighbour.

    Output edges are undirected union of per-node neighbour lists.
    """
    k_max: int = 12
    r_max_km: float = 5.0
    use_quadrants: bool = True
    ensure_one_nn: bool = True

    # column names
    rep_id_col: str = "rep_id"
    lat_col: str = "lat"
    lon_col: str = "lon"


def _require_pd() -> None:
    if pd is None:
        raise RuntimeError("pandas is required for step3a candidate edges. Install with: pip install pandas")


def _canon(u: str, v: str) -> Tuple[str, str]:
    u = str(u)
    v = str(v)
    return (u, v) if u < v else (v, u)


def _bearing_deg_vec(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """
    Initial bearing (azimuth) from (lat1,lon1) -> (lat2,lon2), degrees in [0,360).

    Supports broadcasting.
    """
    lat1 = np.asarray(lat1, dtype="float64")
    lon1 = np.asarray(lon1, dtype="float64")
    lat2 = np.asarray(lat2, dtype="float64")
    lon2 = np.asarray(lon2, dtype="float64")

    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    dlmb = np.deg2rad(lon2 - lon1)

    y = np.sin(dlmb) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlmb)

    th = np.arctan2(y, x)  # [-pi, pi]
    brg = (np.rad2deg(th) + 360.0) % 360.0
    return brg.astype("float64", copy=False)


def _cardinal_sector_masks(az_deg: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Cardinal sectors (each 90°, centered on N/E/S/W):
      N: [315,360) U [0,45)
      E: [45,135)
      S: [135,225)
      W: [225,315)
    """
    az = np.asarray(az_deg, dtype="float64")
    mN = (az >= 315.0) | (az < 45.0)
    mE = (az >= 45.0) & (az < 135.0)
    mS = (az >= 135.0) & (az < 225.0)
    mW = (az >= 225.0) & (az < 315.0)
    return {"N": mN, "E": mE, "S": mS, "W": mW}


def build_candidate_edges(
    reps: "pd.DataFrame",
    *,
    cfg: CandidateGraphConfig,
    bins_meta: Optional["pd.DataFrame"] = None,  # kept for API compatibility; unused
) -> "pd.DataFrame":
    """
    Returns a DataFrame with columns:
      src_rep_id, dst_rep_id, dist_km, source
    where source is one of:
      - "quadrant" (selected as nearest in N/E/S/W for at least one endpoint)
      - "radius"   (selected from within-radius list for at least one endpoint)
      - "quadrant+radius" (selected by both rules across endpoints)
    """
    _require_pd()

    for c in (cfg.rep_id_col, cfg.lat_col, cfg.lon_col):
        if c not in reps.columns:
            raise ValueError(f"reps missing required column: {c}")

    df = reps[[cfg.rep_id_col, cfg.lat_col, cfg.lon_col]].copy()
    df[cfg.rep_id_col] = df[cfg.rep_id_col].astype(str)
    df[cfg.lat_col] = pd.to_numeric(df[cfg.lat_col], errors="coerce")
    df[cfg.lon_col] = pd.to_numeric(df[cfg.lon_col], errors="coerce")

    # Keep only finite lat/lon
    m = np.isfinite(df[cfg.lat_col].to_numpy()) & np.isfinite(df[cfg.lon_col].to_numpy())
    df = df[m].copy()

    rep_ids = df[cfg.rep_id_col].to_numpy(dtype=str)
    lat = df[cfg.lat_col].to_numpy(dtype="float64")
    lon = df[cfg.lon_col].to_numpy(dtype="float64")
    n = int(rep_ids.size)

    if n < 2:
        return pd.DataFrame(columns=["src_rep_id", "dst_rep_id", "dist_km", "source"])

    k_max = max(1, int(cfg.k_max))
    r_max = float(cfg.r_max_km)

    # accumulate undirected edges with best distance + merged source tags
    edges: Dict[Tuple[str, str], Tuple[float, str]] = {}

    for i in range(n):
        # distance + azimuth vectors from node i to all nodes
        di = haversine_km_vec(lat[i], lon[i], lat, lon).reshape(-1)  # km
        az = _bearing_deg_vec(lat[i], lon[i], lat, lon).reshape(-1)

        valid = np.isfinite(di) & (np.arange(n) != i) & (di > 0.0)

        picks: list[int] = []
        pick_src: Dict[int, str] = {}

        # (A) quadrant seeds (any distance)
        if bool(cfg.use_quadrants):
            sect = _cardinal_sector_masks(az)
            for _name, msec in sect.items():
                mm = valid & msec
                if not np.any(mm):
                    continue
                j_rel = np.argmin(np.where(mm, di, np.inf))
                if np.isfinite(di[j_rel]):
                    picks.append(int(j_rel))
                    pick_src[int(j_rel)] = "quadrant"

        # (B) within-radius candidates (<= r_max km)
        if np.isfinite(r_max) and r_max > 0:
            mr = valid & (di <= r_max)
            if np.any(mr):
                j_all = np.where(mr)[0]
                j_all = j_all[np.argsort(di[j_all], kind="mergesort")]
                for j in j_all.tolist():
                    jj = int(j)
                    if jj not in pick_src:
                        picks.append(jj)
                        pick_src[jj] = "radius"
                    if len(pick_src) >= k_max:
                        break

        # (C) ensure at least one NN if still empty
        if bool(cfg.ensure_one_nn) and len(pick_src) == 0 and np.any(valid):
            j_nn = int(np.argmin(np.where(valid, di, np.inf)))
            picks.append(j_nn)
            pick_src[j_nn] = "quadrant" if bool(cfg.use_quadrants) else "radius"

        # de-dup in order (quadrants first, then radius)
        seen: set[int] = set()
        picks_u: list[int] = []
        for j in picks:
            if j in seen:
                continue
            seen.add(j)
            picks_u.append(j)
            if len(picks_u) >= k_max:
                break

        # emit undirected edges
        u0 = str(rep_ids[i])
        for j in picks_u:
            v0 = str(rep_ids[j])
            if u0 == v0:
                continue
            u, v = _canon(u0, v0)
            d = float(di[j])
            src = pick_src.get(int(j), "radius")

            prev = edges.get((u, v))
            if prev is None:
                edges[(u, v)] = (d, src)
            else:
                # keep shortest distance (should be same), merge sources
                d_prev, s_prev = prev
                d_keep = min(float(d_prev), d)
                tags = set(str(s_prev).split("+"))
                tags.add(str(src))
                s_keep = "+".join(sorted(tags))
                edges[(u, v)] = (d_keep, s_keep)

    rows = [{"src_rep_id": u, "dst_rep_id": v, "dist_km": d, "source": s} for (u, v), (d, s) in edges.items()]
    out = pd.DataFrame(rows)
    if out.shape[0] == 0:
        return pd.DataFrame(columns=["src_rep_id", "dst_rep_id", "dist_km", "source"])
    return out.sort_values(["dist_km", "src_rep_id", "dst_rep_id"]).reset_index(drop=True)
