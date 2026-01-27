# src/strataframe/viz/step3d_fence.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from .step3_common import (
    load_npz,
    rep_arrays_from_npz,
    line_collection_from_edges,
    load_dtw_paths_dict,
)
from .step3_colors import yellow_brown_perceptual_cmap

try:
    import pandas as pd  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("pandas is required. pip install pandas") from e

try:
    import networkx as nx  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("networkx is required for step3d. pip install networkx") from e

from strataframe.spatial.geodesy import haversine_km_vec


# -----------------------------------------------------------------------------
# DTW-based scalar shifts (hung to a datum)
# -----------------------------------------------------------------------------

def _scalar_shifts_from_dtw_paths(
    reps_ids: Sequence[str],
    fw_edges: "pd.DataFrame",
    depth_by_rep: Dict[str, np.ndarray],
    paths: Dict[Tuple[str, str], np.ndarray],
) -> Dict[str, float]:
    """
    Build simple scalar datum shifts using median depth differences on DTW paths.

    Constraint for edge (u,v):
      depth_u[i] + s_u  ~=  depth_v[j] + s_v
    => s_v - s_u ~= median(depth_u[i] - depth_v[j])
    """
    reps_ids = [str(r) for r in reps_ids]

    # adjacency restricted to reps_ids
    adj: Dict[str, List[str]] = {r: [] for r in reps_ids}
    for _, e in fw_edges.iterrows():
        u = str(e["u"])
        v = str(e["v"])
        if u in adj and v in adj and u != v:
            adj[u].append(v)
            adj[v].append(u)

    if not reps_ids:
        return {}

    ref = str(reps_ids[0])
    s: Dict[str, float] = {ref: 0.0}
    q: List[str] = [ref]

    while q:
        u = q.pop(0)
        for v in adj.get(u, []):
            if v in s:
                continue

            # get path (u,v) or (v,u)
            p = paths.get((u, v))
            flip = False
            if p is None:
                p = paths.get((v, u))
                flip = True

            if p is None:
                # no DTW path available; propagate shift without increment
                s[v] = s[u]
                q.append(v)
                continue

            p = np.asarray(p, dtype=np.int64)
            if p.ndim != 2 or p.shape[1] != 2 or p.size == 0:
                s[v] = s[u]
                q.append(v)
                continue

            du = depth_by_rep.get(u)
            dv = depth_by_rep.get(v)
            if du is None or dv is None:
                s[v] = s[u]
                q.append(v)
                continue

            ia = p[:, 0]
            ib = p[:, 1]
            if flip:
                ia, ib = ib, ia

            m = (ia >= 0) & (ia < du.size) & (ib >= 0) & (ib < dv.size)
            if not np.any(m):
                s[v] = s[u]
                q.append(v)
                continue

            dz = du[ia[m]].astype("float64") - dv[ib[m]].astype("float64")
            dz = dz[np.isfinite(dz)]
            if dz.size == 0:
                s[v] = s[u]
                q.append(v)
                continue

            inc = float(np.median(dz))
            s[v] = float(s[u] + inc)
            q.append(v)

    # ensure all present
    for rid in reps_ids:
        if rid not in s:
            s[rid] = 0.0
    return s


# -----------------------------------------------------------------------------
# Graph-based fence ordering (connected path through framework graph)
# -----------------------------------------------------------------------------

def _nearest_rep_id_to_point(reps: "pd.DataFrame", *, lon0: float, lat0: float) -> str:
    """
    Return rep_id of nearest rep (by haversine distance) to (lon0, lat0).
    Assumes reps contains 'lat','lon','rep_id'.
    """
    lat = reps["lat"].to_numpy(dtype="float64")
    lon = reps["lon"].to_numpy(dtype="float64")

    d = haversine_km_vec(lat, lon, float(lat0), float(lon0))
    d[~np.isfinite(d)] = np.inf
    i = int(np.argmin(d))
    return str(reps.iloc[i]["rep_id"])


def _build_framework_graph_nx(reps: "pd.DataFrame", fw_edges: "pd.DataFrame") -> "nx.Graph":
    """
    Build an undirected graph from framework edges with edge weights = haversine distance (km).
    Nodes carry lon/lat.
    """
    G = nx.Graph()

    reps2 = reps.copy()
    reps2["rep_id"] = reps2["rep_id"].astype(str)

    # nodes (finite coords only)
    for r in reps2.itertuples(index=False):
        rid = str(getattr(r, "rep_id"))
        lon = float(getattr(r, "lon"))
        lat = float(getattr(r, "lat"))
        if not (np.isfinite(lon) and np.isfinite(lat)):
            continue
        G.add_node(rid, lon=lon, lat=lat)

    # edges + weights
    for e in fw_edges.itertuples(index=False):
        u = str(getattr(e, "u"))
        v = str(getattr(e, "v"))
        if u == v or (u not in G.nodes) or (v not in G.nodes):
            continue

        lu = float(G.nodes[u]["lon"])
        la = float(G.nodes[u]["lat"])
        lv = float(G.nodes[v]["lon"])
        lb = float(G.nodes[v]["lat"])
        w = float(haversine_km_vec(np.array([la]), np.array([lu]), float(lb), float(lv))[0])
        if np.isfinite(w):
            G.add_edge(u, v, w_km=w)

    return G


def _path_has_dtw_for_all_adjacent(
    path: List[str],
    paths: Dict[Tuple[str, str], np.ndarray],
) -> bool:
    for a, b in zip(path[:-1], path[1:]):
        if (a, b) in paths or (b, a) in paths:
            continue
        return False
    return True


def _choose_connected_fence_path(
    G: "nx.Graph",
    reps: "pd.DataFrame",
    *,
    paths: Dict[Tuple[str, str], np.ndarray],
    max_wells: int,
    seed: int,
    min_wells: int = 12,
    max_tries: int = 250,
) -> List[str]:
    """
    Pick two wells near opposite edges of the map and compute a shortest path through G.
    Ensures consecutive wells are connected (by construction) and (optionally) that DTW paths exist.
    If the path is longer than max_wells, take a connected contiguous segment (no skipping).
    """
    rng = np.random.default_rng(int(seed))

    reps2 = reps.dropna(subset=["lon", "lat"]).copy()
    reps2["rep_id"] = reps2["rep_id"].astype(str)

    min_lon = float(reps2["lon"].min())
    max_lon = float(reps2["lon"].max())
    min_lat = float(reps2["lat"].min())
    max_lat = float(reps2["lat"].max())

    if not (np.isfinite(min_lon) and np.isfinite(max_lon) and np.isfinite(min_lat) and np.isfinite(max_lat)):
        raise RuntimeError("Invalid lon/lat extents for fence path selection.")

    for _ in range(int(max_tries)):
        orient = "EW" if float(rng.random()) < 0.5 else "NS"

        if orient == "EW":
            latA = float(rng.uniform(min_lat, max_lat))
            lonA = min_lon
            latB = float(rng.uniform(min_lat, max_lat))
            lonB = max_lon
        else:
            lonA = float(rng.uniform(min_lon, max_lon))
            latA = min_lat
            lonB = float(rng.uniform(min_lon, max_lon))
            latB = max_lat

        s = _nearest_rep_id_to_point(reps2, lon0=lonA, lat0=latA)
        t = _nearest_rep_id_to_point(reps2, lon0=lonB, lat0=latB)
        if s == t:
            continue
        if s not in G.nodes or t not in G.nodes:
            continue

        try:
            p = nx.shortest_path(G, source=s, target=t, weight="w_km")
        except Exception:
            continue

        p = [str(x) for x in p]
        if len(p) < int(min_wells):
            continue

        # If DTW paths are provided, require all adjacent pairs to have DTW paths.
        if paths and (not _path_has_dtw_for_all_adjacent(p, paths)):
            continue

        # Crop to a contiguous connected segment if needed (no skipping nodes)
        if len(p) > int(max_wells):
            start = int(rng.integers(0, len(p) - int(max_wells) + 1))
            p = p[start : start + int(max_wells)]

        return p

    raise RuntimeError(
        f"Could not find a connected fence path (min_wells={min_wells}, max_wells={max_wells}) after {max_tries} tries. "
        "This may indicate the framework graph is too disconnected or DTW paths are missing for many edges."
    )


# -----------------------------------------------------------------------------
# Fence geometry + gap fill (DTW-warped gradient)
# -----------------------------------------------------------------------------

def _station_x_left(j: int, *, track_w: float, gap_w: float) -> float:
    return float(j) * (float(track_w) + float(gap_w))


def _interp1_nan_safe(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """
    1D linear interpolation that tolerates NaNs in fp.
    Returns NaN where interpolation is not possible.
    """
    x = np.asarray(x, dtype="float64")
    xp = np.asarray(xp, dtype="float64")
    fp = np.asarray(fp, dtype="float64")

    m = np.isfinite(xp) & np.isfinite(fp)
    if int(m.sum()) < 2:
        return np.full(x.shape, np.nan, dtype="float64")

    xp2 = xp[m]
    fp2 = fp[m]
    o = np.argsort(xp2)
    xp2 = xp2[o]
    fp2 = fp2[o]
    return np.interp(x, xp2, fp2, left=np.nan, right=np.nan)


def _shift_vec(
    shifts_vec_by_rep: Dict[str, np.ndarray],
    rid: str,
    nS: int,
) -> np.ndarray:
    sv = shifts_vec_by_rep.get(str(rid))
    if sv is None:
        return np.zeros((nS,), dtype="float64")
    sv = np.asarray(sv, dtype="float64").reshape(-1)
    if sv.size == 0:
        return np.zeros((nS,), dtype="float64")
    if sv.size == 1:
        return np.full((nS,), float(sv[0]), dtype="float64")

    # clean non-finite
    if not np.all(np.isfinite(sv)):
        med = float(np.nanmedian(sv))
        if not np.isfinite(med):
            med = 0.0
        sv = np.where(np.isfinite(sv), sv, med)

    if sv.size >= nS:
        return sv[:nS].astype("float64", copy=False)

    # pad short vectors with last value (more stable than zeros)
    out = np.empty((nS,), dtype="float64")
    out[: sv.size] = sv
    out[sv.size :] = float(sv[-1])
    return out


def _dtw_map_yB_of_yA(
    *,
    a: str,
    b: str,
    paths: Dict[Tuple[str, str], np.ndarray],
    depth_by_rep: Dict[str, np.ndarray],
    shifts_vec_by_rep: Dict[str, np.ndarray],
    z0: float,
    nS: int,
    max_knots: int = 600,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Build a monotone mapping yB(yA) from DTW path between a and b.
    Returns (yA_knots, yB_knots), both strictly increasing in yA_knots.

    y-coordinates are *relative depth from z0*, after applying per-sample shifts.
    """
    p = paths.get((a, b))
    flip = False
    if p is None:
        p = paths.get((b, a))
        flip = True
    if p is None:
        return None

    p = np.asarray(p, dtype=np.int64)
    if p.ndim != 2 or p.shape[1] != 2 or p.size == 0:
        return None

    ia = p[:, 0].copy()
    ib = p[:, 1].copy()
    if flip:
        ia, ib = ib, ia

    m = (ia >= 0) & (ia < nS) & (ib >= 0) & (ib < nS)
    ia = ia[m]
    ib = ib[m]
    if ia.size < 4:
        return None

    # Optional thinning, but keep enough for a stable mapping
    if ia.size > int(max_knots):
        idx = np.linspace(0, ia.size - 1, int(max_knots), dtype=int)
        ia = ia[idx]
        ib = ib[idx]

    da = depth_by_rep[a][:nS].astype("float64", copy=False)
    db = depth_by_rep[b][:nS].astype("float64", copy=False)
    sa = _shift_vec(shifts_vec_by_rep, a, nS)
    sb = _shift_vec(shifts_vec_by_rep, b, nS)

    # Sort by ia (monotone in A)
    o = np.argsort(ia)
    ia = ia[o]
    ib = ib[o]

    # Collapse duplicate ia using median ib
    uniq_ia, starts = np.unique(ia, return_index=True)
    yA = (da[uniq_ia] + sa[uniq_ia] - float(z0)).astype("float64")

    yB = np.empty_like(yA)
    for k in range(uniq_ia.size):
        s0 = int(starts[k])
        s1 = int(starts[k + 1]) if k + 1 < uniq_ia.size else int(ia.size)
        ib_seg = ib[s0:s1]
        yB[k] = float(np.median(db[ib_seg].astype("float64") + sb[ib_seg].astype("float64") - float(z0)))

    good = np.isfinite(yA) & np.isfinite(yB)
    yA = yA[good]
    yB = yB[good]
    if yA.size < 2:
        return None

    return yA, yB


def _draw_gap_dtw_warped_pcolormesh(
    ax,
    *,
    j: int,
    a: str,
    b: str,
    paths: Dict[Tuple[str, str], np.ndarray],
    depth_by_rep: Dict[str, np.ndarray],
    shifts_vec_by_rep: Dict[str, np.ndarray],
    z0: float,
    nS: int,
    # values on the global y_grid (relative depth grid)
    y_grid: np.ndarray,           # (nS,)
    M_left: np.ndarray,           # (nS,) values for well a on y_grid
    M_right: np.ndarray,          # (nS,) values for well b on y_grid
    track_w: float,
    gap_w: float,
    px_per_unit: float,
    cmap,
    vmin: float = 0.0,
    vmax: float = 1.0,
    zorder: int = 2,
    max_knots: int = 600,
    y_stride: int = 1,
) -> None:
    """
    Fill the gap using a warped raster whose rows are strat-equivalent slices:
      - rows parameterized by yA (reference coordinate in well a)
      - mapped to yB via DTW: yB(yA)
      - colors blended across u between A(yA) and B(yB(yA))
    """
    mapping = _dtw_map_yB_of_yA(
        a=str(a),
        b=str(b),
        paths=paths,
        depth_by_rep=depth_by_rep,
        shifts_vec_by_rep=shifts_vec_by_rep,
        z0=float(z0),
        nS=int(nS),
        max_knots=int(max_knots),
    )
    if mapping is None:
        return
    yA_knots, yB_knots = mapping

    y_grid = np.asarray(y_grid, dtype="float64")
    if y_grid.size != int(nS):
        return

    # Build y edges from y_grid (optionally decimated)
    y_edges = np.empty((y_grid.size + 1,), dtype="float64")
    y_edges[0] = 0.0
    y_edges[-1] = float(y_grid[-1])
    if y_grid.size >= 2:
        y_edges[1:-1] = 0.5 * (y_grid[:-1] + y_grid[1:])

    if int(y_stride) > 1:
        y_edges_ds = y_edges[:: int(y_stride)]
        if y_edges_ds[-1] != y_edges[-1]:
            y_edges_ds = np.concatenate([y_edges_ds, [y_edges[-1]]])
    else:
        y_edges_ds = y_edges

    y_centers_ds = 0.5 * (y_edges_ds[:-1] + y_edges_ds[1:])

    # Map yA -> yB using DTW knots
    yB_edges_ds = np.interp(y_edges_ds, yA_knots, yB_knots, left=yB_knots[0], right=yB_knots[-1])
    yB_centers_ds = np.interp(y_centers_ds, yA_knots, yB_knots, left=yB_knots[0], right=yB_knots[-1])

    # Sample A at yA_centers and B at yB_centers
    Avals = _interp1_nan_safe(y_centers_ds, y_grid, M_left)
    Bvals = _interp1_nan_safe(yB_centers_ds, y_grid, M_right)

    # Horizontal resolution for the gap
    gap_cols = max(12, int(round(float(gap_w) * float(px_per_unit))))
    u_edges = np.linspace(0.0, 1.0, gap_cols + 1, dtype="float64")
    u_centers = 0.5 * (u_edges[:-1] + u_edges[1:])

    # Enforce continuity at endpoints
    if u_centers.size >= 2:
        u_centers[0] = 0.0
        u_centers[-1] = 1.0

    # X edges for the gap (between right edge of track j and left edge of track j+1)
    xl_a = _station_x_left(int(j), track_w=float(track_w), gap_w=float(gap_w))
    xr_a = xl_a + float(track_w)
    x_edges = xr_a + u_edges * float(gap_w)

    # Build X/Y edge grids for pcolormesh
    X = np.repeat(x_edges[None, :], y_edges_ds.size, axis=0)  # (nE, m+1)
    # Shear: Y(u, yA) = (1-u)*yA + u*yB(yA)
    Y = (1.0 - u_edges[None, :]) * y_edges_ds[:, None] + u_edges[None, :] * yB_edges_ds[:, None]

    # Colors at cell centers: blend per row between A(yA) and B(yB)
    a2 = Avals[:, None]
    b2 = Bvals[:, None]
    u2 = u_centers[None, :]

    base = (1.0 - u2) * a2 + u2 * b2
    C = np.full(base.shape, np.nan, dtype="float64")

    ma = np.isfinite(a2)
    mb = np.isfinite(b2)
    C = np.where(ma & mb, base, C)
    C = np.where(ma & (~mb), a2, C)
    C = np.where((~ma) & mb, b2, C)

    C = np.clip(C, float(vmin), float(vmax))

    ax.pcolormesh(
        X,
        Y,
        C,
        cmap=cmap,
        vmin=float(vmin),
        vmax=float(vmax),
        shading="auto",
        edgecolors="none",
        antialiased=False,
        zorder=int(zorder),
    )


def _expand_tracks_with_nan_gaps(
    M: np.ndarray,
    *,
    track_w: float,
    gap_w: float,
    px_per_unit: float = 24.0,
) -> Tuple[np.ndarray, int, int]:
    """
    Expand (nS, W) matrix M into (nS, NX) with:
      - each well column replicated across track_px pixels
      - each inter-well gap filled with NaNs (later overdrawn with DTW-warped gradient)

    Returns:
      Mx, track_px, gap_px
    """
    if M.ndim != 2:
        raise ValueError("M must be 2D (nS, W)")
    nS, W = M.shape
    if W < 1:
        raise ValueError("M has no columns")

    track_px = max(2, int(round(float(track_w) * float(px_per_unit))))
    gap_px = max(1, int(round(float(gap_w) * float(px_per_unit)))) if W > 1 else 0

    NX = W * track_px + max(0, W - 1) * gap_px
    Mx = np.full((nS, NX), np.nan, dtype="float64")

    col = 0
    for j in range(W):
        Mx[:, col : col + track_px] = M[:, j : j + 1]
        col += track_px
        if j < W - 1 and gap_px > 0:
            col += gap_px  # leave as NaN
    return Mx, track_px, gap_px


# -----------------------------------------------------------------------------
# Plot Step 3d: framework map + connected fence
# -----------------------------------------------------------------------------

ShiftLike = Union[Dict[str, float], Dict[str, np.ndarray]]


def _load_optional_shifts_npz(path: Optional[Path]) -> Optional[Dict[str, np.ndarray]]:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        from strataframe.rgt.shifts_npz import load_shifts_npz  # type: ignore
        out = load_shifts_npz(p)
        # normalize keys to str, values to 1D float64
        return {str(k): np.asarray(v, dtype="float64").reshape(-1) for k, v in out.items()}
    except Exception:
        return None


def plot_step3d_framework_and_fence(
    *,
    reps: "pd.DataFrame",
    rep_arrays_npz: Path,
    framework_edges: "pd.DataFrame",
    dtw_paths_npz: Optional[Path],
    out_map_png: Path,
    out_fence_png: Path,
    max_edges: int = 120_000,
    max_wells_fence: int = 60,
    seed: int = 42,
    title_map: str = "Step 3d — Framework edges",
    title_fence: str = "Step 3d — Fence view (track+gap gradient fill + black trace + DTW tie-lines; hung to datum)",
    tie_color: str = "#0B2E83",
    tie_alpha: float = 0.22,
    tie_lw: float = 0.55,
    out_order_csv: Optional[Path] = None,
    # geometry controls:
    track_w: float = 1.0,
    gap_w: float = 1.25,
    px_per_unit: float = 28.0,
    # trace placement inside track:
    x_pad_frac: float = 0.05,
    x_scale_frac: float = 0.90,
    # OPTIONAL: if provided, overrides DTW-derived scalar hanging shifts
    shifts_npz: Optional[Path] = None,
) -> None:
    out_map_png = Path(out_map_png)
    out_fence_png = Path(out_fence_png)
    out_map_png.parent.mkdir(parents=True, exist_ok=True)
    out_fence_png.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Map: framework edges
    # -------------------------
    fig, ax = plt.subplots(figsize=(11, 9))

    lc, n_used = line_collection_from_edges(reps, framework_edges, max_edges=int(max_edges))
    lc.set_color("0.20")
    lc.set_alpha(0.65)
    try:
        lc.set_linewidth(1.0)
    except Exception:
        pass
    ax.add_collection(lc)

    ax.scatter(reps["lon"], reps["lat"], s=10, alpha=0.85)
    ax.text(0.01, 0.01, f"framework edges plotted: {n_used:,}", transform=ax.transAxes, fontsize=9)

    ax.set_title(str(title_map))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(out_map_png, dpi=175)
    plt.close(fig)

    # -------------------------
    # Fence: load arrays + DTW paths
    # -------------------------
    z = load_npz(rep_arrays_npz)
    rep_ids_npz, depth, log, _imputed = rep_arrays_from_npz(z)

    depth_by_rep: Dict[str, np.ndarray] = {
        str(rep_ids_npz[i]): depth[i, :].astype("float64", copy=False) for i in range(rep_ids_npz.size)
    }
    log_by_rep: Dict[str, np.ndarray] = {
        str(rep_ids_npz[i]): np.clip(log[i, :].astype("float64", copy=False), 0.0, 1.0)
        for i in range(rep_ids_npz.size)
    }

    paths = load_dtw_paths_dict(dtw_paths_npz)

    # Filter reps to those we can plot (coords + arrays)
    reps2 = reps.copy()
    reps2["rep_id"] = reps2["rep_id"].astype(str)
    reps2 = reps2[reps2["rep_id"].isin(depth_by_rep.keys())].dropna(subset=["lon", "lat"]).copy()
    if reps2.shape[0] < 2:
        raise RuntimeError("Not enough wells with both coordinates and rep_arrays to render Step 3d fence.")

    # -------------------------
    # Fence ordering
    # -------------------------
    Gfw = _build_framework_graph_nx(reps2, framework_edges)

    order = _choose_connected_fence_path(
        Gfw,
        reps2,
        paths=paths,
        max_wells=int(max_wells_fence),
        seed=int(seed),
        min_wells=min(12, int(max_wells_fence)),
    )
    if not order:
        raise RuntimeError("Failed to select a connected fence path (order is empty).")

    # -------------------------
    # Decide per-sample shifts
    #   - Prefer explicit shifts_npz (rep_id -> (nS,) float64)
    #   - Else: DTW-derived scalar hanging shifts (constant per well)
    # -------------------------
    nS = int(min(depth_by_rep[rid].size for rid in order))
    if nS < 8:
        raise RuntimeError("rep_arrays depth/log length too small for Step 3d fence plotting.")

    shifts_vec_by_rep: Dict[str, np.ndarray] = {}

    shifts_loaded = _load_optional_shifts_npz(shifts_npz)
    if shifts_loaded:
        for rid in order:
            shifts_vec_by_rep[str(rid)] = _shift_vec(shifts_loaded, str(rid), nS)
    else:
        shifts_scalar = {rid: 0.0 for rid in order}
        if paths:
            shifts_scalar = _scalar_shifts_from_dtw_paths(order, framework_edges, depth_by_rep, paths)
        for rid in order:
            shifts_vec_by_rep[str(rid)] = np.full((nS,), float(shifts_scalar.get(str(rid), 0.0)), dtype="float64")

    # -------------------------
    # Common depth grid (global range across shifted traces)
    # -------------------------
    zmins: List[float] = []
    zmaxs: List[float] = []
    for rid in order:
        dz = depth_by_rep[rid][:nS].astype("float64", copy=False) + _shift_vec(shifts_vec_by_rep, rid, nS)
        zmins.append(float(np.nanmin(dz)))
        zmaxs.append(float(np.nanmax(dz)))

    z0 = float(np.min(zmins))
    z1 = float(np.max(zmaxs))
    if not np.isfinite(z0) or not np.isfinite(z1) or z1 <= z0:
        raise RuntimeError("Invalid depth range for fence plot.")

    z_grid = np.linspace(z0, z1, nS).astype("float64")

    # Relative depth axis: 0 at top, increasing downward
    y_grid = (z_grid - z0).astype("float64")
    y_max = float(y_grid[-1])

    # -------------------------
    # Build matrix M (nS, W) on common depth grid
    # -------------------------
    W = len(order)
    M = np.full((nS, W), np.nan, dtype="float64")

    for j, rid in enumerate(order):
        z_shift = depth_by_rep[rid][:nS].astype("float64", copy=False) + _shift_vec(shifts_vec_by_rep, rid, nS)
        x = log_by_rep[rid][:nS]
        M[:, j] = np.interp(z_grid, z_shift, x, left=np.nan, right=np.nan)

    # Expand to track+gap fill with NaN gaps
    Mx, _track_px, _gap_px = _expand_tracks_with_nan_gaps(
        M,
        track_w=float(track_w),
        gap_w=float(gap_w),
        px_per_unit=float(px_per_unit),
    )

    # Total fence width in "units"
    total_w = float(W) * float(track_w) + float(max(0, W - 1)) * float(gap_w)

    cmap = yellow_brown_perceptual_cmap()
    try:
        cmap = cmap.copy()
        cmap.set_bad(color="#ffffff")
    except Exception:
        pass

    # Size based on expanded width so gaps render visibly
    nx_pix = int(Mx.shape[1])
    fig_w = max(14.0, min(30.0, float(nx_pix) / 140.0))
    fig, ax = plt.subplots(figsize=(fig_w, 9))

    # Background fill (tracks + blank gaps)
    im = ax.imshow(
        Mx,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        extent=[0.0, total_w, float(y_max), 0.0],
        zorder=1,
    )

    # DTW-consistent warped raster fill in each GAP
    if paths:
        for j, (a, b) in enumerate(zip(order[:-1], order[1:])):
            _draw_gap_dtw_warped_pcolormesh(
                ax,
                j=int(j),
                a=str(a),
                b=str(b),
                paths=paths,
                depth_by_rep=depth_by_rep,
                shifts_vec_by_rep=shifts_vec_by_rep,
                z0=float(z0),
                nS=int(nS),
                y_grid=y_grid,
                M_left=M[:, j],
                M_right=M[:, j + 1],
                track_w=float(track_w),
                gap_w=float(gap_w),
                px_per_unit=float(px_per_unit),
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
                zorder=2,
                max_knots=800,
                y_stride=1,
            )

    # Track boundaries
    for j in range(W):
        xl = _station_x_left(j, track_w=float(track_w), gap_w=float(gap_w))
        xr = xl + float(track_w)
        ax.plot([xl, xl], [0.0, y_max], linewidth=0.35, alpha=0.30, color="0.0", zorder=2)
        ax.plot([xr, xr], [0.0, y_max], linewidth=0.35, alpha=0.30, color="0.0", zorder=2)

    # Black trace inside each track
    x_pad = float(x_pad_frac) * float(track_w)
    x_scale = float(x_scale_frac) * float(track_w)
    for j, rid in enumerate(order):
        xl = _station_x_left(j, track_w=float(track_w), gap_w=float(gap_w))
        colv = M[:, j]
        m = np.isfinite(colv)
        if int(m.sum()) < 5:
            continue
        ax.plot(
            xl + x_pad + x_scale * colv[m],
            y_grid[m],
            color="0.0",
            linewidth=0.60,
            alpha=0.95,
            zorder=3,
        )

    # DTW tie-lines: stair-step (flat across track, diagonal across gap)
    segs: List[List[List[float]]] = []
    if paths:
        for j, (a, b) in enumerate(zip(order[:-1], order[1:])):
            p = paths.get((a, b))
            flip = False
            if p is None:
                p = paths.get((b, a))
                flip = True
            if p is None:
                continue

            p = np.asarray(p, dtype=np.int64)
            if p.ndim != 2 or p.shape[1] != 2 or p.size == 0:
                continue

            ia = p[:, 0].copy()
            ib = p[:, 1].copy()
            if flip:
                ia, ib = ib, ia

            # thin tie-lines for readability
            target = int(35 + 20 * float(gap_w))
            step = max(1, int(len(ia) // target))
            ia = ia[::step]
            ib = ib[::step]

            m = (ia >= 0) & (ia < nS) & (ib >= 0) & (ib < nS)
            ia = ia[m]
            ib = ib[m]
            if ia.size == 0:
                continue

            da = depth_by_rep[a][:nS].astype("float64", copy=False)
            db = depth_by_rep[b][:nS].astype("float64", copy=False)
            sa = _shift_vec(shifts_vec_by_rep, a, nS)
            sb = _shift_vec(shifts_vec_by_rep, b, nS)

            ya = (da[ia] + sa[ia] - float(z0)).astype("float64")
            yb = (db[ib] + sb[ib] - float(z0)).astype("float64")

            xl_a = _station_x_left(j, track_w=float(track_w), gap_w=float(gap_w))
            xr_a = xl_a + float(track_w)
            xl_b = _station_x_left(j + 1, track_w=float(track_w), gap_w=float(gap_w))
            xr_b = xl_b + float(track_w)

            for y1, y2 in zip(ya, yb):
                if not (np.isfinite(y1) and np.isfinite(y2)):
                    continue
                segs.append([[float(xl_a), float(y1)], [float(xr_a), float(y1)]])
                segs.append([[float(xr_a), float(y1)], [float(xl_b), float(y2)]])
                segs.append([[float(xl_b), float(y2)], [float(xr_b), float(y2)]])

    if segs:
        lc_tie = LineCollection(segs, colors=str(tie_color), linewidths=float(tie_lw), alpha=float(tie_alpha), zorder=4)
        ax.add_collection(lc_tie)

    # Axes + colorbar
    ax.set_title(str(title_fence))
    ax.set_xlabel("Fence station (well index along section)")
    ax.set_ylabel("Relative depth from datum (0 at top; increasing downward)")
    ax.set_ylim(y_max, 0.0)
    ax.set_xlim(0.0, total_w)

    centers = [
        (_station_x_left(j, track_w=float(track_w), gap_w=float(gap_w)) + 0.5 * float(track_w))
        for j in range(W)
    ]
    step_tick = max(1, W // 12)
    ax.set_xticks([centers[i] for i in range(0, W, step_tick)])
    ax.set_xticklabels([str(i + 1) for i in range(0, W, step_tick)], fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Normalized GR (0→low, 1→high)")

    # Inset map: polyline through fence wells (A → … → A′)
    axins = ax.inset_axes([0.70, 0.68, 0.28, 0.28])
    axins.scatter(reps["lon"], reps["lat"], s=6, alpha=0.25, color="0.3")

    reps_idx = reps.copy()
    reps_idx["rep_id"] = reps_idx["rep_id"].astype(str)
    reps_idx = reps_idx.drop_duplicates(subset=["rep_id"]).set_index("rep_id")

    reps_sel = reps_idx.loc[order, ["lon", "lat"]].reset_index()
    axins.plot(reps_sel["lon"], reps_sel["lat"], linewidth=1.5, alpha=0.95, color="0.0")
    axins.scatter(reps_sel["lon"], reps_sel["lat"], s=10, alpha=0.90, color="0.0")

    A = reps_sel.iloc[0]
    Apr = reps_sel.iloc[-1]
    axins.text(float(A["lon"]), float(A["lat"]), "A", fontsize=10, fontweight="bold", ha="right", va="bottom")
    axins.text(float(Apr["lon"]), float(Apr["lat"]), "A'", fontsize=10, fontweight="bold", ha="left", va="top")

    axins.set_title("Fence A–A′", fontsize=9)
    axins.set_xticks([])
    axins.set_yticks([])
    axins.set_aspect("equal", adjustable="datalim")

    # Write fence order CSV
    if out_order_csv is None:
        out_order_csv = out_fence_png.parent / "step3d_fence_order.csv"
    try:
        out_order_csv = Path(out_order_csv)
        out_order_csv.parent.mkdir(parents=True, exist_ok=True)
        out_rows = reps_sel.copy()
        out_rows.insert(0, "station", np.arange(1, len(reps_sel) + 1))
        out_rows.to_csv(out_order_csv, index=False)
    except Exception:
        pass

    fig.subplots_adjust(right=0.92, top=0.95, bottom=0.08)
    fig.savefig(out_fence_png, dpi=175)
    plt.close(fig)
