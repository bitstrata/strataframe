# src/strataframe/viz/step3b_panel.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from .step3_colors import yellow_brown_perceptual_cmap
from .step3_common import line_collection_from_edges

try:
    import pandas as pd  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("pandas is required. pip install pandas") from e


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _evenly_subsample(order: Sequence[str], k: int) -> List[str]:
    order = list(order)
    if k <= 0 or len(order) <= k:
        return order
    idx = np.linspace(0, len(order) - 1, int(k), dtype=int)
    # unique, stable
    seen: set[str] = set()
    out: List[str] = []
    for i in idx.tolist():
        rid = str(order[int(i)])
        if rid not in seen:
            out.append(rid)
            seen.add(rid)
    return out


def _coerce_1d_float(a: Any) -> Optional[np.ndarray]:
    try:
        x = np.asarray(a, dtype="float64").reshape(-1)
        if x.size == 0:
            return None
        return x
    except Exception:
        return None


ShiftValue = Union[float, int, np.ndarray]


def _get_shift_for_rep(
    rid: str,
    *,
    shifts: Optional[Dict[str, ShiftValue]],
    n: int,
) -> Tuple[Optional[float], Optional[np.ndarray]]:
    """
    Returns either:
      - (scalar_shift, None) OR
      - (None, shift_vec_of_len_n)

    Accepted shift forms per rep:
      - scalar (float/int)
      - 1D array-like (will be resized/interpolated to length n if needed)
    """
    if not shifts:
        return 0.0, None

    v = shifts.get(str(rid))
    if v is None:
        return 0.0, None

    # scalar
    if isinstance(v, (float, int, np.floating, np.integer)):
        return float(v), None

    sv = _coerce_1d_float(v)
    if sv is None:
        return 0.0, None

    if sv.size == 1:
        return float(sv[0]), None

    if int(sv.size) == int(n):
        return None, sv.astype("float64", copy=False)

    # Resize shift vector to length n (index-domain interp)
    t_src = np.linspace(0.0, 1.0, int(sv.size), dtype="float64")
    t_dst = np.linspace(0.0, 1.0, int(n), dtype="float64")
    sv2 = np.interp(t_dst, t_src, sv.astype("float64", copy=False)).astype("float64")
    return None, sv2


def _sorted_unique_xy(z: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare arrays for np.interp:
      - filter finite
      - sort by z
      - enforce strictly increasing z by dropping duplicates
    """
    z = np.asarray(z, dtype="float64").reshape(-1)
    x = np.asarray(x, dtype="float64").reshape(-1)
    m = np.isfinite(z) & np.isfinite(x)
    z = z[m]
    x = x[m]
    if z.size < 2:
        return z, x

    o = np.argsort(z, kind="mergesort")
    z = z[o]
    x = x[o]

    # drop duplicate z (keep first occurrence; stable due to mergesort)
    if z.size >= 2:
        dz = np.diff(z)
        if np.any(dz <= 0):
            _, keep_idx = np.unique(z, return_index=True)
            keep_idx.sort()
            z = z[keep_idx]
            x = x[keep_idx]
    return z, x


def _plot_geom_outline(ax, geom, *, color: str, lw: float, alpha: float) -> None:
    gtype = getattr(geom, "geom_type", "")
    if gtype == "Polygon":
        x, y = geom.exterior.xy
        ax.plot(x, y, color=color, linewidth=lw, alpha=alpha)
        for ring in getattr(geom, "interiors", []):
            try:
                xi, yi = ring.xy
                ax.plot(xi, yi, color=color, linewidth=max(0.4, lw * 0.7), alpha=alpha * 0.7)
            except Exception:
                continue
        return
    if gtype == "MultiPolygon":
        for g in geom.geoms:
            _plot_geom_outline(ax, g, color=color, lw=lw, alpha=alpha)
        return
    if gtype in ("LineString", "LinearRing"):
        try:
            x, y = geom.xy
            ax.plot(x, y, color=color, linewidth=lw, alpha=alpha)
        except Exception:
            pass
        return
    if gtype == "MultiLineString":
        for g in geom.geoms:
            _plot_geom_outline(ax, g, color=color, lw=lw, alpha=alpha)


def _plot_kansas_outline(ax, *, color: str = "0.25", lw: float = 0.6, alpha: float = 0.6) -> None:
    """
    Best-effort Kansas outline. Uses Natural Earth via cartopy if available,
    otherwise falls back to a simple bounding-box outline.
    """
    try:
        import cartopy.io.shapereader as shpreader  # type: ignore
    except Exception:
        shpreader = None

    if shpreader is not None:
        try:
            shp = shpreader.natural_earth(
                resolution="50m",
                category="cultural",
                name="admin_1_states_provinces",
            )
            reader = shpreader.Reader(shp)
            geoms = []
            for rec in reader.records():
                attrs = rec.attributes or {}
                name = (attrs.get("name") or attrs.get("name_en") or "").strip()
                admin = (attrs.get("admin") or "").strip()
                if name == "Kansas" and (admin == "" or admin == "United States of America"):
                    geoms.append(rec.geometry)
            if geoms:
                for g in geoms:
                    _plot_geom_outline(ax, g, color=color, lw=lw, alpha=alpha)
                return
        except Exception:
            pass

    # Fallback: rough bounding box
    ks_lon = [-102.05, -94.6, -94.6, -102.05, -102.05]
    ks_lat = [37.0, 37.0, 40.0, 40.0, 37.0]
    ax.plot(ks_lon, ks_lat, color=color, linewidth=lw, alpha=alpha)


# -----------------------------------------------------------------------------
# Step 3b — fence-like panel (common depth axis)
# -----------------------------------------------------------------------------

def plot_step3b_rep_arrays_panel(
    rep_arrays: dict[str, dict[str, np.ndarray]],
    rep_ids: list[str],
    out_png: str,
    *,
    max_wells: int = 200,
    seed: int = 42,  # kept for API stability (ordering is deterministic; seed unused)
    show_trace: bool = True,
    # NEW:
    shifts: Optional[Dict[str, ShiftValue]] = None,
    depth_key: str = "depth_rs",
    log_key: str = "log_rs",
    pad_frac: float = 0.01,  # small padding to avoid hard clipping
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    gap_w: float = 0.0,
    show_correlations: bool = False,
    correlation_n: int = 35,
    correlation_color: str = "#2E6FDB",
    correlation_alpha: float = 0.35,
    correlation_lw: float = 0.6,
    map_nodes: Optional["pd.DataFrame"] = None,
    map_title: Optional[str] = None,
) -> None:
    """
    Step 3b viz (fence-like panel) with a COMMON DEPTH AXIS:
      - wells side-by-side (each well is a 1.0-wide track)
      - y-axis is depth units (from depth_key), increasing downward
      - traces are optionally "hung" by shifts:
          * scalar shift: z' = z + shift
          * vector shift (len nS): z'[i] = z[i] + shift[i]
      - continuous 2-color perceptual yellow->brown ramp
      - optional GR trace overlay
      - optional correlation lines between adjacent wells (RGT ties)
      - optional map showing the path through the rep_ids

    Notes on shift vectors:
      If z' becomes non-monotonic, the function sorts by z' and drops duplicates
      to keep np.interp stable. This is visualization-focused, not a strict geologic transform.
    """
    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Deterministic ordering: preserve given rep_ids order; just filter and subsample evenly if needed
    rep_ids = [str(r) for r in rep_ids if str(r) in rep_arrays]
    if not rep_ids:
        raise RuntimeError("No rep_ids available for 3b panel (rep_arrays empty or IDs mismatch).")

    if len(rep_ids) > int(max_wells):
        rep_ids = _evenly_subsample(rep_ids, int(max_wells))

    # Gather per-well arrays; track a common sample count nS
    depth_by_rep: Dict[str, np.ndarray] = {}
    log_by_rep: Dict[str, np.ndarray] = {}
    nS: Optional[int] = None

    for rid in rep_ids:
        a = rep_arrays.get(str(rid), {})
        z = _coerce_1d_float(a.get(depth_key))
        x = _coerce_1d_float(a.get(log_key))
        if z is None or x is None:
            continue
        if z.size == 0 or x.size == 0:
            continue

        # trim to common length across wells (defensive)
        n = int(min(z.size, x.size))
        if n < 8:
            continue

        if nS is None:
            nS = n
        else:
            nS = int(min(nS, n))

        depth_by_rep[str(rid)] = z[:n]
        # Normalize/clamp to [0,1] to match the ramp
        log_by_rep[str(rid)] = np.clip(x[:n], 0.0, 1.0)

    if not depth_by_rep or nS is None or int(nS) < 8:
        raise RuntimeError(f"No usable ({depth_key}, {log_key}) arrays found for 3b panel.")

    # Final trim to common nS (and trim any shift vectors later to nS)
    nS = int(nS)
    for rid in list(depth_by_rep.keys()):
        depth_by_rep[rid] = depth_by_rep[rid][:nS]
        log_by_rep[rid] = log_by_rep[rid][:nS]

    # Build global shifted depth range
    z_mins: List[float] = []
    z_maxs: List[float] = []

    for rid in rep_ids:
        if rid not in depth_by_rep:
            continue
        z = depth_by_rep[rid]
        s_scalar, s_vec = _get_shift_for_rep(rid, shifts=shifts, n=nS)
        if s_vec is not None:
            z2 = z + s_vec
        else:
            z2 = z + float(s_scalar or 0.0)
        if np.isfinite(z2).any():
            z_mins.append(float(np.nanmin(z2)))
            z_maxs.append(float(np.nanmax(z2)))

    if not z_mins or not z_maxs:
        raise RuntimeError("Could not determine global depth range for 3b panel.")

    z0 = float(np.min(z_mins))
    z1 = float(np.max(z_maxs))
    if not np.isfinite(z0) or not np.isfinite(z1) or z1 <= z0:
        raise RuntimeError("Invalid global depth range for 3b panel.")

    # Pad slightly so the image isn't tight-clipped
    span = z1 - z0
    z0 -= float(pad_frac) * span
    z1 += float(pad_frac) * span

    # Common depth grid (same vertical scale for all wells)
    z_grid = np.linspace(z0, z1, nS).astype("float64")

    # Build matrix on common depth grid (nS, W)
    cols: List[np.ndarray] = []
    used_ids: List[str] = []

    for rid in rep_ids:
        if rid not in depth_by_rep:
            continue

        z = depth_by_rep[rid]
        x = log_by_rep[rid]

        s_scalar, s_vec = _get_shift_for_rep(rid, shifts=shifts, n=nS)
        if s_vec is not None:
            z2 = z + s_vec
        else:
            z2 = z + float(s_scalar or 0.0)

        # prep for interp (sorted + unique z2)
        z2s, xs = _sorted_unique_xy(z2, x)
        if z2s.size < 2:
            continue

        # outside range -> NaN so background stays neutral
        x_rs = np.interp(z_grid, z2s, xs, left=np.nan, right=np.nan).astype("float64")
        cols.append(x_rs)
        used_ids.append(str(rid))

    if not cols:
        raise RuntimeError("No columns could be built on the common depth grid for 3b panel.")

    M = np.stack(cols, axis=1)  # (nS, W)
    W = int(M.shape[1])

    cmap = yellow_brown_perceptual_cmap()
    # ensure missing values don't introduce hue; use white background
    try:
        cmap = cmap.copy()
        cmap.set_bad(color="#ffffff")
    except Exception:
        pass

    # Optional map (below panel) to show the path
    map_df: Optional["pd.DataFrame"] = None
    map_id_col: Optional[str] = None
    if map_nodes is not None:
        try:
            map_df = map_nodes.copy()
            if "node_id" in map_df.columns:
                map_id_col = "node_id"
            elif "rep_id" in map_df.columns:
                map_id_col = "rep_id"
            else:
                map_df = None
            if map_df is not None:
                if "lon" not in map_df.columns or "lat" not in map_df.columns:
                    map_df = None
            if map_df is not None and map_id_col is not None:
                map_df[map_id_col] = map_df[map_id_col].astype(str)
                map_df["lon"] = pd.to_numeric(map_df["lon"], errors="coerce")
                map_df["lat"] = pd.to_numeric(map_df["lat"], errors="coerce")
                map_df = map_df.dropna(subset=["lon", "lat"])
        except Exception:
            map_df = None
            map_id_col = None

    use_map = map_df is not None and map_id_col is not None and len(map_df) > 0

    # Track geometry (insert gaps to allow correlation ties)
    track_w = 1.0
    gap_w = float(max(0.0, gap_w))
    x_lefts: List[float] = []
    x_cur = 0.0
    for j in range(W):
        x_lefts.append(x_cur)
        x_cur += float(track_w)
        if j < W - 1:
            x_cur += float(gap_w)
    total_w = float(x_cur)
    x_centers = [xl + 0.5 * float(track_w) for xl in x_lefts]

    # Figure sizing: compress x aggressively
    fig_w_base = float(max(7.0, min(18.0, 0.06 * max(total_w, 1.0))))
    fig_h = 10.0 if use_map else 9.0
    if use_map:
        fig = plt.figure(figsize=(fig_w_base, fig_h), dpi=200)
        gs = fig.add_gridspec(
            nrows=2,
            ncols=1,
            height_ratios=[1.0, 0.28],
            hspace=0.12,
        )
        ax = fig.add_subplot(gs[0, 0])
        ax_map = fig.add_subplot(gs[1, 0])
    else:
        fig, ax = plt.subplots(figsize=(fig_w_base, fig_h), dpi=200)

    # Build plot matrix with optional gap columns
    if gap_w > 0.0:
        cols_gap: List[np.ndarray] = []
        widths: List[float] = []
        for j in range(W):
            cols_gap.append(M[:, j])
            widths.append(float(track_w))
            if j < W - 1:
                cols_gap.append(np.full((nS,), np.nan, dtype="float64"))
                widths.append(float(gap_w))
        M_plot = np.stack(cols_gap, axis=1)
        x_edges = np.concatenate([[0.0], np.cumsum(widths)]).astype("float64")
    else:
        M_plot = M
        x_edges = np.linspace(0.0, float(track_w) * float(W), int(W) + 1, dtype="float64")

    # y edges for pcolormesh
    y_edges = np.empty((nS + 1,), dtype="float64")
    y_edges[0] = float(z_grid[0])
    y_edges[-1] = float(z_grid[-1])
    if nS >= 2:
        y_edges[1:-1] = 0.5 * (z_grid[:-1] + z_grid[1:])

    im = ax.pcolormesh(
        x_edges,
        y_edges,
        M_plot,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        shading="auto",
    )

    # Track boundaries
    for xl in x_lefts:
        xr = float(xl) + float(track_w)
        ax.plot([xl, xl], [z0, z1], linewidth=0.25, alpha=0.25, color="0.0")
        ax.plot([xr, xr], [z0, z1], linewidth=0.25, alpha=0.25, color="0.0")

    # Trace overlay: x = x_left + GR * track_w
    if bool(show_trace):
        for j in range(W):
            xcol = M[:, j]
            m = np.isfinite(xcol)
            if int(m.sum()) < 5:
                continue
            ax.plot(
                float(x_lefts[j]) + xcol[m] * float(track_w),
                z_grid[m],
                linewidth=0.45,
                alpha=0.85,
                color="0.10",
            )

    # Correlation lines (RGT ties) between adjacent wells
    if bool(show_correlations) and W > 1:
        rgt_by_rep: Dict[str, np.ndarray] = {}
        for rid in used_ids:
            z = depth_by_rep.get(str(rid))
            if z is None:
                continue
            s_scalar, s_vec = _get_shift_for_rep(str(rid), shifts=shifts, n=nS)
            if s_vec is not None:
                rgt = z + s_vec
            else:
                rgt = z + float(s_scalar or 0.0)
            rgt_by_rep[str(rid)] = rgt[:nS]

        n_ties = int(max(4, min(int(correlation_n), nS)))
        tie_idx = np.linspace(0, nS - 1, n_ties, dtype=int)
        tie_idx = np.unique(tie_idx)

        segs: List[List[List[float]]] = []
        for j in range(W - 1):
            a = used_ids[j]
            b = used_ids[j + 1]
            ra = rgt_by_rep.get(str(a))
            rb = rgt_by_rep.get(str(b))
            if ra is None or rb is None:
                continue
            xl_a = float(x_lefts[j])
            xr_a = xl_a + float(track_w)
            xl_b = float(x_lefts[j + 1])
            xr_b = xl_b + float(track_w)
            for k in tie_idx:
                y1 = float(ra[int(k)])
                y2 = float(rb[int(k)])
                if not (np.isfinite(y1) and np.isfinite(y2)):
                    continue
                segs.append([[xl_a, y1], [xr_a, y1]])
                segs.append([[xr_a, y1], [xl_b, y2]])
                segs.append([[xl_b, y2], [xr_b, y2]])

        if segs:
            lc = LineCollection(
                segs,
                colors=str(correlation_color),
                linewidths=float(correlation_lw),
                alpha=float(correlation_alpha),
                zorder=5,
            )
            ax.add_collection(lc)

    ax.set_xlim(0.0, float(total_w))
    ax.set_ylim(float(z1), float(z0))

    ax.set_title(
        title
        if title is not None
        else "Step 3b: Representative GR arrays — common depth axis (hung to shared datum)",
        pad=22,
    )
    ax.set_xlabel(xlabel if xlabel is not None else "Representative wells (topology + geography)")
    ax.set_ylabel(ylabel if ylabel is not None else "Depth (same units as LAS; increasing downward)")

    # X labels (well identifiers) placed above the panel
    ax.set_xticks([])
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(x_centers)
    ax_top.set_xticklabels(used_ids, rotation=90, fontsize=6)
    ax_top.xaxis.set_ticks_position("top")
    ax_top.tick_params(axis="x", pad=2, length=0)
    ax_top.set_yticks([])
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["bottom"].set_visible(False)
    ax_top.spines["left"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    # Y ticks
    yt = np.linspace(z0, z1, 8)
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{v:.0f}" for v in yt])

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Normalized GR (0→low, 1→high)")

    # Optional map (below) showing the path through the wells
    if use_map:
        map_idx = map_df.drop_duplicates(subset=[map_id_col]).set_index(map_id_col)
        order_ids = [rid for rid in used_ids if rid in map_idx.index]
        reps_sel = map_idx.loc[order_ids, ["lon", "lat"]].reset_index() if order_ids else None

        _plot_kansas_outline(ax_map, color="0.25", lw=0.6, alpha=0.6)
        ax_map.scatter(map_df["lon"], map_df["lat"], s=6, alpha=0.25, color="0.3")
        if reps_sel is not None and len(reps_sel) > 0:
            ax_map.plot(reps_sel["lon"], reps_sel["lat"], linewidth=1.5, alpha=0.95, color="0.0")
            ax_map.scatter(reps_sel["lon"], reps_sel["lat"], s=10, alpha=0.90, color="0.0")

            if len(reps_sel) >= 2:
                A = reps_sel.iloc[0]
                Apr = reps_sel.iloc[-1]
                ax_map.text(float(A["lon"]), float(A["lat"]), "A", fontsize=9, fontweight="bold",
                            ha="right", va="bottom")
                ax_map.text(float(Apr["lon"]), float(Apr["lat"]), "A'", fontsize=9, fontweight="bold",
                            ha="left", va="top")

        ax_map.set_title(map_title if map_title is not None else "Panel path", fontsize=9)
        ax_map.set_xticks([])
        ax_map.set_yticks([])
        ax_map.set_aspect("equal", adjustable="datalim")

    if use_map:
        fig.subplots_adjust(top=0.90, left=0.08, right=0.96, bottom=0.08)
    else:
        fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Step 3c — DTW edges overlay map (belongs with Step 3b/3c viz set)
# -----------------------------------------------------------------------------

def plot_step3c_dtw_map(
    *,
    reps: "pd.DataFrame",
    candidate_edges: Optional["pd.DataFrame"],
    dtw_edges: "pd.DataFrame",
    out_png: Path,
    max_edges: int = 80_000,
    title: str = "Step 3c — DTW edges (overlay on candidates)",
) -> None:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 9))

    # candidates: light grey
    if candidate_edges is not None and len(candidate_edges) > 0:
        lc_c, n_used = line_collection_from_edges(reps, candidate_edges, max_edges=int(max_edges))
        lc_c.set_color("0.80")
        lc_c.set_alpha(0.25)
        ax.add_collection(lc_c)
        ax.text(0.01, 0.01, f"candidate edges plotted: {n_used:,}", transform=ax.transAxes, fontsize=9)

    # dtw edges: darker
    lc_d, n_used_d = line_collection_from_edges(reps, dtw_edges, max_edges=int(max_edges))
    lc_d.set_color("0.20")
    lc_d.set_alpha(0.65)
    try:
        lc_d.set_linewidth(0.9)
    except Exception:
        pass
    ax.add_collection(lc_d)
    ax.text(0.01, 0.04, f"dtw edges plotted: {n_used_d:,}", transform=ax.transAxes, fontsize=9)

    # nodes
    ax.scatter(reps["lon"], reps["lat"], s=10, alpha=0.85)

    ax.set_title(str(title))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(out_png, dpi=175)
    plt.close(fig)
