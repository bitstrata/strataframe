# src/strataframe/rgt/chronostrat.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

try:
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover
    nx = None  # type: ignore

from .monotonic import MonotonicConfig, enforce_monotonic_rgt


@dataclass(frozen=True)
class ChronostratConfig:
    """
    Constructs a chronostrat diagram on a common RGT grid.
    """
    n_rgt: int = 800
    rgt_pad_frac: float = 0.02  # pad min/max by this fraction of span
    monotonic: MonotonicConfig = MonotonicConfig(mode="nondecreasing")


def _require_nx() -> None:
    if nx is None:
        raise RuntimeError("networkx is required. Install with: pip install networkx")


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if not np.isfinite(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _west_to_east_key(lat: float, lon: float) -> Tuple[float, float]:
    # Primary sort by lon (west->east), tie-break by lat.
    return (float(lon), float(lat))


def _dedupe_monotonic_x(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    np.interp expects x increasing. After monotonic enforcement, x is typically
    nondecreasing (may include repeats). Deduplicate repeats by taking the LAST
    occurrence for each x (stable + consistent with "flattening" behavior).
    """
    if x.size <= 1:
        return x, y, z

    # Sort by x first (defensive). This should already be sorted in caller.
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    z = z[order]

    # Keep last occurrence per unique x:
    # reverse -> unique returns sorted unique values, and first occurrence in reversed
    # corresponds to last occurrence in original.
    u, idx_rev = np.unique(x[::-1], return_index=True)
    idx_last = (x.size - 1) - idx_rev

    # u is sorted ascending; idx_last corresponds to each u
    x_u = u.astype("float64", copy=False)
    y_u = y[idx_last].astype("float64", copy=False)
    z_u = z[idx_last].astype("float64", copy=False)
    return x_u, y_u, z_u


def build_chronostrat_diagram(
    G: "nx.Graph",
    shifts: Dict[str, np.ndarray],
    *,
    cfg: ChronostratConfig,
    z_key: str = "depth_rs",
    log_key: str = "log_rs",
) -> Dict[str, object]:
    """
    Returns dict with:
      - node_order: list[str] (west->east)
      - rgt_grid: (T,) float64
      - diag: (W,T) float64 type-log-space values (NaN outside coverage)
      - type_log: (T,) float64 (nanmean across wells)
      - depth_grid: (W,T) float64 depth at each RGT sample for each well (NaN outside)
      - reversal_frac: list[float] per well (how much got flattened by monotonic fix)
      - diag_meta: dict (light diagnostics; safe to ignore)
    """
    _require_nx()

    nodes = list(G.nodes)
    if not nodes:
        raise ValueError("Graph has no nodes.")

    # Sort wells west->east (ChronoLog convention for diagram ordering)
    nodes_sorted = sorted(
        nodes,
        key=lambda n: _west_to_east_key(
            _to_float(G.nodes[n].get("lat", 0.0), default=0.0),
            _to_float(G.nodes[n].get("lon", 0.0), default=0.0),
        ),
    )

    rgt_list: List[np.ndarray] = []
    log_list: List[np.ndarray] = []
    depth_list: List[np.ndarray] = []
    reversal_frac: List[float] = []
    used_nodes: List[str] = []
    skipped: List[Dict[str, str]] = []

    for n in nodes_sorted:
        # Node payloads must exist
        if z_key not in G.nodes[n] or log_key not in G.nodes[n]:
            skipped.append({"node": str(n), "reason": "missing_node_arrays"})
            continue
        if n not in shifts:
            skipped.append({"node": str(n), "reason": "missing_shift"})
            continue

        z = np.asarray(G.nodes[n][z_key], dtype="float64")
        x = np.asarray(G.nodes[n][log_key], dtype="float64")
        s = np.asarray(shifts[n], dtype="float64")

        if z.size == 0 or x.size == 0 or s.size == 0 or z.size != s.size or z.size != x.size:
            skipped.append({"node": str(n), "reason": "inconsistent_sizes"})
            continue

        # Require some finite support
        fin0 = np.isfinite(z) & np.isfinite(x) & np.isfinite(s)
        if int(fin0.sum()) < 5:
            skipped.append({"node": str(n), "reason": "too_few_finite_samples"})
            continue

        rgt = z + s

        rgt_fix, rev = enforce_monotonic_rgt(rgt, cfg=cfg.monotonic)

        # rev is assumed array-like; compute reversal fraction only on finite original rgt
        mfin = np.isfinite(rgt)
        if mfin.any() and np.asarray(rev).shape == rgt.shape:
            reversal_frac.append(float(np.mean(np.asarray(rev, dtype=bool)[mfin])))
        else:
            reversal_frac.append(0.0)

        rgt_list.append(np.asarray(rgt_fix, dtype="float64"))
        log_list.append(x)
        depth_list.append(z)
        used_nodes.append(str(n))

    if not used_nodes:
        raise ValueError("No usable nodes for chronostrat (all nodes missing/invalid arrays or shifts).")

    # Global RGT grid using only finite per-well support
    mins: List[float] = []
    maxs: List[float] = []
    for r in rgt_list:
        fin = np.isfinite(r)
        if not fin.any():
            continue
        mins.append(float(np.nanmin(r[fin])))
        maxs.append(float(np.nanmax(r[fin])))

    if not mins or not maxs:
        raise RuntimeError("Invalid global RGT range computed (no finite RGT samples).")

    rgt_min = float(min(mins))
    rgt_max = float(max(maxs))
    if not np.isfinite(rgt_min) or not np.isfinite(rgt_max) or rgt_max <= rgt_min:
        raise RuntimeError("Invalid global RGT range computed.")

    span = rgt_max - rgt_min
    pad = float(cfg.rgt_pad_frac) * span
    rgt0 = rgt_min - pad
    rgt1 = rgt_max + pad

    T = max(50, int(cfg.n_rgt))
    rgt_grid = np.linspace(rgt0, rgt1, T).astype("float64")

    # Interpolate each well's log into RGT space (chronostrat diagram)
    W = len(used_nodes)
    diag = np.full((W, T), np.nan, dtype="float64")
    depth_grid = np.full((W, T), np.nan, dtype="float64")

    for wi, (rgt, x, z) in enumerate(zip(rgt_list, log_list, depth_list)):
        fin = np.isfinite(rgt) & np.isfinite(x) & np.isfinite(z)
        if int(fin.sum()) < 5:
            continue

        r = rgt[fin]
        xv = x[fin]
        zv = z[fin]

        # rgt after monotonic enforcement should be nondecreasing; still defensively sort
        order = np.argsort(r)
        r = r[order]
        xv = xv[order]
        zv = zv[order]

        # Deduplicate repeated r values (np.interp likes strictly increasing)
        r_u, xv_u, zv_u = _dedupe_monotonic_x(r, xv, zv)
        if r_u.size < 2:
            continue

        r_lo, r_hi = float(r_u[0]), float(r_u[-1])
        m = (rgt_grid >= r_lo) & (rgt_grid <= r_hi)
        if not np.any(m):
            continue

        diag[wi, m] = np.interp(rgt_grid[m], r_u, xv_u)
        # Inversion: depth as function of RGT
        depth_grid[wi, m] = np.interp(rgt_grid[m], r_u, zv_u)

    type_log = np.nanmean(diag, axis=0)

    return {
        "node_order": used_nodes,   # only nodes that contributed
        "rgt_grid": rgt_grid,
        "diag": diag,
        "type_log": type_log,
        "depth_grid": depth_grid,
        "reversal_frac": reversal_frac,
        "diag_meta": {
            "n_graph_nodes": int(len(nodes)),
            "n_used_nodes": int(len(used_nodes)),
            "n_skipped_nodes": int(len(skipped)),
            "skipped": skipped[:200],  # keep bounded; expand if you want
            "z_key": str(z_key),
            "log_key": str(log_key),
            "n_rgt": int(T),
            "rgt_range": [float(rgt0), float(rgt1)],
        },
    }
