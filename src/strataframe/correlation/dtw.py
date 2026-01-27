# src/strataframe/correlation/dtw.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional

import numpy as np

try:
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover
    nx = None  # type: ignore

try:
    import librosa  # type: ignore
except Exception:  # pragma: no cover
    librosa = None  # type: ignore


@dataclass(frozen=True)
class DtwConfig:
    alpha: float = 0.15
    # Optional global constraint band (Sakoe-Chiba). None => unconstrained.
    band_rad: int | None = None
    # If too many NaNs, you can decide to fail rather than impute.
    min_finite: int = 20


def _require_librosa() -> None:
    if librosa is None:
        raise RuntimeError("librosa is required. Install with: pip install librosa")


def _require_nx() -> None:
    if nx is None:
        raise RuntimeError("networkx is required. Install with: pip install networkx")


def _fill_nans_1d(x: np.ndarray, *, min_finite: int) -> np.ndarray:
    """
    Fill NaNs in a 1D series with linear interpolation on index, plus edge fill.
    Raises if there are too few finite samples to be meaningful.
    """
    x = np.asarray(x, dtype="float64").reshape(-1)
    fin = np.isfinite(x)
    n_fin = int(fin.sum())
    if n_fin < int(min_finite):
        raise ValueError(f"Too few finite samples for DTW: {n_fin} < {int(min_finite)}")

    if n_fin == x.size:
        return x

    idx = np.arange(x.size, dtype="float64")
    xf = x.copy()

    # Interpolate only over finite indices
    xf[~fin] = np.interp(idx[~fin], idx[fin], x[fin])

    # Defensive: if interp produced non-finite (shouldn't), edge-fill
    fin2 = np.isfinite(xf)
    if not fin2.all():
        first = np.argmax(fin2)
        last = x.size - 1 - np.argmax(fin2[::-1])
        xf[:first] = xf[first]
        xf[last + 1 :] = xf[last]
        bad = ~np.isfinite(xf)
        if bad.any():
            xf[bad] = 0.0

    return xf


def _cost_matrix(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    """
    Custom ChronoLog-style local distance:
      d = |l2 - l1|^alpha
    """
    if alpha <= 0.0:
        raise ValueError("alpha must be > 0")
    a = np.asarray(a, dtype="float64").reshape(-1)
    b = np.asarray(b, dtype="float64").reshape(-1)

    # Outer absolute difference
    D = np.abs(a[:, None] - b[None, :])
    # Robustify outliers
    return np.power(D, float(alpha))


def _downsample_path(path: np.ndarray, n_tiepoints: int) -> np.ndarray:
    """
    Downsample a warping path to reduce storage.
    Keeps endpoints.
    """
    path = np.asarray(path, dtype="int64")
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError("dtw path must be (K,2)")
    if n_tiepoints <= 0 or path.shape[0] <= n_tiepoints:
        return path
    idx = np.linspace(0, path.shape[0] - 1, int(n_tiepoints), dtype="int64")
    return path[idx]


def dtw_path_and_cost(
    a: np.ndarray,
    b: np.ndarray,
    *,
    cfg: DtwConfig,
    downsample_path_to: int | None = None,
) -> Tuple[float, np.ndarray]:
    """
    Returns:
      total_cost (float), path (Kx2 array of indices)

    Notes:
    - librosa DTW is endpoint-constrained by construction: path starts at (0,0) and ends at (N-1,M-1).
    - We impute NaNs before DTW; if you want strict behavior, raise earlier upstream.
    """
    _require_librosa()

    aa = _fill_nans_1d(a, min_finite=int(cfg.min_finite))
    bb = _fill_nans_1d(b, min_finite=int(cfg.min_finite))

    if aa.size < 8 or bb.size < 8:
        raise ValueError("Inputs too short for DTW (need >= 8 samples each).")

    C = _cost_matrix(aa, bb, float(cfg.alpha))

    kwargs: Dict[str, Any] = {}
    if cfg.band_rad is not None:
        br = int(cfg.band_rad)
        if br < 0:
            raise ValueError("band_rad must be >= 0")
        # librosa expects band_rad in frames; clamp to something sensible
        br = min(br, max(int(aa.size), int(bb.size)))
        kwargs["global_constraints"] = True
        kwargs["band_rad"] = br

    # Returns (Dcum, wp). wp is warping path in reverse order.
    Dcum, wp = librosa.sequence.dtw(C=C, **kwargs)  # type: ignore[attr-defined]
    wp = np.asarray(wp, dtype="int64")[::-1]  # forward order
    total_cost = float(Dcum[-1, -1])

    if downsample_path_to is not None:
        wp = _downsample_path(wp, int(downsample_path_to))

    return total_cost, wp


def correlate_graph_edges_dtw(
    G: "nx.Graph",
    *,
    cfg: DtwConfig,
    a_key: str = "log_rs",
    z_key: str = "depth_rs",
    store_path: bool = True,
    downsample_path_to: int | None = None,
) -> None:
    """
    In-place: for each edge (u,v) in G, computes DTW between node arrays and stores:
      edge["dtw_cost"], edge["dtw_cost_per_step"], edge["dtw_path"] (optional), edge["dtw_steps"]

    Also writes:
      edge["dtw_status"] in {"OK","SKIP","FAIL"}
      edge["dtw_error"] (string, only on FAIL)

    Canonical note:
      This function is intentionally mnemonic-agnostic. Ensure upstream node building
      writes the intended canonical curve into `a_key` (e.g., GR->log_rs).
    """
    _require_nx()
    _require_librosa()

    for u, v, ed in G.edges(data=True):
        try:
            if a_key not in G.nodes[u] or a_key not in G.nodes[v]:
                ed["dtw_cost"] = np.nan
                ed["dtw_cost_per_step"] = np.nan
                ed["dtw_steps"] = 0
                ed["dtw_path"] = None if store_path else None
                ed["dtw_status"] = "SKIP"
                ed["dtw_error"] = f"missing_node_key:{a_key}"
                continue

            a = np.asarray(G.nodes[u][a_key], dtype="float64")
            b = np.asarray(G.nodes[v][a_key], dtype="float64")

            if a.size == 0 or b.size == 0:
                ed["dtw_cost"] = np.nan
                ed["dtw_cost_per_step"] = np.nan
                ed["dtw_steps"] = 0
                ed["dtw_path"] = None if store_path else None
                ed["dtw_status"] = "SKIP"
                ed["dtw_error"] = "empty_series"
                continue

            cost, path = dtw_path_and_cost(a, b, cfg=cfg, downsample_path_to=downsample_path_to)

            ed["dtw_cost"] = float(cost)
            ed["dtw_steps"] = int(path.shape[0]) if path is not None else 0
            ed["dtw_cost_per_step"] = float(cost / max(1, ed["dtw_steps"]))

            if store_path:
                ed["dtw_path"] = path  # numpy array
            else:
                ed["dtw_path"] = None

            ed["dtw_status"] = "OK"
            ed.pop("dtw_error", None)

        except Exception as e:
            ed["dtw_cost"] = np.nan
            ed["dtw_cost_per_step"] = np.nan
            ed["dtw_steps"] = 0
            ed["dtw_path"] = None
            ed["dtw_status"] = "FAIL"
            ed["dtw_error"] = f"{type(e).__name__}: {e}"
