# src/strataframe/pipelines/step3b_rep_arrays.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

# Loader signature: given rep_id -> (depth, gr)
RepLogLoader = Callable[[str], Tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class RepArraysConfig:
    """
    Step 3b: Extract, resample, normalize GR for each representative well.

    Outputs:
      - depth_rs: (n_samples,) float64
      - gr_rs:    (n_samples,) float64 normalized to [0,1]
      - imputed_mask_rs: (n_samples,) bool  (True where value is "less trustworthy")

    Notes on imputed_mask_rs:
      - True outside the convex hull of finite original GR samples
      - True where the bracketing finite samples are farther apart than max_gap_depth (if set)
      - If fill_nans=False, interpolation still occurs between finite samples; mask still reflects gaps.
    """
    n_samples: int = 400

    # percentile normalization bounds on the (resampled) series
    p_lo: float = 1.0
    p_hi: float = 99.0

    # DTW stability
    fill_nans: bool = True

    # Minimum count of finite original GR samples to attempt a rep
    min_finite: int = 20

    # If set (in depth units): mark imputed_mask_rs=True where the finite bracketing
    # samples are farther apart than this. Useful when logs have long gaps.
    max_gap_depth: Optional[float] = None

    # Optional: clamp depth span to avoid insane ranges (None disables)
    max_depth_span: Optional[float] = None

    # If True, drop non-finite depth and then require strictly increasing after dedupe
    require_increasing_depth: bool = True


@dataclass(frozen=True)
class RepArraysBuildResult:
    """
    In-memory result. Intended to be cached to NPZ via save_rep_arrays_npz().
    """
    rep_ids: List[str]
    depth_rs: np.ndarray          # (R, nS)
    gr_rs: np.ndarray             # (R, nS)
    imputed_mask_rs: np.ndarray   # (R, nS) bool
    meta: Optional["pd.DataFrame"] = None


# ---------------------------------------------------------------------
# Core transforms
# ---------------------------------------------------------------------

def _dedupe_by_depth(z: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ensure z increasing with unique values by taking LAST occurrence for each z.
    """
    order = np.argsort(z, kind="mergesort")
    z = z[order]
    x = x[order]

    if z.size <= 1:
        return z, x

    # keep last occurrence per unique z
    u, idx_rev = np.unique(z[::-1], return_index=True)
    idx_last = (z.size - 1) - idx_rev
    # u is sorted ascending already
    z_u = u.astype("float64", copy=False)
    x_u = x[idx_last].astype("float64", copy=False)
    return z_u, x_u


def _interp_fill_1d(z: np.ndarray, x: np.ndarray, *, min_finite: int) -> np.ndarray:
    """
    Fill NaNs in x via linear interpolation on z, with edge fill.
    Requires >= min_finite finite samples.
    """
    z = np.asarray(z, dtype="float64").reshape(-1)
    x = np.asarray(x, dtype="float64").reshape(-1)

    fin = np.isfinite(z) & np.isfinite(x)
    n_fin = int(fin.sum())
    if n_fin < int(min_finite):
        raise ValueError(f"Too few finite samples: {n_fin} < {int(min_finite)}")

    if n_fin == x.size:
        return x

    xf = x.copy()
    # interpolate at non-finite x positions but finite z positions
    m = np.isfinite(z) & ~np.isfinite(x)
    if np.any(m):
        xf[m] = np.interp(z[m], z[fin], x[fin])

    # edge fill any remaining non-finite (e.g., non-finite z)
    bad = ~np.isfinite(xf)
    if bad.any():
        xf[bad] = 0.0

    return xf


def _imputed_mask_resampled(
    z_fin: np.ndarray,
    z_rs: np.ndarray,
    *,
    max_gap_depth: Optional[float],
) -> np.ndarray:
    """
    Build imputed mask on resampled grid:
      - True outside [min(z_fin), max(z_fin)]
      - True where bracketing finite samples are farther apart than max_gap_depth (if set)
    """
    z_fin = np.asarray(z_fin, dtype="float64").reshape(-1)
    z_rs = np.asarray(z_rs, dtype="float64").reshape(-1)

    if z_fin.size < 2:
        return np.ones_like(z_rs, dtype=bool)

    z0 = float(z_fin[0])
    z1 = float(z_fin[-1])

    out = np.zeros_like(z_rs, dtype=bool)
    out |= (z_rs < z0) | (z_rs > z1)

    if max_gap_depth is None:
        return out

    mg = float(max_gap_depth)
    if not np.isfinite(mg) or mg <= 0:
        return out

    # For each z_rs inside [z0,z1], find bracketing finite samples
    # idx = first index where z_fin[idx] >= z
    idx = np.searchsorted(z_fin, z_rs, side="left")

    # clamp idx into [1, len-1] so we have left/right
    idx = np.clip(idx, 1, z_fin.size - 1)
    z_left = z_fin[idx - 1]
    z_right = z_fin[idx]
    gap = z_right - z_left

    out |= (gap > mg)
    return out


def _percentile_normalize_01(
    x: np.ndarray,
    *,
    p_lo: float,
    p_hi: float,
    mask_for_stats: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, float]:
    """
    Normalize to [0,1] using percentiles on x[mask_for_stats].
    """
    x = np.asarray(x, dtype="float64").reshape(-1)

    if mask_for_stats is None:
        xs = x[np.isfinite(x)]
    else:
        m = np.asarray(mask_for_stats, dtype=bool).reshape(-1)
        xs = x[np.isfinite(x) & m]

    if xs.size < 5:
        # fallback: use any finite values
        xs = x[np.isfinite(x)]

    if xs.size == 0:
        return np.zeros_like(x), float("nan"), float("nan")

    lo = float(np.percentile(xs, float(p_lo)))
    hi = float(np.percentile(xs, float(p_hi)))

    # Avoid divide-by-zero / nonsense
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        y = np.zeros_like(x)
        return y, lo, hi

    y = (x - lo) / (hi - lo)
    y = np.clip(y, 0.0, 1.0)
    return y, lo, hi


def build_rep_arrays_for_one(
    rep_id: str,
    depth: np.ndarray,
    gr: np.ndarray,
    *,
    cfg: RepArraysConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """
    Returns (depth_rs, gr_rs_norm01, imputed_mask_rs, meta_row).
    Raises on hard failures; caller decides whether to SKIP/FAIL.
    """
    rep_id = str(rep_id)
    z = np.asarray(depth, dtype="float64").reshape(-1)
    x = np.asarray(gr, dtype="float64").reshape(-1)

    if z.size == 0 or x.size == 0 or z.size != x.size:
        raise ValueError("depth/gr missing or mismatched sizes")

    # drop non-finite depth first (depth is the independent axis)
    m_z = np.isfinite(z)
    z = z[m_z]
    x = x[m_z]
    if z.size < 2:
        raise ValueError("too_few_depth_samples_after_drop")

    # sort + dedupe depth
    z, x = _dedupe_by_depth(z, x)
    if cfg.require_increasing_depth and z.size >= 2 and not np.all(np.diff(z) > 0):
        # after dedupe, should be strictly increasing
        raise ValueError("non_increasing_depth_after_dedupe")

    # finite GR support for viability and mask construction
    fin = np.isfinite(x)
    n_fin = int(fin.sum())
    if n_fin < int(cfg.min_finite):
        raise ValueError(f"too_few_finite_gr:{n_fin}<{int(cfg.min_finite)}")

    z_fin = z[fin]
    z_min = float(z[0])
    z_max = float(z[-1])
    span = z_max - z_min
    if not np.isfinite(span) or span <= 0:
        raise ValueError("invalid_depth_span")
    if cfg.max_depth_span is not None and span > float(cfg.max_depth_span):
        raise ValueError(f"depth_span_too_large:{span}")

    # resampled depth grid across full available depth span
    nS = max(8, int(cfg.n_samples))
    depth_rs = np.linspace(z_min, z_max, nS).astype("float64")

    # fill NaNs (optional) on original sample axis, then interpolate to depth_rs
    if cfg.fill_nans:
        x_fill = _interp_fill_1d(z, x, min_finite=int(cfg.min_finite))
        # interpolation uses all z (already finite) and x_fill (finite)
        gr_rs = np.interp(depth_rs, z, x_fill).astype("float64")
    else:
        # interpolate using only finite GR samples
        # np.interp will edge-fill outside range of z_fin (mask will flag that)
        gr_rs = np.interp(depth_rs, z_fin, x[fin]).astype("float64")

    # imputation mask on resampled axis
    imputed_mask_rs = _imputed_mask_resampled(
        z_fin=z_fin,
        z_rs=depth_rs,
        max_gap_depth=cfg.max_gap_depth,
    )

    # normalize to [0,1] using percentiles on "less-imputed" region
    # i.e., use ~imputed_mask_rs for stats by default
    stats_mask = ~imputed_mask_rs
    gr_rs_norm, p_lo_val, p_hi_val = _percentile_normalize_01(
        gr_rs,
        p_lo=cfg.p_lo,
        p_hi=cfg.p_hi,
        mask_for_stats=stats_mask,
    )

    meta = {
        "rep_id": rep_id,
        "status": "OK",
        "n_raw": int(depth.size),
        "n_finite_gr": int(n_fin),
        "z_min": float(z_min),
        "z_max": float(z_max),
        "z_fin_min": float(z_fin[0]) if z_fin.size else float("nan"),
        "z_fin_max": float(z_fin[-1]) if z_fin.size else float("nan"),
        "n_samples": int(nS),
        "p_lo": float(cfg.p_lo),
        "p_hi": float(cfg.p_hi),
        "p_lo_val": float(p_lo_val),
        "p_hi_val": float(p_hi_val),
        "fill_nans": bool(cfg.fill_nans),
        "max_gap_depth": float(cfg.max_gap_depth) if cfg.max_gap_depth is not None else float("nan"),
        "imputed_frac_rs": float(np.mean(imputed_mask_rs)) if imputed_mask_rs.size else float("nan"),
    }
    return depth_rs, gr_rs_norm, imputed_mask_rs, meta


def build_rep_arrays(
    rep_ids: Sequence[str],
    loader: RepLogLoader,
    *,
    cfg: RepArraysConfig = RepArraysConfig(),
    keep_meta: bool = True,
) -> RepArraysBuildResult:
    """
    Build arrays for many reps.

    rep_ids:
      iterable of rep_id strings (stable order preserved)

    loader(rep_id) must return:
      depth: (N,) array, gr: (N,) array

    Returns stacked arrays:
      depth_rs: (R, n_samples), gr_rs: (R, n_samples), imputed_mask_rs: (R, n_samples)

    Reps that fail are excluded from the stacked arrays; see meta for SKIP/FAIL.
    """
    rep_ids = [str(r) for r in rep_ids]

    used_ids: List[str] = []
    depth_list: List[np.ndarray] = []
    gr_list: List[np.ndarray] = []
    imp_list: List[np.ndarray] = []
    meta_rows: List[Dict[str, object]] = []

    for rid in rep_ids:
        try:
            z, x = loader(rid)
            depth_rs, gr_rs, imputed_rs, meta = build_rep_arrays_for_one(rid, z, x, cfg=cfg)
            used_ids.append(rid)
            depth_list.append(depth_rs)
            gr_list.append(gr_rs)
            imp_list.append(imputed_rs)
            meta_rows.append(meta)
        except Exception as e:
            meta_rows.append(
                {
                    "rep_id": rid,
                    "status": "SKIP",
                    "error": f"{type(e).__name__}: {e}",
                    "n_samples": int(cfg.n_samples),
                    "fill_nans": bool(cfg.fill_nans),
                }
            )

    if not used_ids:
        # still return well-typed empty arrays
        nS = max(8, int(cfg.n_samples))
        depth_rs = np.zeros((0, nS), dtype="float64")
        gr_rs = np.zeros((0, nS), dtype="float64")
        imputed = np.zeros((0, nS), dtype=bool)
    else:
        depth_rs = np.stack(depth_list, axis=0).astype("float64", copy=False)
        gr_rs = np.stack(gr_list, axis=0).astype("float64", copy=False)
        imputed = np.stack(imp_list, axis=0).astype(bool, copy=False)

    meta_df: Optional["pd.DataFrame"] = None
    if keep_meta and pd is not None:
        meta_df = pd.DataFrame(meta_rows)

    return RepArraysBuildResult(
        rep_ids=used_ids,
        depth_rs=depth_rs,
        gr_rs=gr_rs,
        imputed_mask_rs=imputed,
        meta=meta_df,
    )


# ---------------------------------------------------------------------
# NPZ cache IO
# ---------------------------------------------------------------------

def save_rep_arrays_npz(
    res: RepArraysBuildResult,
    out_path: str | Path,
    *,
    include_imputed_mask: bool = True,
    compress: bool = True,
) -> None:
    """
    Writes a single NPZ with "keyed" access via rep_ids + stacked arrays.

    NPZ keys written:
      rep_ids      (R,) unicode
      depth_rs     (R,nS) float64
      gr_rs        (R,nS) float64
      imputed_mask_rs (R,nS) bool   (optional)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rep_ids_arr = np.asarray([str(r) for r in res.rep_ids], dtype=str)
    payload = {
        "rep_ids": rep_ids_arr,
        "depth_rs": np.asarray(res.depth_rs, dtype="float64"),
        "gr_rs": np.asarray(res.gr_rs, dtype="float64"),
    }
    if include_imputed_mask:
        payload["imputed_mask_rs"] = np.asarray(res.imputed_mask_rs, dtype=bool)

    if compress:
        np.savez_compressed(out_path, **payload)
    else:
        np.savez(out_path, **payload)


def load_rep_arrays_npz(
    path: str | Path,
    *,
    require_imputed_mask: bool = False,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Loads NPZ produced by save_rep_arrays_npz() into a dict:
      rep_arrays[rep_id] = {"depth_rs": (nS,), "log_rs": (nS,), "imputed_mask_rs": (nS,)}

    Note:
      - The series key is "log_rs" for downstream DTW compatibility (your dtw.py defaults).
      - Depth key is "depth_rs".
    """
    z = np.load(Path(path), allow_pickle=False)

    rep_ids = np.asarray(z["rep_ids"], dtype=np.str_)
    depth_rs = np.asarray(z["depth_rs"], dtype="float64")
    gr_rs = np.asarray(z["gr_rs"], dtype="float64")

    has_mask = "imputed_mask_rs" in z.files
    if require_imputed_mask and not has_mask:
        raise ValueError("NPZ missing imputed_mask_rs but require_imputed_mask=True")

    imputed = np.asarray(z["imputed_mask_rs"], dtype=bool) if has_mask else None

    if depth_rs.ndim != 2 or gr_rs.ndim != 2 or depth_rs.shape != gr_rs.shape:
        raise ValueError("Invalid NPZ shapes: depth_rs/gr_rs must be (R,nS) and match")
    if rep_ids.size != depth_rs.shape[0]:
        raise ValueError("Invalid NPZ shapes: rep_ids length must match first dim of arrays")

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for i, rid in enumerate(rep_ids.tolist()):
        d = depth_rs[i, :].copy()
        g = gr_rs[i, :].copy()
        rec: Dict[str, np.ndarray] = {"depth_rs": d, "log_rs": g}
        if imputed is not None:
            rec["imputed_mask_rs"] = imputed[i, :].copy()
        out[str(rid)] = rec

    return out


def save_meta_table(res: RepArraysBuildResult, out_path: str | Path) -> None:
    """
    Convenience: persist meta as CSV (if pandas is available and meta exists).
    """
    if res.meta is None:
        return
    if pd is None:
        raise RuntimeError("pandas not available; cannot save meta table")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res.meta.to_csv(out_path, index=False)
