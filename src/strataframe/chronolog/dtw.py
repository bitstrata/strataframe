from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def pad_for_distance(
    *,
    base_pad_ft: float,
    pad_slope_ft_per_km: float,
    max_pad_ft: float,
    dist_km: float,
) -> float:
    pad = float(base_pad_ft) + float(pad_slope_ft_per_km) * float(dist_km)
    if float(max_pad_ft) > 0:
        pad = min(float(max_pad_ft), pad)
    return max(0.0, float(pad))


def compute_overlap_window(
    zmin1: float, zmax1: float, zmin2: float, zmax2: float, pad_ft: float
) -> Tuple[float, float, float, float]:
    overlap_min = max(zmin1, zmin2)
    overlap_max = min(zmax1, zmax2)
    overlap_ft = max(0.0, overlap_max - overlap_min)

    win_min = overlap_min - float(pad_ft)
    win_max = overlap_max + float(pad_ft)

    len1 = max(0.0, zmax1 - zmin1)
    len2 = max(0.0, zmax2 - zmin2)
    denom = min(len1, len2) if min(len1, len2) > 0 else 0.0
    overlap_frac = float(overlap_ft / denom) if denom > 0 else 0.0
    return win_min, win_max, overlap_ft, overlap_frac


def load_gr_vectors_cache(path: Path) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=False)
    required = {"node_id", "z_top", "z_base", "x_norm"}
    if not required.issubset(set(data.files)):
        raise ValueError(f"gr_vectors cache missing required arrays: {required}")
    node_id = data["node_id"].astype("int64", copy=False)
    z_top = data["z_top"].astype("float64", copy=False)
    z_base = data["z_base"].astype("float64", copy=False)
    if "z_gr_top" in data.files:
        z_gr_top = data["z_gr_top"].astype("float64", copy=False)
    else:
        z_gr_top = z_top
    if "z_gr_base" in data.files:
        z_gr_base = data["z_gr_base"].astype("float64", copy=False)
    else:
        z_gr_base = z_base
    imputed_mask = None
    if "imputed_mask" in data.files:
        imputed_mask = data["imputed_mask"].astype(bool, copy=False)
    x_norm = data["x_norm"]
    if x_norm.ndim != 2:
        raise ValueError("x_norm must be 2D (n_wells, n_samples)")
    index = {int(n): i for i, n in enumerate(node_id.tolist())}
    meta = {}
    if "meta_json" in data.files:
        try:
            meta = json.loads(str(data["meta_json"].item()))
        except Exception:
            meta = {}
    return {
        "node_id": node_id,
        "z_top": z_top,
        "z_base": z_base,
        "z_gr_top": z_gr_top,
        "z_gr_base": z_gr_base,
        "x_norm": x_norm,
        "imputed_mask": imputed_mask,
        "index": index,
        "meta": meta,
    }


def resample_from_cache(
    x_full: np.ndarray,
    z_top: float,
    z_base: float,
    win_min: float,
    win_max: float,
    *,
    n_samples: int,
    p_lo: float,
    p_hi: float,
    valid_min: float | None = None,
    valid_max: float | None = None,
    mask_outside: bool = False,
) -> np.ndarray:
    n_full = int(x_full.shape[0])
    if n_full < 2:
        raise RuntimeError("Cached vector too short")
    z_full = np.linspace(float(z_top), float(z_base), n_full, dtype="float64")
    z_win = np.linspace(float(win_min), float(win_max), int(n_samples), dtype="float64")
    use_mask = bool(mask_outside) and (valid_min is not None) and (valid_max is not None)
    if use_mask:
        left = np.nan
        right = np.nan
    else:
        left = float(x_full[0])
        right = float(x_full[-1])
    x_win = np.interp(z_win, z_full, x_full, left=left, right=right)
    if use_mask:
        vmin = float(valid_min)  # type: ignore[arg-type]
        vmax = float(valid_max)  # type: ignore[arg-type]
        if np.isfinite(vmin) and np.isfinite(vmax):
            m = (z_win < vmin) | (z_win > vmax)
            if np.any(m):
                x_win = x_win.astype("float64", copy=True)
                x_win[m] = np.nan
    fin = np.isfinite(x_win)
    if not np.any(fin):
        return np.zeros((int(n_samples),), dtype="float64")
    plo = float(np.percentile(x_win[fin], float(p_lo)))
    phi = float(np.percentile(x_win[fin], float(p_hi)))
    if (not np.isfinite(plo)) or (not np.isfinite(phi)) or (phi <= plo):
        plo = float(np.nanmin(x_win[fin]))
        phi = float(np.nanmax(x_win[fin]))
        if (not np.isfinite(plo)) or (not np.isfinite(phi)) or (phi <= plo):
            return np.zeros((int(n_samples),), dtype="float64")
    x_norm = (x_win - plo) / (phi - plo)
    return np.clip(x_norm, 0.0, 1.0).astype("float64", copy=False)
