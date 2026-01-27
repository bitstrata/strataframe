# src/strataframe/preprocess/well.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from strataframe.io.las import read_las_safely
from strataframe.curves.normalize_header import aliases_for, norm_mnemonic


@dataclass(frozen=True)
class WellLoadConfig:
    # Curve family name (preferred) or a mnemonic-like token. We resolve via aliases_for().
    curve: str = "GR"
    pct_lo: float = 1.0
    pct_hi: float = 99.0
    # Resample count used for DTW/RGT acceleration
    resample_n: int = 400
    # If True, fill NaNs by 1D interpolation (internal); edges clamped to nearest
    fill_nans: bool = True


def _as_float_1d(x: object) -> np.ndarray:
    arr = np.asarray(x, dtype="float64")
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def _list_curve_mnemonics(las) -> List[str]:
    """
    Return curve mnemonics in LAS ~CURVE order (actual mnemonics as stored in file),
    e.g., ["DEPT", "GR", "GR:1", ...].
    """
    out: List[str] = []
    for c in getattr(las, "curves", []) or []:
        try:
            mn = str(getattr(c, "mnemonic", "") or "").strip()
        except Exception:
            mn = ""
        if mn:
            out.append(mn)
    return out


def _finite_count(a: np.ndarray) -> int:
    try:
        x = np.asarray(a, dtype="float64")
        return int(np.isfinite(x).sum())
    except Exception:
        return 0


def _pick_best_curve_by_canon(
    las,
    *,
    wanted_canons: Sequence[str],
) -> Tuple[str, np.ndarray]:
    """
    Pick the *actual* LAS mnemonic and values for any curve whose canonical mnemonic
    (norm_mnemonic) matches one of wanted_canons. If multiple match, pick the one
    with the most finite samples.

    Returns (mnemonic_used, values).
    """
    # Canonical targets (suffix-stripped + alias-collapsed)
    wanted: List[str] = []
    wanted_set = set()
    for w in wanted_canons:
        cw = norm_mnemonic(w)
        if cw and cw not in wanted_set:
            wanted.append(cw)
            wanted_set.add(cw)

    if not wanted_set:
        raise RuntimeError("No valid wanted_canons provided.")

    # Candidate actual mnemonics in file order
    actual_mnems = _list_curve_mnemonics(las)
    cands = [mn for mn in actual_mnems if norm_mnemonic(mn) in wanted_set]

    # Last-resort fallback: raw-prefix match (keeps suffix variants like GR:1)
    if not cands:
        raw_wanted = {str(w).strip().upper() for w in wanted_canons if str(w).strip()}
        for mn in actual_mnems:
            up = mn.strip().upper()
            if any(up.startswith(w) for w in raw_wanted):
                cands.append(mn)

        # de-dupe while preserving order
        seen = set()
        cands = [m for m in cands if not (m in seen or seen.add(m))]

    best_mn: Optional[str] = None
    best_arr: Optional[np.ndarray] = None
    best_n = -1

    for mn in cands:
        try:
            arr = np.asarray(las[mn], dtype="float64")  # type: ignore[index]
        except Exception:
            continue
        if arr.size == 0:
            continue
        nfin = _finite_count(arr)
        if nfin > best_n:
            best_mn, best_arr, best_n = mn, arr, nfin

    if best_mn is None or best_arr is None:
        wanted_s = ",".join(wanted)
        raise RuntimeError(f"Could not load any curve matching canonical set: {wanted_s}")

    return best_mn, best_arr


def _extract_depth(las) -> np.ndarray:
    """
    Prefer las.index if numeric; else fallback to DEPT/DEPTH/MD curves (including suffix forms like DEPT:1).
    Returns float64 1D depth array (may include NaN; caller filters/sorts).
    """
    # las.index is the canonical depth axis in most LAS
    try:
        idx = _as_float_1d(getattr(las, "index"))
        if idx.size and np.isfinite(idx).any():
            return idx
    except Exception:
        pass

    # Fallback: explicit depth curves (robust to suffixes)
    mn_used, d = _pick_best_curve_by_canon(las, wanted_canons=("DEPT", "DEPTH", "MD"))
    dd = _as_float_1d(d)
    if dd.size and np.isfinite(dd).any():
        return dd

    raise RuntimeError("Could not determine depth (no numeric las.index and no DEPT/DEPTH/MD curve).")


def _pick_curve(las, family_or_mnemonic: str) -> Tuple[str, np.ndarray]:
    """
    Resolve a curve using alias families, robust to suffixes (e.g., GR:1) and duplicates.
    Returns (actual_mnemonic_used, values).
    """
    fam = norm_mnemonic(family_or_mnemonic)
    # aliases_for("GR") should include "GR" and acceptable variants
    fam_aliases = aliases_for(fam) if fam else [family_or_mnemonic]

    # Convert aliases to canonical tokens for matching against file curves
    wanted_canons: List[str] = []
    seen = set()
    for a in fam_aliases:
        ca = norm_mnemonic(a)
        if ca and ca not in seen:
            wanted_canons.append(ca)
            seen.add(ca)

    # Pick best actual curve among all matching canonical tokens
    mn_used, arr = _pick_best_curve_by_canon(las, wanted_canons=wanted_canons)
    vv = _as_float_1d(arr)
    if vv.size:
        return mn_used, vv

    raise RuntimeError(f"Curve family '{family_or_mnemonic}' not found in LAS (wanted={wanted_canons}).")


def _fill_nans_1d(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Linear-interpolate interior NaNs; clamp ends to nearest valid.
    Returns (filled, imputed_mask).
    """
    x = np.asarray(x, dtype="float64")
    fin = np.isfinite(x)
    if fin.all():
        return x, np.zeros_like(fin, dtype=bool)
    if (~fin).all():
        return x, np.ones_like(fin, dtype=bool)

    idx = np.arange(x.size, dtype="float64")
    good = np.where(fin)[0]
    bad = np.where(~fin)[0]

    filled = x.copy()
    filled[bad] = np.interp(idx[bad], idx[good], x[good])
    return filled, ~fin


def _percentile_norm_to_unit(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """
    ChronoLog-style normalization to [0,1]:
      x_n = clip( (x - p_lo) / (p_hi - p_lo), 0, 1 )
    computed on finite values only.
    """
    x = np.asarray(x, dtype="float64")
    fin = np.isfinite(x)
    if int(fin.sum()) < 5:
        return np.full_like(x, np.nan, dtype="float64")

    p_lo = float(np.percentile(x[fin], float(lo)))
    p_hi = float(np.percentile(x[fin], float(hi)))
    denom = p_hi - p_lo
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        return np.full_like(x, np.nan, dtype="float64")

    out = (x - p_lo) / denom
    out = np.clip(out, 0.0, 1.0)
    out[~fin] = np.nan
    return out


def _resample_to_n(depth: np.ndarray, values: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample values to n samples on a uniform depth grid between min/max finite depths.
    Requires sorted/unique depth for np.interp; we enforce it here.

    Returns:
      depth_rs (n,), values_rs (n,)
    """
    depth = np.asarray(depth, dtype="float64").reshape(-1)
    values = np.asarray(values, dtype="float64").reshape(-1)

    if depth.size == 0 or values.size == 0 or depth.size != values.size:
        return np.array([], dtype="float64"), np.array([], dtype="float64")

    if int(n) <= 2:
        return depth.copy(), values.copy()

    fin = np.isfinite(depth) & np.isfinite(values)
    if int(fin.sum()) < 5:
        # still emit a grid, but values unknown
        zmin = float(np.nanmin(depth)) if np.isfinite(np.nanmin(depth)) else 0.0
        zmax = float(np.nanmax(depth)) if np.isfinite(np.nanmax(depth)) else 1.0
        z = np.linspace(zmin, zmax, int(n), dtype="float64")
        return z, np.full(int(n), np.nan, dtype="float64")

    d = depth[fin]
    v = values[fin]

    # Sort by depth
    order = np.argsort(d)
    d = d[order]
    v = v[order]

    # Ensure unique x for np.interp (take last occurrence deterministically)
    # (np.unique returns first index; to take last, operate on reversed array)
    d_rev = d[::-1]
    v_rev = v[::-1]
    d_u_rev, idx_rev = np.unique(d_rev, return_index=True)
    # idx_rev are indices into reversed arrays; convert to forward order
    d_u = d_u_rev[::-1]
    v_u = v_rev[idx_rev][::-1]

    zmin = float(d_u[0])
    zmax = float(d_u[-1])
    if not np.isfinite(zmin) or not np.isfinite(zmax) or zmax <= zmin:
        return np.array([], dtype="float64"), np.array([], dtype="float64")

    z = np.linspace(zmin, zmax, int(n), dtype="float64")
    v_rs = np.interp(z, d_u, v_u).astype("float64", copy=False)
    return z, v_rs


@dataclass
class Well:
    """
    Core ChronoLog-like Well object (minimal, deterministic).
    """
    well_id: str
    lat: float
    lon: float
    las_path: Path

    curve_used: str  # mnemonic actually loaded (post-alias resolution)

    depth: np.ndarray
    log_raw: np.ndarray
    log_norm: np.ndarray

    depth_rs: np.ndarray
    log_rs: np.ndarray

    imputed_mask: np.ndarray

    @classmethod
    def from_las(
        cls,
        *,
        well_id: str,
        lat: float,
        lon: float,
        las_path: Path,
        cfg: WellLoadConfig,
    ) -> "Well":
        las_path = Path(las_path)
        rr = read_las_safely(las_path, prefer_fast_for_unwrapped=True, quiet=True)
        las = rr.las

        depth = _extract_depth(las)
        curve_used, log_raw = _pick_curve(las, cfg.curve)

        # Align lengths defensively (LAS can be odd)
        n = int(min(depth.size, log_raw.size))
        depth = depth[:n]
        log_raw = log_raw[:n]

        # Sort by depth once, so all downstream operations are stable
        order = np.argsort(depth)
        depth = depth[order]
        log_raw = log_raw[order]

        # Optional fill NaNs before percentile normalization
        imputed = np.zeros(depth.shape, dtype=bool)
        if bool(cfg.fill_nans):
            log_use, imputed = _fill_nans_1d(log_raw)
        else:
            log_use = log_raw

        log_norm = _percentile_norm_to_unit(log_use, cfg.pct_lo, cfg.pct_hi)

        # Resample to fixed length for DTW/RGT speed
        depth_rs, log_rs = _resample_to_n(depth, log_norm, int(cfg.resample_n))

        return cls(
            well_id=str(well_id),
            lat=float(lat),
            lon=float(lon),
            las_path=las_path,
            curve_used=str(curve_used),
            depth=depth,
            log_raw=log_raw,
            log_norm=log_norm,
            depth_rs=depth_rs,
            log_rs=log_rs,
            imputed_mask=imputed,
        )
