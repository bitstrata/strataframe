# src/strataframe/graph/las_utils.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Optional lasio (preferred for LAS). If absent, we fail with a clear message.
try:
    import lasio  # type: ignore
except Exception:  # pragma: no cover
    lasio = None  # type: ignore

# Optional librosa for DTW
try:
    import librosa  # type: ignore
except Exception:  # pragma: no cover
    librosa = None  # type: ignore


# =============================================================================
# Canonical header normalization (shared owner in strataframe.curves.*)
# =============================================================================
try:
    from strataframe.curves.normalize_header import norm_mnemonic as _norm_mnemonic  # type: ignore
except Exception:  # pragma: no cover

    def _norm_mnemonic(m: Optional[str]) -> str:  # type: ignore
        return (m or "").strip().upper()


def normalize_mnemonic(m: str) -> str:
    """
    Backward-compatible wrapper returning the CANONICAL mnemonic.
    """
    return _norm_mnemonic(m)


# =============================================================================
# Exceptions + requirements
# =============================================================================

class LasReadError(RuntimeError):
    """Raised when a LAS file cannot be read in the requested mode."""


def require_lasio() -> None:
    if lasio is None:
        raise RuntimeError(
            "lasio is required. Install it with:\n"
            "  pip install lasio\n"
        )


def require_librosa() -> None:
    if librosa is None:
        raise RuntimeError(
            "librosa is required for DTW. Install it with:\n"
            "  pip install librosa\n"
        )


# =============================================================================
# LAS read helpers (deduplicated; wrapped-file tolerant)
# =============================================================================

_LASIO_SUPPORTS_ENGINE: Optional[bool] = None


def _lasio_supports_engine_kw() -> bool:
    """
    Detect whether lasio.read supports engine= kwarg (varies by lasio version).
    Cached to avoid repeated inspect overhead.
    """
    global _LASIO_SUPPORTS_ENGINE
    if _LASIO_SUPPORTS_ENGINE is not None:
        return bool(_LASIO_SUPPORTS_ENGINE)

    require_lasio()
    try:
        import inspect

        sig = inspect.signature(lasio.read)  # type: ignore[attr-defined]
        _LASIO_SUPPORTS_ENGINE = bool("engine" in sig.parameters)
    except Exception:
        _LASIO_SUPPORTS_ENGINE = False
    return bool(_LASIO_SUPPORTS_ENGINE)


def _read_lasio(
    las_path: Path,
    *,
    ignore_data: bool,
) -> object:
    """
    Read LAS using lasio with best-effort wrapped-file handling.

    Strategy:
      - Prefer engine='normal' when supported (robust for wrapped LAS).
      - Fall back to no engine kw when not supported.
      - If wrapped LAS fails on older lasio, raise a clear message.
    """
    require_lasio()
    p = Path(las_path)

    if not p.exists():
        raise FileNotFoundError(p)

    # Prefer engine='normal' if available
    if _lasio_supports_engine_kw():
        try:
            return lasio.read(str(p), ignore_data=bool(ignore_data), engine="normal")  # type: ignore[attr-defined]
        except Exception as e:
            # If engine path fails, try without engine once (some edge-case builds)
            try:
                return lasio.read(str(p), ignore_data=bool(ignore_data))  # type: ignore[attr-defined]
            except Exception as e2:
                msg = (str(e2) or str(e)).lower()
                if "wrapped" in msg:
                    raise LasReadError(
                        "LAS appears wrapped but this lasio build cannot read it reliably. "
                        "Upgrade lasio (recommended) or pre-unwind wrapped LAS files."
                    ) from e2
                raise LasReadError(f"Failed to read LAS: {p}") from e2

    # Older lasio: no engine kw
    try:
        return lasio.read(str(p), ignore_data=bool(ignore_data))  # type: ignore[attr-defined]
    except Exception as e:
        msg = str(e).lower()
        if "wrapped" in msg:
            raise LasReadError(
                "LAS appears wrapped but this lasio version cannot force engine='normal'. "
                "Upgrade lasio (recommended) or pre-unwind wrapped LAS files."
            ) from e
        raise LasReadError(f"Failed to read LAS: {p}") from e


def read_las_normal(las_path: Path):
    """
    Read full LAS (data + headers), wrapped-file tolerant.
    """
    return _read_lasio(Path(las_path), ignore_data=False)


def read_las_header_only(las_path: Path):
    """
    Read LAS headers only (fast), wrapped-file tolerant.
    """
    return _read_lasio(Path(las_path), ignore_data=True)


# =============================================================================
# Curve discovery + resolution
# =============================================================================

def list_curve_mnemonics(las) -> List[str]:
    """
    Returns RAW curve mnemonics as they appear in the LAS (trimmed).
    These raw names are what you can safely use to index las[raw].
    """
    out: List[str] = []
    for c in getattr(las, "curves", []) or []:
        try:
            raw = (getattr(c, "mnemonic", "") or "").strip()
            if raw:
                out.append(raw)
        except Exception:
            continue
    return out


def available_canonical_curves(las) -> Dict[str, List[str]]:
    """
    Map canonical_mnemonic -> list of raw mnemonics present in this LAS.
    """
    m: Dict[str, List[str]] = {}
    for raw in list_curve_mnemonics(las):
        canon = _norm_mnemonic(raw)
        if canon:
            m.setdefault(canon, []).append(raw)
    return m


def resolve_curve_mnemonic(las, canonical_mnemonic: str) -> Optional[str]:
    """
    Resolve a CANONICAL mnemonic (e.g., 'GR') to a RAW mnemonic present
    in the LAS suitable for las[raw] indexing.

    Never assume canonical == raw.
    """
    canon = _norm_mnemonic(canonical_mnemonic)
    if not canon:
        return None

    raw_list = available_canonical_curves(las).get(canon, [])
    if not raw_list:
        return None

    def _pref_key(raw: str) -> Tuple[int, int, int]:
        r = (raw or "").strip()
        r_up = r.upper()
        exact = 0 if r_up == canon else 1
        no_suffix = 0 if ":" not in r_up else 1  # prefer GR over GR:1, etc.
        return (exact, no_suffix, len(r))

    return sorted(raw_list, key=_pref_key)[0]


def find_curve_mnemonic(mnemonics: Sequence[str], preferred: Sequence[str]) -> Optional[str]:
    """
    Choose the first match in preferred order, where preferred is intended to be CANONICAL.

    Returns a RAW mnemonic from `mnemonics` suitable for las[raw] indexing.

    Backward tolerant: if preferred includes raw/alias names, they are canonicalized anyway.
    """
    if not mnemonics:
        return None

    canon_map: Dict[str, List[str]] = {}
    for raw in mnemonics:
        c = _norm_mnemonic(raw)
        if c:
            canon_map.setdefault(c, []).append(raw)

    def _best_raw(canon: str) -> Optional[str]:
        raws = canon_map.get(canon, [])
        if not raws:
            return None

        def _pref_key(raw: str) -> Tuple[int, int, int]:
            r = (raw or "").strip()
            r_up = r.upper()
            exact = 0 if r_up == canon else 1
            no_suffix = 0 if ":" not in r_up else 1
            return (exact, no_suffix, len(r))

        return sorted(raws, key=_pref_key)[0]

    for p in preferred:
        canon = _norm_mnemonic(p)
        raw = _best_raw(canon)
        if raw is not None:
            return raw
    return None


# =============================================================================
# Numeric helpers
# =============================================================================

def _as_float_array(x) -> np.ndarray:
    """
    Convert arbitrary array-like to float64, coercing non-numeric entries to NaN.
    Avoids repeated element-wise Python loops for already-numeric arrays.
    """
    arr = np.asarray(x)
    if arr.dtype.kind in {"f", "i", "u"}:
        return arr.astype("float64", copy=False)

    # Fast-ish coercion attempt first
    try:
        return arr.astype("float64")
    except Exception:
        out = np.full(arr.shape, np.nan, dtype="float64")
        it = np.nditer(arr, flags=["multi_index", "refs_ok"])
        for v in it:
            try:
                out[it.multi_index] = float(v.item())
            except Exception:
                out[it.multi_index] = np.nan
        return out


def extract_depth_and_curve(
    las,
    *,
    curve_mnemonic: str,
    depth_preferred: Sequence[str] = ("DEPT", "MD"),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (depth, curve_values) as float arrays.

    - `curve_mnemonic` is treated as CANONICAL; we resolve to RAW before indexing.
    - Depth selection:
        1) Use numeric las.index if it has at least 2 finite values and a non-zero range.
        2) Otherwise resolve a depth curve from canonical preferences.

    Output is not guaranteed monotonic; downstream resampling should sort by depth.
    """
    target_canon = _norm_mnemonic(curve_mnemonic)
    if not target_canon:
        raise ValueError("curve_mnemonic is empty")

    # 1) Depth from las.index if viable
    depth: Optional[np.ndarray] = None
    try:
        idx = np.asarray(getattr(las, "index"))
        d = _as_float_array(idx)
        fin = np.isfinite(d)
        if d.ndim == 1 and int(np.count_nonzero(fin)) >= 2:
            dmin = float(np.nanmin(d))
            dmax = float(np.nanmax(d))
            if np.isfinite(dmin) and np.isfinite(dmax) and (dmax > dmin):
                depth = d
    except Exception:
        depth = None

    # 2) Fallback: depth curve
    if depth is None:
        raw_mn = list_curve_mnemonics(las)
        dep_raw = find_curve_mnemonic(raw_mn, depth_preferred)
        if dep_raw is None:
            avail = sorted({c for c in (_norm_mnemonic(m) for m in raw_mn) if c})
            raise RuntimeError(
                "Could not determine depth (no usable las.index and no depth curve found). "
                f"Depth preferred={list(depth_preferred)}. Available canonical={avail[:50]}"
            )
        depth = _as_float_array(las[dep_raw])  # type: ignore[index]

    # Resolve target curve raw mnemonic (critical: do not assume canonical exists as raw)
    curve_raw = resolve_curve_mnemonic(las, target_canon)
    if curve_raw is None:
        raw_mn = list_curve_mnemonics(las)
        avail = sorted({c for c in (_norm_mnemonic(m) for m in raw_mn) if c})
        raise KeyError(
            f"Canonical curve {target_canon} not found in LAS. "
            f"Available canonical={avail[:50]}"
        )

    y = _as_float_array(las[curve_raw])  # type: ignore[index]

    # Align lengths defensively
    n = int(min(depth.size, y.size))
    if n <= 0:
        raise RuntimeError("Depth/curve arrays are empty after extraction.")
    depth = depth[:n]
    y = y[:n]
    return depth.astype("float64", copy=False), y.astype("float64", copy=False)


# =============================================================================
# Resampling + DTW
# =============================================================================

def resample_and_normalize_curve(
    depth: np.ndarray,
    y: np.ndarray,
    *,
    n_samples: int,
    p_lo: float = 1.0,
    p_hi: float = 99.0,
    min_finite: int = 10,
) -> Tuple[np.ndarray, float, float]:
    """
    Resample curve to a uniform depth grid and normalize to [0,1] using robust percentiles.

    Returns:
      x_norm (n_samples,), z_top, z_base
    """
    depth = np.asarray(depth, dtype="float64").reshape(-1)
    y = np.asarray(y, dtype="float64").reshape(-1)

    n_samples_i = int(n_samples)
    if n_samples_i < 8:
        raise ValueError("n_samples too small; use >= 8")
    if not (0.0 <= float(p_lo) < float(p_hi) <= 100.0):
        raise ValueError("Percentiles must satisfy 0 <= p_lo < p_hi <= 100")

    if depth.size < 2:
        raise RuntimeError("Insufficient depth samples.")

    # sort by depth; keep y aligned
    order = np.argsort(depth)
    depth = depth[order]
    y = y[order]

    # finite depth only (y may still be non-finite; handled below)
    m = np.isfinite(depth)
    depth = depth[m]
    y = y[m]
    if depth.size < 2:
        raise RuntimeError("Insufficient finite depth after filtering.")

    z_top = float(np.nanmin(depth))
    z_base = float(np.nanmax(depth))
    if not np.isfinite(z_top) or not np.isfinite(z_base) or z_base <= z_top:
        raise RuntimeError("Invalid depth range.")

    z = np.linspace(z_top, z_base, n_samples_i, dtype="float64")

    fin = np.isfinite(y)
    if int(np.count_nonzero(fin)) < int(min_finite):
        raise RuntimeError("Too few finite curve samples to resample.")

    d_fin = depth[fin]
    y_fin = y[fin]

    # Aggregate duplicates by mean value (more stable than 'first wins')
    uniq_d, inv = np.unique(d_fin, return_inverse=True)
    if uniq_d.size < 2:
        raise RuntimeError("Degenerate depth after removing duplicates.")
    y_sum = np.bincount(inv, weights=y_fin.astype("float64", copy=False))
    y_cnt = np.bincount(inv).astype("float64", copy=False)
    y_mean = y_sum / np.maximum(1.0, y_cnt)

    y_rs = np.interp(z, uniq_d, y_mean, left=float(y_mean[0]), right=float(y_mean[-1])).astype("float64", copy=False)

    plo = float(np.percentile(y_rs, float(p_lo)))
    phi = float(np.percentile(y_rs, float(p_hi)))

    # If percentiles collapse, fall back to min/max normalization (still robust)
    if (not np.isfinite(plo)) or (not np.isfinite(phi)) or (phi <= plo):
        plo = float(np.nanmin(y_rs))
        phi = float(np.nanmax(y_rs))
        if (not np.isfinite(plo)) or (not np.isfinite(phi)) or (phi <= plo):
            # fully degenerate: return zeros but preserve top/base
            return np.zeros((n_samples_i,), dtype="float64"), z_top, z_base

    x = (y_rs - plo) / (phi - plo)
    x = np.clip(x, 0.0, 1.0)
    return x.astype("float64", copy=False), z_top, z_base


def dtw_cost_and_path(
    x: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = 0.15,
    backtrack: bool = True,
) -> Tuple[float, float, Optional[np.ndarray]]:
    """
    DTW with local distance d = |x_i - y_j|^alpha (alpha < 1 de-emphasizes outliers)

    Returns:
      cost_total, cost_per_step, path (optional; shape [L,2] with (i,j))
    """
    require_librosa()

    x = np.asarray(x, dtype="float64").reshape(-1)
    y = np.asarray(y, dtype="float64").reshape(-1)

    if x.size < 8 or y.size < 8:
        raise ValueError("x and y too short for DTW (use >= 8 samples each).")
    if float(alpha) <= 0.0:
        raise ValueError("alpha must be > 0")

    C = np.abs(x[:, None] - y[None, :]) ** float(alpha)

    # librosa API is stable here, but keep return handling defensive
    res = librosa.sequence.dtw(C=C, backtrack=bool(backtrack))  # type: ignore[attr-defined]
    if isinstance(res, tuple) and len(res) == 2:
        D, wp = res
    else:
        raise RuntimeError("Unexpected librosa.sequence.dtw return value.")

    D = np.asarray(D, dtype="float64")
    cost_total = float(D[-1, -1]) if D.size else float("nan")

    if wp is None:
        # Conservative denominator if path not available
        denom = float(max(1, x.size + y.size))
        return float(cost_total), float(cost_total / denom), None

    path = np.asarray(wp[::-1], dtype="int64")
    L = int(path.shape[0]) if path.ndim == 2 else 0
    cost_per_step = float(cost_total) / float(max(1, L))
    return float(cost_total), float(cost_per_step), path


def downsample_path(path: np.ndarray, n_tiepoints: int = 64) -> np.ndarray:
    """
    Downsample a DTW path to a small set of tiepoints.

    Returns array shape (K,2), with K<=n_tiepoints.
    """
    path = np.asarray(path)
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError("path must be (L,2)")
    L = int(path.shape[0])

    n = int(n_tiepoints)
    if n <= 0:
        raise ValueError("n_tiepoints must be > 0")

    if L <= n:
        return path.astype("int64", copy=False)

    idx = np.linspace(0, L - 1, n, dtype="int64")
    # Avoid duplicates for some (L,n) combos; keep sorted order
    idx = np.unique(idx)
    return path[idx].astype("int64", copy=False)
