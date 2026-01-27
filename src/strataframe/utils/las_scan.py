# src/strataframe/utils/las_scan.py
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from strataframe.curves.normalize_header import norm_mnemonic


def list_header_mnemonics(las: Any) -> List[str]:
    """
    Best-effort extraction of curve mnemonics from a LAS-like object.

    Supports (commonly):
      - lasio LASFile: las.curves -> objects with .mnemonic
      - other libs: curves as dicts/tuples/strings
    """
    out: List[str] = []
    curves = getattr(las, "curves", None)

    if curves is None:
        return out

    try:
        for c in curves or []:
            mn = ""
            try:
                # lasio curve objects
                mn = str(getattr(c, "mnemonic", "") or "").strip()
            except Exception:
                mn = ""

            if not mn:
                # tuple/list like ("GR", "GAPI", "Gamma Ray")
                try:
                    if isinstance(c, (tuple, list)) and len(c) >= 1:
                        mn = str(c[0] or "").strip()
                except Exception:
                    mn = ""

            if not mn:
                # dict like {"mnemonic": "GR", ...}
                try:
                    if isinstance(c, dict):
                        mn = str(c.get("mnemonic", "") or "").strip()
                except Exception:
                    mn = ""

            if not mn:
                # bare string
                try:
                    if isinstance(c, str):
                        mn = c.strip()
                except Exception:
                    mn = ""

            if mn:
                out.append(mn)
    except Exception:
        return out

    return out


def finite_count(a: np.ndarray) -> int:
    try:
        x = np.asarray(a, dtype="float64").reshape(-1)
        return int(np.isfinite(x).sum())
    except Exception:
        return 0


def _get_curve_array(las: Any, mnemonic: str) -> Optional[np.ndarray]:
    """
    Best-effort curve fetch as float64 1D array.

    Supports:
      - las[mnemonic] (lasio-style)
      - getattr(las, mnemonic) (rare)
      - las.get(mnemonic) (dict-like)
    """
    mn = (mnemonic or "").strip()
    if not mn:
        return None

    # lasio-style indexing
    try:
        arr = las[mn]  # type: ignore[index]
        x = np.asarray(arr, dtype="float64").reshape(-1)
        return x
    except Exception:
        pass

    # attribute-style
    try:
        arr = getattr(las, mn)
        x = np.asarray(arr, dtype="float64").reshape(-1)
        return x
    except Exception:
        pass

    # dict-like
    try:
        if hasattr(las, "get"):
            arr = las.get(mn)  # type: ignore[call-arg]
            if arr is not None:
                x = np.asarray(arr, dtype="float64").reshape(-1)
                return x
    except Exception:
        pass

    return None


def pick_best_actual_curve(las: Any, *, actual_mnemonics: Sequence[str]) -> Optional[str]:
    """
    Among provided actual mnemonics, pick the one with max finite samples.
    """
    best: Optional[str] = None
    best_n = -1

    for mn in actual_mnemonics:
        x = _get_curve_array(las, str(mn))
        if x is None or x.size == 0:
            continue
        nfin = finite_count(x)
        if nfin > best_n:
            best, best_n = str(mn), nfin

    return best


def pick_best_by_canon_family(las: Any, *, wanted_canons: Sequence[str]) -> Optional[str]:
    """
    Return the best actual mnemonic whose canonical mnemonic matches wanted_canons.

    - Handles suffix mnemonics via norm_mnemonic() (e.g., GR:1 -> GR).
    - If multiple exist, chooses the one with max finite count (requires data loaded).
    """
    wanted_set = {norm_mnemonic(w) for w in wanted_canons if norm_mnemonic(w)}
    if not wanted_set:
        return None

    actuals = list_header_mnemonics(las)
    candidates = [mn for mn in actuals if norm_mnemonic(mn) in wanted_set]

    if not candidates:
        # Last resort: raw prefix match for odd tokens that survive cleaning differently
        raw_wanted = {str(w).strip().upper() for w in wanted_canons if str(w).strip()}
        tmp: List[str] = []
        for mn in actuals:
            up = str(mn).strip().upper()
            if any(up.startswith(w) for w in raw_wanted):
                tmp.append(str(mn))
        # de-dupe preserving order
        seen: set[str] = set()
        candidates = [m for m in tmp if not (m in seen or seen.add(m))]

    if not candidates:
        return None

    return pick_best_actual_curve(las, actual_mnemonics=candidates)


def extract_depth_array(las: Any) -> Tuple[np.ndarray, str]:
    """
    Returns (depth_array, depth_source) where depth_source is 'las.index' or the mnemonic used.
    """
    # Prefer numeric index if present (lasio convention)
    try:
        idx = np.asarray(getattr(las, "index"), dtype="float64").reshape(-1)
        if idx.size and np.isfinite(idx).any():
            return idx, "las.index"
    except Exception:
        pass

    # Fallback: explicit depth curve
    depth_mn = pick_best_by_canon_family(las, wanted_canons=("DEPT", "DEPTH", "MD", "TVD"))
    if depth_mn is None:
        raise RuntimeError("No depth axis found (no numeric las.index and no DEPT/DEPTH/MD/TVD curves).")

    arr = _get_curve_array(las, depth_mn)
    if arr is None or arr.size == 0:
        raise RuntimeError(f"Depth curve {depth_mn!r} could not be read as numeric array.")
    return arr, depth_mn


def compute_percentiles(x: np.ndarray, pct: Sequence[float]) -> List[float]:
    """
    Compute percentiles on finite values only; returns NaN for empty/invalid input.

    pct is interpreted the same way as numpy.percentile (0..100).
    """
    try:
        a = np.asarray(x, dtype="float64").reshape(-1)
    except Exception:
        return [float("nan") for _ in pct]

    fin = np.isfinite(a)
    if int(fin.sum()) == 0:
        return [float("nan") for _ in pct]

    try:
        vals = np.percentile(a[fin], list(pct))
        return [float(v) for v in np.asarray(vals, dtype="float64").reshape(-1)]
    except Exception:
        return [float("nan") for _ in pct]
