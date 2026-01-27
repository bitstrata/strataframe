# src/strataframe/rgt/wavelet_tops.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

try:
    from scipy.signal import ricker, fftconvolve  # type: ignore
except Exception:  # pragma: no cover
    ricker = None  # type: ignore
    fftconvolve = None  # type: ignore


@dataclass(frozen=True)
class CwtConfig:
    """
    ChronoLog 2.6: CWT with Ricker wavelets at multiple scales.

    widths: increasing -> lower frequency / coarser scale.
    """
    widths: Sequence[int] = (2, 4, 6, 8, 12, 16, 24, 32)
    snap_window: int = 25
    include_endpoints: bool = True


def _require_scipy() -> None:
    if ricker is None or fftconvolve is None:
        raise RuntimeError("scipy is required for CWT. Install with: pip install scipy")


def _cwt_ricker(x: np.ndarray, widths: Sequence[int]) -> np.ndarray:
    """
    Minimal CWT via convolution with Ricker wavelets.
    Returns array W of shape (n_scales, n_samples).
    """
    _require_scipy()
    x = np.asarray(x, dtype="float64")
    if x.size == 0:
        return np.zeros((0, 0), dtype="float64")

    out = np.zeros((len(widths), x.size), dtype="float64")
    for i, w in enumerate(widths):
        a = max(1, int(w))
        # ~10*a is common stable support; ensure odd length
        M = max(5, int(10 * a))
        if M % 2 == 0:
            M += 1

        wave = ricker(M, a)  # type: ignore[misc]

        # Normalize per-scale energy (L1 keeps response magnitudes comparable)
        wave = wave / (np.sum(np.abs(wave)) + 1e-12)

        out[i, :] = fftconvolve(x, wave, mode="same")  # type: ignore[misc]

    return out


def _zero_crossings_to_sample_indices(y: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """
    Find zero crossings robustly and return *sample indices* (0..n-1).

    We treat a crossing between i and i+1 as a "top" located at i+1, so
    returned indices map directly onto rgt_grid/depth_grid columns.

    Returns indices in [1, n-1] (interior crossings); endpoints are handled separately.
    """
    y = np.asarray(y, dtype="float64")
    n = int(y.size)
    if n < 2:
        return np.array([], dtype=int)

    # Stabilize near-zeros
    yy = y.copy()
    yy[np.abs(yy) <= eps] = 0.0

    # Build sign array in {-1, 0, +1}
    s = np.sign(yy).astype(int)

    # Fill zeros by carrying last nonzero sign forward, then backward
    # so flat/zero runs don’t break sign-change detection.
    if np.any(s == 0):
        # forward fill
        last = 0
        for i in range(n):
            if s[i] != 0:
                last = s[i]
            else:
                s[i] = last
        # backward fill (handles leading zeros)
        last = 0
        for i in range(n - 1, -1, -1):
            if s[i] != 0:
                last = s[i]
            else:
                s[i] = last

    # Crossings where sign flips between consecutive samples
    flips = np.where((s[:-1] != 0) & (s[1:] != 0) & (s[:-1] != s[1:]))[0]

    # Convert boundary index i (between i and i+1) -> sample index i+1
    idx = (flips + 1).astype(int)

    # Clamp to valid interior (should already be)
    idx = idx[(idx >= 1) & (idx <= n - 1)]
    return idx


def _trace_indices(coarse: np.ndarray, fine: np.ndarray, *, window: int) -> np.ndarray:
    """
    For each index in coarse, snap to nearest in fine within +-window.
    If none found, keep original.
    """
    coarse = np.asarray(coarse, dtype=int)
    fine = np.asarray(fine, dtype=int)
    if coarse.size == 0 or fine.size == 0:
        return coarse.copy()

    fine = np.unique(fine)
    out = coarse.copy()

    for i, c in enumerate(coarse):
        lo = c - window
        hi = c + window
        cand = fine[(fine >= lo) & (fine <= hi)]
        if cand.size == 0:
            continue
        out[i] = int(cand[np.argmin(np.abs(cand - c))])

    return out


def derive_tops_multiscale(type_log: np.ndarray, *, cfg: CwtConfig) -> Dict[str, object]:
    """
    Returns dict:
      - widths: list[int]
      - W: (n_scales, n) CWT output
      - tops_by_level: list[list[int]] where level i corresponds to cfg.widths[i]
        Each list contains sample indices (0..n-1) suitable for indexing rgt_grid/depth_grid.
    """
    x = np.asarray(type_log, dtype="float64")
    n = int(x.size)

    widths = [int(w) for w in cfg.widths if int(w) > 0]
    if n == 0 or not widths:
        return {"widths": widths, "W": np.zeros((0, 0), dtype="float64"), "tops_by_level": []}

    W = _cwt_ricker(x, widths)

    # zero-crossings per level (level i corresponds to widths[i])
    zeros = [_zero_crossings_to_sample_indices(W[i, :]) for i in range(len(widths))]

    tops_by_level: List[List[int]] = []
    for k in range(len(widths)):
        z = zeros[k].copy()
        # trace indices down to finest (0) for hierarchical consistency
        for j in range(k - 1, -1, -1):
            z = _trace_indices(z, zeros[j], window=int(cfg.snap_window))

        z = np.unique(z)
        z.sort()

        if cfg.include_endpoints:
            z = np.unique(np.concatenate([np.array([0], dtype=int), z, np.array([n - 1], dtype=int)]))
            z.sort()

        tops_by_level.append([int(v) for v in z.tolist()])

    return {"widths": widths, "W": W, "tops_by_level": tops_by_level}
