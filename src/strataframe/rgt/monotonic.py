# src/strataframe/rgt/monotonic.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np


MonotonicMode = Literal["nondecreasing", "nonincreasing"]


@dataclass(frozen=True)
class MonotonicConfig:
    """
    ChronoLog 2.5 'No going back in time'.

    If your RGT increases with depth (older downward), use mode='nondecreasing'
    (cumulative maximum).

    If your RGT decreases with depth (older upward), use mode='nonincreasing'
    (cumulative minimum).

    The paper text is direction-ambiguous without the sign convention; this config
    makes the choice explicit and easy to flip.
    """
    mode: MonotonicMode = "nondecreasing"


def enforce_monotonic_rgt(
    rgt: np.ndarray,
    *,
    cfg: MonotonicConfig,
    atol: float = 1e-10,
    rtol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      rgt_fixed: monotonic series with reversals flattened into constant-RGT intervals
      reversed_mask: boolean mask where original rgt violated monotonicity (approximate)

    Notes:
      - Enforcement is applied only to finite samples; NaNs are preserved.
      - reversed_mask uses tolerant float comparison to avoid spurious flags.
    """
    rgt = np.asarray(rgt, dtype="float64")
    if rgt.size == 0:
        return rgt.copy(), np.zeros_like(rgt, dtype=bool)

    fin = np.isfinite(rgt)
    out = rgt.copy()

    if fin.sum() <= 1:
        return out, np.zeros_like(rgt, dtype=bool)

    x = out[fin]

    if cfg.mode == "nondecreasing":
        fixed = np.maximum.accumulate(x)
    elif cfg.mode == "nonincreasing":
        fixed = np.minimum.accumulate(x)
    else:
        raise ValueError(f"Unknown MonotonicConfig.mode: {cfg.mode}")

    # tolerant reversal marking
    rev = np.zeros_like(out, dtype=bool)
    idx = np.where(fin)[0]
    changed = ~np.isclose(fixed, x, atol=float(atol), rtol=float(rtol))
    rev[idx] = changed

    out[fin] = fixed
    return out, rev
