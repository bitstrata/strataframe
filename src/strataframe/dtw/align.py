from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from dtaidistance import dtw as dtw_fast  # type: ignore
except Exception:
    dtw_fast = None


@dataclass(frozen=True)
class DTWResult:
    dist: float
    path: Optional[np.ndarray]


def dtw_align(
    x: np.ndarray,
    y: np.ndarray,
    *,
    band_frac: float,
    use_path: bool = True,
) -> DTWResult:
    if dtw_fast is None:
        raise RuntimeError("Install DTW extras: pip install -e '.[dtw]'")

    w = int(max(len(x), len(y)) * band_frac)
    dist = float(dtw_fast.distance_fast(x, y, window=w))

    if not use_path:
        return DTWResult(dist=dist, path=None)

    path = np.asarray(dtw_fast.warping_path(x, y, window=w), dtype=np.int32)
    return DTWResult(dist=dist, path=path)
