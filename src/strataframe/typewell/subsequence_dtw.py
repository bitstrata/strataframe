# src/strataframe/typewell/subsequence_dtw.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SubseqDTWResult:
    cost_total: float
    cost_per_step: float
    path: np.ndarray            # (L,2) as (i_query, j_template)
    j_start: int
    j_end: int


def subsequence_dtw(
    x: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = 0.15,
) -> SubseqDTWResult:
    """
    Subsequence DTW: align x (query) to any contiguous subsequence of y (template).
    Uses local distance |x_i - y_j|^alpha (ChronoLog-style robustness).

    DP convention:
      dp[0, j] = 0 for all j (free start in y)
      dp[i, 0] = inf for i>0
      dp[i, j] = c(i,j) + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])

    Best end is argmin_j dp[n, j].
    """
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays")
    if x.size < 8 or y.size < 8:
        raise ValueError("x and y too short for DTW")
    if alpha <= 0.0:
        raise ValueError("alpha must be > 0")

    n = int(x.size)
    m = int(y.size)

    # cost matrix (n,m)
    C = np.abs(x[:, None] - y[None, :]) ** float(alpha)

    inf = 1e30
    dp = np.full((n + 1, m + 1), inf, dtype="float64")
    bt = np.full((n + 1, m + 1), -1, dtype="int8")  # 0=diag,1=up,2=left

    dp[0, :] = 0.0
    dp[1:, 0] = inf

    for i in range(1, n + 1):
        ci = C[i - 1]
        for j in range(1, m + 1):
            a = dp[i - 1, j - 1]  # diag
            b = dp[i - 1, j]      # up
            c = dp[i, j - 1]      # left
            if a <= b and a <= c:
                dp[i, j] = ci[j - 1] + a
                bt[i, j] = 0
            elif b <= c:
                dp[i, j] = ci[j - 1] + b
                bt[i, j] = 1
            else:
                dp[i, j] = ci[j - 1] + c
                bt[i, j] = 2

    j_end = int(np.argmin(dp[n, 1:])) + 1
    cost_total = float(dp[n, j_end])

    # backtrack
    i = n
    j = j_end
    path = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        move = int(bt[i, j])
        if move == 0:
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1

    if not path:
        raise RuntimeError("Subsequence DTW produced empty path")

    path = np.asarray(path[::-1], dtype="int64")
    js = path[:, 1]
    j_start = int(np.min(js))
    j_end0 = int(np.max(js))

    cost_per_step = cost_total / float(max(1, path.shape[0]))
    return SubseqDTWResult(
        cost_total=float(cost_total),
        cost_per_step=float(cost_per_step),
        path=path,
        j_start=j_start,
        j_end=j_end0,
    )


def path_to_template_indices(path: np.ndarray, m: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reduce (i,j) path into a template-indexed mapping.
    Returns:
      j_idx (unique sorted template indices),
      x_at_j (aggregated x values mapped to each j, mean if multiple).
    """
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError("path must be (L,2)")
    js = path[:, 1]
    is_ = path[:, 0]
    order = np.argsort(js)
    js = js[order]
    is_ = is_[order]

    j_unique = []
    i_groups = []
    cur = None
    grp = []
    for j, i in zip(js.tolist(), is_.tolist()):
        if cur is None or j != cur:
            if cur is not None:
                j_unique.append(cur)
                i_groups.append(grp)
            cur = j
            grp = [i]
        else:
            grp.append(i)
    if cur is not None:
        j_unique.append(cur)
        i_groups.append(grp)

    j_idx = np.asarray(j_unique, dtype="int64")
    # Placeholder for x values; caller will fill using x vector
    return j_idx, np.asarray([0.0] * j_idx.size, dtype="float64")
