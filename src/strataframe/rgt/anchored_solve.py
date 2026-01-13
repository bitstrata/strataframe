from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg


@dataclass(frozen=True)
class Anchor:
    idx_global: int
    target_shift: float
    weight: float


@dataclass(frozen=True)
class PairEq:
    i: int
    j: int
    dz: float
    w: float = 1.0


def solve_shifts_operator(
    n: int,
    pair_eqs: List[PairEq],
    anchors: List[Anchor],
    lambda_anchor: float,
    tol: float = 1e-6,
    maxiter: int = 2000,
) -> np.ndarray:
    pe = pair_eqs
    b = np.zeros(n, dtype=float)
    for e in pe:
        b[e.i] += e.w * e.dz
        b[e.j] -= e.w * e.dz

    if anchors:
        a_idx = np.array([a.idx_global for a in anchors], dtype=np.int32)
        a_w = np.array([a.weight for a in anchors], dtype=float)
        a_t = np.array([a.target_shift for a in anchors], dtype=float)
    else:
        a_idx = np.array([], dtype=np.int32)
        a_w = np.array([], dtype=float)
        a_t = np.array([], dtype=float)

    b2 = b.copy()
    if anchors:
        b2[a_idx] += lambda_anchor * (a_w * a_w) * a_t

    def matvec(x: np.ndarray) -> np.ndarray:
        y = np.zeros_like(x)
        for e in pe:
            d = e.w * (x[e.i] - x[e.j])
            y[e.i] += d
            y[e.j] -= d
        if anchors:
            y[a_idx] += lambda_anchor * (a_w * a_w) * x[a_idx]
        return y

    Aop = LinearOperator((n, n), matvec=matvec, dtype=float)
    s, info = cg(Aop, b2, rtol=tol, maxiter=maxiter)
    if info != 0:
        raise RuntimeError(f"CG did not converge: info={info}")
    return s
