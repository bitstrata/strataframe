from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class BlockTransform:
    a: float
    b: float


@dataclass(frozen=True)
class TieObs:
    block_a: str
    block_b: str
    rgt_a: float
    rgt_b: float
    w: float


def solve_affine_transforms(
    blocks: List[str],
    ties: List[TieObs],
    *,
    reg_a: float = 10.0,
    reg_b: float = 1.0,
) -> Dict[str, BlockTransform]:
    if not blocks:
        return {}
    ref = blocks[0]
    unk = [b for b in blocks if b != ref]
    m = len(unk)
    if m == 0:
        return {ref: BlockTransform(1.0, 0.0)}

    b2i = {b: i for i, b in enumerate(unk)}

    rows: List[np.ndarray] = []
    rhs: List[float] = []
    wts: List[float] = []

    for t in ties:
        row = np.zeros(2 * m, dtype=float)
        val = 0.0

        def add_term(block: str, sign: float, rgt: float):
            nonlocal val
            if block == ref:
                val += sign * rgt
            else:
                i = b2i[block]
                row[i] += sign * rgt
                row[m + i] += sign

        add_term(t.block_a, +1.0, t.rgt_a)
        add_term(t.block_b, -1.0, t.rgt_b)

        rows.append(row)
        rhs.append(-val)
        wts.append(t.w)

    A = np.vstack(rows) if rows else np.zeros((0, 2*m), dtype=float)
    y = np.asarray(rhs, dtype=float)
    W = np.sqrt(np.asarray(wts, dtype=float)) if wts else np.ones(0)

    if A.shape[0] > 0:
        A = A * W[:, None]
        y = y * W

    Ra = np.zeros((m, 2*m), dtype=float)
    Rb = np.zeros((m, 2*m), dtype=float)
    for i in range(m):
        Ra[i, i] = np.sqrt(reg_a)
        Rb[i, m + i] = np.sqrt(reg_b)

    A = np.vstack([A, Ra, Rb])
    y = np.concatenate([y, np.sqrt(reg_a) * np.ones(m), np.zeros(m)])

    x, *_ = np.linalg.lstsq(A, y, rcond=None)

    out: Dict[str, BlockTransform] = {ref: BlockTransform(1.0, 0.0)}
    for b in unk:
        i = b2i[b]
        out[b] = BlockTransform(a=float(x[i]), b=float(x[m + i]))
    return out
