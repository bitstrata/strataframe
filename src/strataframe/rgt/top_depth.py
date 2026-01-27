# src/strataframe/rgt/top_depth.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass(frozen=True)
class TopsExportConfig:
    """
    Exporting tops per well for a given hierarchy level.
    """
    level: int = 0  # index into tops_by_level
    prefix: str = "TOP"


def tops_to_depth_table(
    *,
    node_order: Sequence[str],
    depth_grid: np.ndarray,     # (W,T)
    rgt_grid: np.ndarray,       # (T,)
    tops_by_level: Sequence[Sequence[int]],
    cfg: TopsExportConfig,
) -> List[Dict[str, object]]:
    """
    Returns rows: one per well, with depth and rgt values for each top index.

    depth_grid is the robust inversion (depth as a function of RGT) and will be NaN
    where a well does not cover that RGT. We emit blanks for NaNs (depth only).
    """
    depth_grid = np.asarray(depth_grid, dtype="float64")
    rgt_grid = np.asarray(rgt_grid, dtype="float64")

    if depth_grid.ndim != 2:
        raise ValueError(f"depth_grid must be 2D (W,T). Got ndim={depth_grid.ndim}")
    if rgt_grid.ndim != 1:
        raise ValueError(f"rgt_grid must be 1D (T,). Got ndim={rgt_grid.ndim}")

    W, T = depth_grid.shape
    if len(node_order) != W:
        raise ValueError("node_order length does not match depth_grid rows.")
    if rgt_grid.size != T:
        raise ValueError("rgt_grid length does not match depth_grid columns.")

    level = int(cfg.level)
    if level < 0 or level >= len(tops_by_level):
        raise ValueError(f"Requested level {level} out of range for tops_by_level (n={len(tops_by_level)}).")

    # Normalize/validate top indices
    tops_idx = []
    for i in tops_by_level[level]:
        try:
            ii = int(i)
        except Exception:
            continue
        if 0 <= ii < T:
            tops_idx.append(ii)
    tops_idx = sorted(set(tops_idx))

    # Precompute column names + RGT strings once (RGT is global, not per-well)
    top_defs = []
    for ti, idx in enumerate(tops_idx):
        col_d = f"{cfg.prefix}{ti:03d}_DEPTH"
        col_r = f"{cfg.prefix}{ti:03d}_RGT"
        r = float(rgt_grid[idx])
        r_str = "" if not np.isfinite(r) else f"{r:.3f}"
        top_defs.append((idx, col_d, col_r, r_str))

    rows: List[Dict[str, object]] = []
    for wi, well_id in enumerate(node_order):
        row: Dict[str, object] = {"well_id": well_id}
        dg = depth_grid[wi]

        for idx, col_d, col_r, r_str in top_defs:
            d = float(dg[idx])
            row[col_d] = "" if not np.isfinite(d) else f"{d:.3f}"
            row[col_r] = r_str  # emit RGT regardless of depth coverage

        rows.append(row)

    return rows


def tops_depth_fieldnames(*, rgt_grid: np.ndarray, tops_by_level: Sequence[Sequence[int]], cfg: TopsExportConfig) -> List[str]:
    rgt_grid = np.asarray(rgt_grid)
    T = int(rgt_grid.size)
    level = int(cfg.level)
    tops = sorted(set(int(i) for i in tops_by_level[level] if 0 <= int(i) < T))
    cols = ["well_id"]
    for ti, _idx in enumerate(tops):
        cols.append(f"{cfg.prefix}{ti:03d}_DEPTH")
        cols.append(f"{cfg.prefix}{ti:03d}_RGT")
    return cols
