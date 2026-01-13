from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class Well:
    well_id: str
    x: float
    y: float
    md: np.ndarray
    logs: Dict[str, np.ndarray]


def load_wells_parquet(path: str, log_cols: Optional[List[str]] = None) -> List[Well]:
    """
    Expected long format:
      well_id, x, y, md, GR, (optional other logs...)
    """
    df = pd.read_parquet(path)

    if "well_id" not in df.columns:
        raise ValueError("Expected column 'well_id'")
    for c in ("x", "y", "md"):
        if c not in df.columns:
            raise ValueError(f"Expected column '{c}'")

    if log_cols is None:
        skip = {"well_id", "x", "y", "md"}
        log_cols = [c for c in df.columns if c not in skip]

    wells: List[Well] = []
    for wid, g in df.groupby("well_id", sort=False):
        g2 = g.sort_values("md")
        logs = {c: g2[c].to_numpy(dtype=float) for c in log_cols if c in g2.columns}
        wells.append(
            Well(
                well_id=str(wid),
                x=float(g2["x"].iloc[0]),
                y=float(g2["y"].iloc[0]),
                md=g2["md"].to_numpy(dtype=float),
                logs=logs,
            )
        )
    return wells
