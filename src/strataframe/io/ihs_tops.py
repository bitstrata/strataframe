from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


@dataclass(frozen=True)
class TopPick:
    top_name: str
    depth: float
    sigma_m: float


def load_ihs_tops(path: str) -> Dict[str, List[TopPick]]:
    """
    Expected columns:
      well_id, top_name, depth, sigma_m
    """
    df = pd.read_parquet(path)
    req = {"well_id", "top_name", "depth"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out: Dict[str, List[TopPick]] = {}
    for wid, g in df.groupby("well_id", sort=False):
        picks: List[TopPick] = []
        for _, r in g.iterrows():
            picks.append(
                TopPick(
                    top_name=str(r["top_name"]),
                    depth=float(r["depth"]),
                    sigma_m=float(r["sigma_m"]) if "sigma_m" in df.columns else 5.0,
                )
            )
        out[str(wid)] = picks
    return out
