# src/strataframe/io/tops.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd


@dataclass(frozen=True)
class TopPick:
    top_name: str
    depth: float
    sigma_m: float


def load_tops(
    path: Union[str, Path],
    *,
    default_sigma_m: float = 5.0,
    sort_by_depth: bool = True,
) -> Dict[str, List[TopPick]]:
    """
    Load tops from a parquet file into a mapping: well_id -> list[TopPick].

    Expected columns (case-sensitive):
      - well_id (any dtype; coerced to str)
      - top_name (any dtype; coerced to str)
      - depth (numeric; coerced; non-numeric dropped)
      - sigma_m (optional; numeric; if missing/NaN -> default_sigma_m)

    Notes:
      - Rows with missing/blank well_id or top_name are dropped.
      - Rows with non-numeric depth are dropped.
      - If sort_by_depth=True, picks are sorted ascending by depth within each well.
    """
    p = Path(path)

    df = pd.read_parquet(p)

    req = {"well_id", "top_name", "depth"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)} (in {p})")

    # Coerce to sane types
    out_df = df.loc[:, [c for c in ("well_id", "top_name", "depth", "sigma_m") if c in df.columns]].copy()

    out_df["well_id"] = out_df["well_id"].astype(str)
    out_df["top_name"] = out_df["top_name"].astype(str)

    out_df["depth"] = pd.to_numeric(out_df["depth"], errors="coerce")

    if "sigma_m" in out_df.columns:
        out_df["sigma_m"] = pd.to_numeric(out_df["sigma_m"], errors="coerce").fillna(float(default_sigma_m))
    else:
        out_df["sigma_m"] = float(default_sigma_m)

    # Drop unusable rows
    out_df["well_id"] = out_df["well_id"].str.strip()
    out_df["top_name"] = out_df["top_name"].str.strip()
    out_df = out_df[(out_df["well_id"] != "") & (out_df["top_name"] != "")]
    out_df = out_df[out_df["depth"].notna()]

    out: Dict[str, List[TopPick]] = {}

    # Group and build TopPick objects efficiently
    for wid, g in out_df.groupby("well_id", sort=False):
        if sort_by_depth:
            g = g.sort_values("depth", kind="mergesort")  # stable sort

        picks: List[TopPick] = []
        for r in g.itertuples(index=False):
            picks.append(
                TopPick(
                    top_name=str(getattr(r, "top_name")),
                    depth=float(getattr(r, "depth")),
                    sigma_m=float(getattr(r, "sigma_m")),
                )
            )
        out[str(wid)] = picks

    return out
