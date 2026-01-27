# src/strataframe/io/wells.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from strataframe.curves.normalize_header import norm_mnemonic, aliases_for


@dataclass
class Well:
    well_id: str
    x: float
    y: float
    md: np.ndarray
    logs: Dict[str, np.ndarray]


def _finite_count(a: np.ndarray) -> int:
    try:
        x = np.asarray(a, dtype="float64")
        return int(np.isfinite(x).sum())
    except Exception:
        return 0


def _build_canon_col_index(columns: Sequence[str]) -> Dict[str, List[str]]:
    """
    Map canonical mnemonic -> list of raw column names that collapse to it.
    """
    idx: Dict[str, List[str]] = {}
    for c in columns:
        cc = norm_mnemonic(c)
        if not cc:
            continue
        idx.setdefault(cc, []).append(c)
    return idx


def _pick_best_column(g: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Pick the best raw column among candidates using finite sample count.
    """
    best: Optional[str] = None
    best_n = -1
    for c in candidates:
        if c not in g.columns:
            continue
        n = _finite_count(g[c].to_numpy())
        if n > best_n:
            best, best_n = c, n
    return best


def load_wells_parquet(
    path: str,
    log_cols: Optional[List[str]] = None,
    *,
    canonicalize: bool = True,
) -> List[Well]:
    """
    Expected long format:
      well_id, x, y, md, GR, (optional other logs...)

    If canonicalize=True (default):
      - Log keys in Well.logs are canonical mnemonics (e.g., 'GR', 'RHOB').
      - If multiple raw columns map to same canonical mnemonic, we keep the one
        with the most finite samples (per-well), tie-break by first seen.
      - If log_cols is provided, entries are treated as requested curves and are
        resolved via canonical compares + aliases_for().
    """
    df = pd.read_parquet(path)

    if "well_id" not in df.columns:
        raise ValueError("Expected column 'well_id'")
    for c in ("x", "y", "md"):
        if c not in df.columns:
            raise ValueError(f"Expected column '{c}'")

    skip = {"well_id", "x", "y", "md"}

    # Determine candidate log columns (raw)
    if log_cols is None:
        raw_log_cols = [c for c in df.columns if c not in skip]
        requested: Optional[List[str]] = None
    else:
        # Only consider columns that exist; resolution happens per-well when canonicalize=True
        raw_log_cols = [c for c in log_cols if c in df.columns]
        requested = list(log_cols)

    wells: List[Well] = []

    for wid, g in df.groupby("well_id", sort=False):
        g2 = g.sort_values("md")

        if not canonicalize:
            logs = {c: g2[c].to_numpy(dtype=float) for c in raw_log_cols if c in g2.columns}
        else:
            # Build canonical index from *available* columns in this frame
            canon_index = _build_canon_col_index([c for c in g2.columns if c not in skip])

            logs_canon: Dict[str, np.ndarray] = {}

            if requested is None:
                # Canonicalize everything (non-index columns)
                for canon_mn, cols in canon_index.items():
                    best = _pick_best_column(g2, cols)
                    if best is None:
                        continue
                    logs_canon[canon_mn] = g2[best].to_numpy(dtype=float)
            else:
                # Resolve requested curves by canonical mnemonic + family aliases
                for req in requested:
                    req_canon = norm_mnemonic(req)
                    if not req_canon:
                        continue

                    # Preferred search order: canonical itself, then family aliases
                    fam = aliases_for(req_canon)
                    # Convert family into canonical mnemonics as well (defensive)
                    fam_canon = [norm_mnemonic(x) for x in fam if norm_mnemonic(x)]

                    # Collect candidate raw columns across the family
                    cand_raw: List[str] = []
                    for fmn in fam_canon:
                        cand_raw.extend(canon_index.get(fmn, []))

                    if not cand_raw:
                        continue

                    best = _pick_best_column(g2, cand_raw)
                    if best is None:
                        continue

                    # Store under the requested canonical key (not the raw/best alias)
                    logs_canon[req_canon] = g2[best].to_numpy(dtype=float)

                # Also: if requested contains duplicates that collapse, collision is last-write.
                # That’s fine because req_canon is the key; if you want a different policy later,
                # we can make it deterministic (e.g., keep max finite).

            logs = logs_canon

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
