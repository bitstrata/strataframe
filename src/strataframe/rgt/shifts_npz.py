# src/strataframe/rgt/shifts_npz.py
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np


def save_shifts_npz(shifts: Dict[str, np.ndarray], out_path: str | Path) -> None:
    """
    Save shifts as:
      rep_ids: (R,) str
      shifts:  (R,nS) float64
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rep_ids = np.asarray(sorted([str(k) for k in shifts.keys()]), dtype=np.str_)
    if rep_ids.size == 0:
        np.savez_compressed(out_path, rep_ids=rep_ids, shifts=np.zeros((0, 0), dtype="float64"))
        return

    S = np.stack([np.asarray(shifts[str(r)], dtype="float64") for r in rep_ids.tolist()], axis=0)
    np.savez_compressed(out_path, rep_ids=rep_ids, shifts=S)


def load_shifts_npz(path: str | Path) -> Dict[str, np.ndarray]:
    """
    Load shifts NPZ written by save_shifts_npz() (or the internal _save_shifts_npz()).
    Returns dict: rep_id -> (nS,) float64 shift vector.
    """
    z = np.load(Path(path), allow_pickle=False)
    rep_ids = np.asarray(z["rep_ids"], dtype=np.str_)
    S = np.asarray(z["shifts"], dtype="float64")

    if S.ndim != 2:
        raise ValueError(f"Invalid shifts NPZ: shifts must be 2D (R,nS). Got ndim={S.ndim}")
    if rep_ids.size != S.shape[0]:
        raise ValueError("Invalid shifts NPZ: rep_ids length must match shifts first dimension")

    out: Dict[str, np.ndarray] = {}
    for i, rid in enumerate(rep_ids.tolist()):
        out[str(rid)] = S[i, :].copy()
    return out
