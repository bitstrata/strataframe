# src/strataframe/viz/step3_common.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("pandas is required. pip install pandas") from e

try:
    from matplotlib.collections import LineCollection  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("matplotlib is required. pip install matplotlib") from e


# -------------------------
# Basic filesystem helpers
# -------------------------

def ensure_dir(p: Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def find_first(base: Path, names: Sequence[str]) -> Optional[Path]:
    base = Path(base)
    for n in names:
        cand = base / str(n)
        if cand.exists():
            return cand
    return None


# -------------------------
# CSV normalization helpers
# -------------------------

def pick_col(df: "pd.DataFrame", cols: Sequence[str]) -> Optional[str]:
    """
    Return the first matching column name from `cols`, case-insensitive.
    """
    if df is None or df.empty:
        # still allow matching on columns even if empty
        pass

    col_map = {str(c).strip().lower(): str(c) for c in df.columns}
    for c in cols:
        key = str(c).strip().lower()
        if key in col_map:
            return col_map[key]
    return None


def _require_col(df: "pd.DataFrame", name: str, *, path: Path) -> None:
    if name not in df.columns:
        raise ValueError(f"CSV missing required column '{name}': {path}")


def load_reps(reps_csv: Path) -> "pd.DataFrame":
    """
    Load representatives.csv (Step2). Normalizes columns to:
      rep_id (str), bin_id (str), lat/lon (float), optional score (float)

    Notes:
      - Supports legacy aliases for rep_id: well_id / id / node_id
      - Supports bin id aliasing: bin_id <- h3_cell if needed (grid still uses h3_cell as cell id)
    """
    reps_csv = Path(reps_csv)
    df = pd.read_csv(reps_csv)

    # rep_id normalization
    if "rep_id" not in df.columns:
        c = pick_col(df, ["rep_id", "well_id", "id", "node_id"])
        if c is None:
            raise ValueError(f"reps CSV missing rep_id (or fallback well_id/id/node_id): {reps_csv}")
        if c != "rep_id":
            df = df.rename(columns={c: "rep_id"})
    df["rep_id"] = df["rep_id"].astype(str)

    # bin_id normalization
    if "bin_id" not in df.columns:
        if "h3_cell" in df.columns:
            df["bin_id"] = df["h3_cell"].astype(str)
        elif "grid_cell" in df.columns:
            df["bin_id"] = df["grid_cell"].astype(str)
        else:
            df["bin_id"] = "BIN0"
    else:
        df["bin_id"] = df["bin_id"].astype(str)

    # lat/lon normalization (case-insensitive + common aliases)
    lat_c = pick_col(df, ["lat", "latitude"])
    lon_c = pick_col(df, ["lon", "longitude", "long"])
    if lat_c is None or lon_c is None:
        raise ValueError(f"reps CSV missing lat/lon columns: {reps_csv}. Found: {list(df.columns)}")

    if lat_c != "lat":
        df = df.rename(columns={lat_c: "lat"})
    if lon_c != "lon":
        df = df.rename(columns={lon_c: "lon"})

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    if "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce")

    return df


def load_edges_csv(path: Path) -> "pd.DataFrame":
    """
    Load an edges CSV and normalize endpoints to columns:
      u (str), v (str)

    Accepts many historical column aliases.
    """
    path = Path(path)
    df = pd.read_csv(path)

    cu = pick_col(df, ["u", "src", "src_id", "src_rep_id", "rep_i", "i", "node_u"])
    cv = pick_col(df, ["v", "dst", "dst_id", "dst_rep_id", "rep_j", "j", "node_v"])
    if cu is None or cv is None:
        raise ValueError(f"Edges CSV {path} missing recognizable endpoint columns. Found: {list(df.columns)}")

    if cu != "u" or cv != "v":
        df = df.rename(columns={cu: "u", cv: "v"})

    df["u"] = df["u"].astype(str)
    df["v"] = df["v"].astype(str)
    return df


# -------------------------
# Line drawing helper
# -------------------------

def line_collection_from_edges(
    reps: "pd.DataFrame",
    edges: "pd.DataFrame",
    *,
    max_edges: int = 50_000,
    linewidth: float = 0.6,
    alpha: float = 0.35,
) -> Tuple["LineCollection", int]:
    """
    Build a matplotlib LineCollection from edges (u,v) using reps lon/lat.

    Returns (LineCollection, n_used).
    """
    if reps is None or edges is None:
        return LineCollection([]), 0

    # Normalize required cols (case-insensitive safety)
    if "rep_id" not in reps.columns:
        c = pick_col(reps, ["rep_id", "well_id", "id", "node_id"])
        if c is None:
            raise ValueError("reps missing rep_id (or fallback aliases).")
        if c != "rep_id":
            reps = reps.rename(columns={c: "rep_id"})
    if "lon" not in reps.columns or "lat" not in reps.columns:
        lon_c = pick_col(reps, ["lon", "longitude", "long"])
        lat_c = pick_col(reps, ["lat", "latitude"])
        if lon_c is None or lat_c is None:
            raise ValueError("reps missing lon/lat (or aliases).")
        if lon_c != "lon":
            reps = reps.rename(columns={lon_c: "lon"})
        if lat_c != "lat":
            reps = reps.rename(columns={lat_c: "lat"})

    if "u" not in edges.columns or "v" not in edges.columns:
        # attempt normalize
        edges = load_edges_csv(Path("<in-memory>"))  # pragma: no cover  (won't execute; placeholder)
        # above is unreachable in normal use; keep explicit error instead
        raise ValueError("edges missing required columns u/v.")

    pts = reps.set_index("rep_id")[["lon", "lat"]]
    segs: List[np.ndarray] = []
    n_used = 0

    # iterrows is fine here (max_edges bounded)
    for _, r in edges.iterrows():
        if n_used >= int(max_edges):
            break
        u = str(r["u"])
        v = str(r["v"])
        if u not in pts.index or v not in pts.index:
            continue
        a = pts.loc[u].to_numpy(dtype="float64")
        b = pts.loc[v].to_numpy(dtype="float64")
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            continue
        segs.append(np.vstack([a, b]))
        n_used += 1

    lc = LineCollection(segs, linewidths=float(linewidth), alpha=float(alpha))
    return lc, n_used


# -------------------------
# NPZ helpers
# -------------------------

def load_npz(path: Path) -> Dict[str, np.ndarray]:
    path = Path(path)
    z = np.load(path, allow_pickle=False)
    return {k: np.asarray(z[k]) for k in z.files}


def rep_arrays_from_npz(z: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Accepts NPZ dicts with keys:
      rep_ids: (R,) str
      depth_rs or depth: (R,nS) or (nS,)
      log_rs / gr_rs / gr / log: (R,nS) or (nS,)
      imputed_mask: optional bool mask (same shape as log)

    Returns:
      rep_ids (R,), depth, log, imputed_mask(optional)
    """
    if "rep_ids" not in z:
        raise ValueError("rep_arrays npz missing key 'rep_ids'.")

    rep_ids = np.asarray(z["rep_ids"], dtype=np.str_)

    depth_key = "depth_rs" if "depth_rs" in z else ("depth" if "depth" in z else None)
    if depth_key is None:
        raise ValueError("rep_arrays npz missing depth array (expected 'depth_rs' or 'depth').")

    log_key = None
    for k in ("log_rs", "gr_rs", "gr", "log"):
        if k in z:
            log_key = k
            break
    if log_key is None:
        raise ValueError("rep_arrays npz missing log array (expected 'log_rs' or 'gr_rs' or 'gr' or 'log').")

    depth = np.asarray(z[depth_key], dtype="float64")
    log = np.asarray(z[log_key], dtype="float64")

    imputed = None
    if "imputed_mask" in z:
        imputed = np.asarray(z["imputed_mask"], dtype=bool)

    return rep_ids, depth, log, imputed


# -------------------------
# DTW paths loader (best-effort across variants)
# -------------------------

def load_dtw_paths_dict(dtw_paths_npz: Optional[Path]) -> Dict[Tuple[str, str], np.ndarray]:
    """
    Return dict[(u,v)] -> path array. Safe no-op if file missing.

    Supports multiple historical helpers:
      - strataframe.correlation.paths_npz.unpack_paths(path)
      - strataframe.correlation.paths_npz.load_paths_npz(path).to_dict()
    """
    if dtw_paths_npz is None:
        return {}
    dtw_paths_npz = Path(dtw_paths_npz)
    if not dtw_paths_npz.exists():
        return {}

    # Preferred (newer helper name in your repo variants)
    try:
        from strataframe.correlation.paths_npz import unpack_paths  # type: ignore
        out = unpack_paths(str(dtw_paths_npz))
        return out if isinstance(out, dict) else {}
    except Exception:
        pass

    # Alternate packed object loader
    try:
        from strataframe.correlation.paths_npz import load_paths_npz  # type: ignore
        packed = load_paths_npz(str(dtw_paths_npz))
        if hasattr(packed, "to_dict"):
            out = packed.to_dict()
            return out if isinstance(out, dict) else {}
    except Exception:
        pass

    return {}
