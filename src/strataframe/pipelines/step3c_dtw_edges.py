# src/strataframe/pipelines/step3c_dtw_edges.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from strataframe.correlation.dtw import DtwConfig, dtw_path_and_cost
from strataframe.correlation.paths_npz import pack_paths, save_paths_npz


@dataclass(frozen=True)
class DtwEdgesConfig:
    a_key: str = "log_rs"    # key in rep_arrays dict
    z_key: str = "depth_rs"  # key in rep_arrays dict (optional here)
    downsample_path_to: int = 80  # tiepoints per edge
    min_sim_length: int = 8       # defensive input length


def _require_pd() -> None:
    if pd is None:
        raise RuntimeError("pandas is required. Install with: pip install pandas")


def _canon(u: str, v: str) -> Tuple[str, str]:
    u = str(u)
    v = str(v)
    return (u, v) if u < v else (v, u)


def run_dtw_on_candidate_edges(
    rep_arrays: Dict[str, Dict[str, np.ndarray]],
    candidates: "pd.DataFrame",
    *,
    dtw_cfg: DtwConfig,
    cfg: DtwEdgesConfig = DtwEdgesConfig(),
) -> Tuple["pd.DataFrame", List[Tuple[str, str]], List[np.ndarray]]:
    """
    rep_arrays:
      mapping rep_id -> {"log_rs": (nS,), "depth_rs": (nS,), ...}

    candidates columns:
      src_rep_id, dst_rep_id, dist_km, source

    Returns:
      dtw_edges_df, edge_ids, paths
    """
    _require_pd()

    need_cols = {"src_rep_id", "dst_rep_id"}
    if not need_cols.issubset(set(candidates.columns)):
        raise ValueError(f"candidates must contain columns: {sorted(need_cols)}")

    rows: List[dict] = []
    edge_ids: List[Tuple[str, str]] = []
    paths: List[np.ndarray] = []

    for r in candidates.itertuples(index=False):
        u0 = str(getattr(r, "src_rep_id"))
        v0 = str(getattr(r, "dst_rep_id"))
        u, v = _canon(u0, v0)

        au = rep_arrays.get(u)
        av = rep_arrays.get(v)
        if au is None or av is None:
            rows.append(
                {
                    "src_rep_id": u,
                    "dst_rep_id": v,
                    "dtw_cost": np.nan,
                    "dtw_steps": 0,
                    "dtw_cost_per_step": np.nan,
                    "dtw_status": "SKIP",
                    "dtw_error": "missing_rep_arrays",
                }
            )
            continue

        a = np.asarray(au.get(cfg.a_key, np.array([])), dtype="float64").reshape(-1)
        b = np.asarray(av.get(cfg.a_key, np.array([])), dtype="float64").reshape(-1)

        if a.size < int(cfg.min_sim_length) or b.size < int(cfg.min_sim_length):
            rows.append(
                {
                    "src_rep_id": u,
                    "dst_rep_id": v,
                    "dtw_cost": np.nan,
                    "dtw_steps": 0,
                    "dtw_cost_per_step": np.nan,
                    "dtw_status": "SKIP",
                    "dtw_error": "too_short_series",
                }
            )
            continue

        try:
            cost, path = dtw_path_and_cost(a, b, cfg=dtw_cfg, downsample_path_to=int(cfg.downsample_path_to))
            steps = int(path.shape[0])
            cps = float(cost / max(1, steps))

            rows.append(
                {
                    "src_rep_id": u,
                    "dst_rep_id": v,
                    "dtw_cost": float(cost),
                    "dtw_steps": steps,
                    "dtw_cost_per_step": cps,
                    "dtw_status": "OK",
                    "dtw_error": "",
                }
            )

            edge_ids.append((u, v))
            paths.append(np.asarray(path, dtype=np.int64))

        except Exception as e:
            rows.append(
                {
                    "src_rep_id": u,
                    "dst_rep_id": v,
                    "dtw_cost": np.nan,
                    "dtw_steps": 0,
                    "dtw_cost_per_step": np.nan,
                    "dtw_status": "FAIL",
                    "dtw_error": f"{type(e).__name__}: {e}",
                }
            )

    # Ensure expected columns even if no rows were produced.
    cols = [
        "src_rep_id",
        "dst_rep_id",
        "dtw_cost",
        "dtw_steps",
        "dtw_cost_per_step",
        "dtw_status",
        "dtw_error",
    ]
    dtw_edges = pd.DataFrame(rows, columns=cols)
    return dtw_edges, edge_ids, paths


def save_dtw_outputs(
    dtw_edges: "pd.DataFrame",
    edge_ids: List[Tuple[str, str]],
    paths: List[np.ndarray],
    *,
    dtw_edges_csv: str,
    dtw_paths_npz: str,
) -> None:
    """
    Writes:
      - dtw_edges.csv with scalar metrics/status
      - dtw_paths.npz packed tiepoint paths (only for OK edges included in edge_ids/paths)
    """
    _require_pd()

    # Save edges
    dtw_edges.to_csv(dtw_edges_csv, index=False)

    # Pack and save paths
    packed = pack_paths(edge_ids, paths, canonicalize=True)
    save_paths_npz(packed, dtw_paths_npz)
