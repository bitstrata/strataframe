# src/strataframe/pipelines/step3d_framework_from_edges.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover
    nx = None  # type: ignore

import numpy as np

from strataframe.correlation.framework import FrameworkConfig, add_similarity_scores, prune_to_framework


def _require() -> None:
    if pd is None:
        raise RuntimeError("pandas is required. Install with: pip install pandas")
    if nx is None:
        raise RuntimeError("networkx is required. Install with: pip install networkx")


@dataclass(frozen=True)
class BuildFrameworkGraphConfig:
    node_id_col: str = "rep_id"
    lat_col: str = "lat"
    lon_col: str = "lon"
    bin_id_col: str = "bin_id"


def build_graph_from_edges(
    reps: "pd.DataFrame",
    edges: "pd.DataFrame",
    *,
    cfg: BuildFrameworkGraphConfig = BuildFrameworkGraphConfig(),
) -> "nx.Graph":
    """
    reps:
      rep_id, lat, lon, bin_id (optional)

    edges:
      src_rep_id, dst_rep_id plus any edge attributes
    """
    _require()

    for c in (cfg.node_id_col, cfg.lat_col, cfg.lon_col):
        if c not in reps.columns:
            raise ValueError(f"reps missing required column: {c}")
    if "src_rep_id" not in edges.columns or "dst_rep_id" not in edges.columns:
        raise ValueError("edges must include src_rep_id and dst_rep_id")

    G = nx.Graph()

    # nodes
    for r in reps.itertuples(index=False):
        rid = str(getattr(r, cfg.node_id_col))
        attrs = {
            "lat": float(getattr(r, cfg.lat_col)),
            "lon": float(getattr(r, cfg.lon_col)),
        }
        if cfg.bin_id_col in reps.columns:
            attrs["bin_id"] = str(getattr(r, cfg.bin_id_col))
        G.add_node(rid, **attrs)

    # edges with attrs
    drop_cols = {"src_rep_id", "dst_rep_id"}
    for e in edges.itertuples(index=False):
        u = str(getattr(e, "src_rep_id"))
        v = str(getattr(e, "dst_rep_id"))
        if u == v:
            continue
        attrs = {}
        for col in edges.columns:
            if col in drop_cols:
                continue
            val = getattr(e, col)
            # normalize numpy scalars
            if isinstance(val, (np.floating, np.integer)):
                val = val.item()
            attrs[col] = val
        G.add_edge(u, v, **attrs)

    return G


def prune_framework_graph(
    G: "nx.Graph",
    *,
    fw_cfg: FrameworkConfig,
    sim_scale: Optional[float] = None,
    cost_key: str = "dtw_cost_per_step",
    sim_key: str = "sim",
) -> "nx.Graph":
    """
    Adds similarity (optionally using auto median sim_scale if not provided), then prunes.
    """
    _require()

    if sim_scale is None:
        # robust default: median CPS over OK edges
        cps = []
        for _, _, ed in G.edges(data=True):
            try:
                if str(ed.get("dtw_status", "OK")).upper() != "OK":
                    continue
                v = float(ed.get(cost_key, np.nan))
                if np.isfinite(v):
                    cps.append(v)
            except Exception:
                continue
        if cps:
            sim_scale = float(np.median(np.asarray(cps, dtype="float64")))
        else:
            sim_scale = float(fw_cfg.sim_scale)

    add_similarity_scores(G, cost_key=cost_key, sim_key=sim_key, scale=float(sim_scale))
    return prune_to_framework(G, cfg=fw_cfg, sim_key=sim_key, cost_key=cost_key)
