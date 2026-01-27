# src/strataframe/pipelines/step3_io_framework.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover
    nx = None  # type: ignore


def _require() -> None:
    if pd is None:
        raise RuntimeError("pandas is required. Install with: pip install pandas")
    if nx is None:
        raise RuntimeError("networkx is required. Install with: pip install networkx")


@dataclass(frozen=True)
class FrameworkCsvConfig:
    node_id_col: str = "rep_id"
    lat_col: str = "lat"
    lon_col: str = "lon"
    bin_id_col: str = "bin_id"


def load_framework_graph(
    *,
    nodes_csv: str | Path,
    edges_csv: str | Path,
    cfg: FrameworkCsvConfig = FrameworkCsvConfig(),
) -> "nx.Graph":
    """
    Reconstruct the pruned framework graph from framework_nodes.csv and framework_edges.csv.
    Preserves all edge columns as edge attributes.
    """
    _require()
    nodes_csv = Path(nodes_csv)
    edges_csv = Path(edges_csv)

    nodes = pd.read_csv(nodes_csv)
    edges = pd.read_csv(edges_csv)

    for c in (cfg.node_id_col, cfg.lat_col, cfg.lon_col):
        if c not in nodes.columns:
            raise ValueError(f"framework_nodes missing required column: {c}")
    if "src_rep_id" not in edges.columns or "dst_rep_id" not in edges.columns:
        raise ValueError("framework_edges must include src_rep_id and dst_rep_id")

    G = nx.Graph()

    # nodes
    for r in nodes.itertuples(index=False):
        rid = str(getattr(r, cfg.node_id_col))
        attrs = {
            "lat": float(getattr(r, cfg.lat_col)),
            "lon": float(getattr(r, cfg.lon_col)),
        }
        if cfg.bin_id_col in nodes.columns:
            v = getattr(r, cfg.bin_id_col)
            attrs["bin_id"] = "" if (v is None or (isinstance(v, float) and not np.isfinite(v))) else str(v)
        G.add_node(rid, **attrs)

    # edges
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
            if isinstance(val, (np.floating, np.integer)):
                val = val.item()
            attrs[col] = val
        G.add_edge(u, v, **attrs)

    return G
