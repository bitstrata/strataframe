# src/strataframe/steps/step3_graph_attach_arrays.py
from __future__ import annotations

from typing import Dict

import numpy as np

try:
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover
    nx = None  # type: ignore


def _require_nx() -> None:
    if nx is None:
        raise RuntimeError("networkx is required. Install with: pip install networkx")


def attach_rep_arrays(
    G: "nx.Graph",
    rep_arrays: Dict[str, Dict[str, np.ndarray]],
    *,
    log_key: str = "log_rs",
    depth_key: str = "depth_rs",
    node_log_key: str = "log_rs",
    node_depth_key: str = "depth_rs",
) -> int:
    """
    Adds node arrays for each node in G if present in rep_arrays.
    Returns number of nodes attached.
    """
    _require_nx()
    n_ok = 0
    for n in G.nodes:
        a = rep_arrays.get(str(n))
        if a is None:
            continue
        if log_key in a:
            G.nodes[n][node_log_key] = np.asarray(a[log_key], dtype="float64")
        if depth_key in a:
            G.nodes[n][node_depth_key] = np.asarray(a[depth_key], dtype="float64")
        n_ok += 1
    return n_ok
