# src/strataframe/correlation/framework.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Set

import numpy as np

try:
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover
    nx = None  # type: ignore


@dataclass(frozen=True)
class FrameworkConfig:
    """
    Framework selection/pruning from DTW edges.

    Similarity convention:
      sim = exp(-dtw_cost_per_step / sim_scale)

    Modes:
      - "mst": maximum spanning forest on sim (usable edges only)
      - "topk": keep topk strongest edges per node (usable edges only)
      - "threshold": keep edges with sim >= sim_threshold (usable edges only)
      - "mst_plus_topk": MST backbone + add topk_extra edges per node
    """

    mode: str = "mst"  # "mst" | "topk" | "threshold" | "mst_plus_topk"

    # top-k modes
    topk: int = 3
    topk_extra: int = 3  # used only by mst_plus_topk (edges per node to add beyond MST)

    # threshold gating
    sim_threshold: float = 0.60  # used by "threshold" mode
    extra_sim_min: float = 0.0   # used by mst_plus_topk to avoid adding weak edges

    # Similarity conversion from dtw_cost_per_step
    sim_scale: float = 0.25


def _require() -> None:
    if nx is None:
        raise RuntimeError("networkx is required. Install with: pip install networkx")


def edge_similarity(cost_per_step: float, *, scale: float) -> float:
    if not np.isfinite(cost_per_step):
        return 0.0
    scale = float(max(scale, 1e-9))
    return float(np.exp(-float(cost_per_step) / scale))


def add_similarity_scores(
    G: "nx.Graph",
    *,
    cost_key: str = "dtw_cost_per_step",
    sim_key: str = "sim",
    scale: float,
    status_key: str = "dtw_status",
    ok_status: str = "OK",
    require_ok_status: bool | None = None,
) -> None:
    """
    Writes ed[sim_key] for each edge.

    If require_ok_status is None:
      - auto-enable gating if ANY edge has status_key present.
    If enabled:
      - edges with dtw_status != OK are assigned sim=0.0
    """
    if require_ok_status is None:
        require_ok_status = any(status_key in ed for _, _, ed in G.edges(data=True))

    for u, v, ed in G.edges(data=True):
        if require_ok_status:
            if str(ed.get(status_key, "")).upper() != str(ok_status).upper():
                ed[sim_key] = 0.0
                continue

        cps_raw = ed.get(cost_key, np.nan)
        try:
            cps = float(cps_raw)
        except Exception:
            cps = float("nan")

        ed[sim_key] = edge_similarity(cps, scale=scale)


def prune_to_framework(
    G: "nx.Graph",
    *,
    cfg: FrameworkConfig,
    sim_key: str = "sim",
    cost_key: str = "dtw_cost_per_step",
    status_key: str = "dtw_status",
    ok_status: str = "OK",
) -> "nx.Graph":
    """
    Returns a new graph containing only the selected framework edges.

    Selection is performed on a usable-edge subgraph:
      * dtw_status == OK (if dtw_status exists on any edge)
      * dtw_cost_per_step finite
      * sim > 0
    """
    _require()

    # Ensure similarity exists
    if any(sim_key not in ed for _, _, ed in G.edges(data=True)):
        add_similarity_scores(G, scale=cfg.sim_scale, sim_key=sim_key, cost_key=cost_key)

    require_ok = any(status_key in ed for _, _, ed in G.edges(data=True))

    def _usable(u: str, v: str, ed: dict) -> bool:
        if require_ok:
            if str(ed.get(status_key, "")).upper() != str(ok_status).upper():
                return False
        try:
            cps = float(ed.get(cost_key, np.nan))
        except Exception:
            return False
        if not np.isfinite(cps):
            return False
        try:
            s = float(ed.get(sim_key, 0.0))
        except Exception:
            s = 0.0
        return np.isfinite(s) and s > 0.0

    # Build usable-edge subgraph (copy attrs)
    Guse = nx.Graph()
    Guse.add_nodes_from(G.nodes(data=True))
    for u, v, ed in G.edges(data=True):
        if _usable(u, v, ed):
            Guse.add_edge(u, v, **dict(ed))

    # Output graph (nodes always carried through)
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))

    mode = str(cfg.mode).lower().strip()

    if mode == "mst":
        T = nx.maximum_spanning_tree(Guse, weight=sim_key)
        H.add_edges_from((u, v, dict(ed)) for u, v, ed in T.edges(data=True))
        return H

    if mode == "threshold":
        thr = float(cfg.sim_threshold)
        for u, v, ed in Guse.edges(data=True):
            if float(ed.get(sim_key, 0.0)) >= thr:
                H.add_edge(u, v, **dict(ed))
        return H

    if mode == "topk":
        keep: Set[Tuple[str, str]] = set()
        k = max(1, int(cfg.topk))
        for n in Guse.nodes:
            nbrs = []
            for m in Guse.neighbors(n):
                s = float(Guse.edges[n, m].get(sim_key, 0.0))
                nbrs.append((s, n, m))
            nbrs.sort(reverse=True, key=lambda t: t[0])
            for s, a, b in nbrs[:k]:
                u, v = (a, b) if a < b else (b, a)
                keep.add((u, v))

        for u, v in keep:
            if Guse.has_edge(u, v):
                H.add_edge(u, v, **dict(Guse.edges[u, v]))
        return H

    if mode == "mst_plus_topk":
        # 1) MST backbone
        T = nx.maximum_spanning_tree(Guse, weight=sim_key)
        H.add_edges_from((u, v, dict(ed)) for u, v, ed in T.edges(data=True))

        # 2) Add extra top-k edges per node (excluding already-kept)
        k_extra = max(0, int(cfg.topk_extra))
        if k_extra <= 0:
            return H

        smin = float(cfg.extra_sim_min)
        kept_edges: Set[Tuple[str, str]] = set()
        for u, v in H.edges:
            a, b = (u, v) if u < v else (v, u)
            kept_edges.add((a, b))

        for n in Guse.nodes:
            # rank neighbors by sim
            nbrs = []
            for m in Guse.neighbors(n):
                a, b = (n, m) if n < m else (m, n)
                if (a, b) in kept_edges:
                    continue
                s = float(Guse.edges[n, m].get(sim_key, 0.0))
                if s < smin:
                    continue
                nbrs.append((s, n, m))

            nbrs.sort(reverse=True, key=lambda t: t[0])
            added = 0
            for s, a, b in nbrs:
                u, v = (a, b) if a < b else (b, a)
                if (u, v) in kept_edges:
                    continue
                if Guse.has_edge(u, v):
                    H.add_edge(u, v, **dict(Guse.edges[u, v]))
                    kept_edges.add((u, v))
                    added += 1
                    if added >= k_extra:
                        break

        return H

    raise ValueError(f"Unknown FrameworkConfig.mode: {cfg.mode}")
