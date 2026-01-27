# src/strataframe/pipelines/step3e_rgt_from_paths.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

try:
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover
    nx = None  # type: ignore

from strataframe.correlation.paths_npz import load_paths_npz
from strataframe.rgt.rgt import RgtConfig, Anchor as RgtAnchor, solve_rgt_shifts


def _require_nx() -> None:
    if nx is None:
        raise RuntimeError("networkx is required. Install with: pip install networkx")


@dataclass(frozen=True)
class InjectPathsConfig:
    path_key: str = "dtw_path"
    require_paths: bool = True


def inject_paths_onto_graph(
    G: "nx.Graph",
    paths_npz: str,
    *,
    cfg: InjectPathsConfig = InjectPathsConfig(),
) -> int:
    """
    For each edge (u,v) in G, loads a DTW path from packed npz.
    Writes ed[cfg.path_key] = (K,2) int64 path in the requested u->v orientation.

    Returns number of edges with paths injected.
    """
    _require_nx()
    packed = load_paths_npz(paths_npz)

    # Build index once: (src,dst) -> edge_idx
    E = int(packed.src_ids.size)
    index: Dict[tuple[str, str], int] = {}
    for e in range(E):
        index[(str(packed.src_ids[e]), str(packed.dst_ids[e]))] = e

    n_ok = 0
    for u0, v0, ed in G.edges(data=True):
        u = str(u0)
        v = str(v0)

        e = index.get((u, v))
        flip = False
        if e is None:
            e = index.get((v, u))
            flip = e is not None

        if e is None:
            if cfg.require_paths:
                ed[cfg.path_key] = None
            continue

        k0, k1 = int(packed.ptr[e]), int(packed.ptr[e + 1])
        ii = packed.ii[k0:k1]
        jj = packed.jj[k0:k1]

        if not flip:
            path = np.stack([ii, jj], axis=1).astype(np.int64)
        else:
            # stored opposite direction; swap to satisfy requested u->v
            path = np.stack([jj, ii], axis=1).astype(np.int64)

        ed[cfg.path_key] = path
        n_ok += 1

    return n_ok


def default_component_anchors(
    G: "nx.Graph",
    *,
    sample_idx: int,
    target_shift: float = 0.0,
    weight: float = 1.0,
) -> List[RgtAnchor]:
    _require_nx()
    anchors: List[RgtAnchor] = []

    for comp in nx.connected_components(G):
        nodes = list(comp)
        if not nodes:
            continue
        best = max(nodes, key=lambda n: int(G.degree[n]))
        anchors.append(
            RgtAnchor(
                node_id=str(best),
                sample_idx=int(sample_idx),
                target_shift=float(target_shift),
                weight=float(weight),
            )
        )
    return anchors


def solve_rgt_from_framework(
    G: "nx.Graph",
    *,
    rgt_cfg: RgtConfig,
    paths_npz: str,
    z_key: str = "depth_rs",
    path_key: str = "dtw_path",
    anchors: Optional[List[RgtAnchor]] = None,
    inject_cfg: InjectPathsConfig = InjectPathsConfig(),
) -> Dict[str, np.ndarray]:
    _require_nx()

    inject_paths_onto_graph(
        G,
        paths_npz,
        cfg=InjectPathsConfig(path_key=path_key, require_paths=bool(inject_cfg.require_paths)),
    )

    has_any = any(ed.get(path_key, None) is not None for _, _, ed in G.edges(data=True))
    if not has_any:
        raise RuntimeError("No DTW paths found on any framework edges after injection. Check paths_npz and edge IDs.")

    return solve_rgt_shifts(G, cfg=rgt_cfg, z_key=z_key, path_key=path_key, anchors=anchors)
