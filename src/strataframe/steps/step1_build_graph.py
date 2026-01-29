# src/strataframe/steps/step1_build_graph.py
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from strataframe.chronolog.geo import haversine_km, latlon_to_xy_km
from strataframe.chronolog.graph import build_delaunay_edges
from strataframe.chronolog.io import load_wells_gr


try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception:  # pragma: no cover
    cKDTree = None  # type: ignore


@dataclass(frozen=True)
class Step1GraphConfig:
    k_max: int = 12
    r_max_km: float = 8.0
    knn_candidates: int = 24  # candidate neighbors to consider per node


@dataclass(frozen=True)
class Step1GraphPaths:
    out_dir: Path

    @property
    def graph_nodes_csv(self) -> Path:
        return self.out_dir / "graph_nodes.csv"

    @property
    def graph_edges_csv(self) -> Path:
        return self.out_dir / "graph_edges.csv"

    @property
    def diagnostics_json(self) -> Path:
        return self.out_dir / "diagnostics.json"

    @property
    def manifest_json(self) -> Path:
        return self.out_dir / "manifest.json"


def _edge_key(i: int, j: int) -> Tuple[int, int]:
    return (i, j) if i < j else (j, i)


def _build_knn_edges(xy: np.ndarray, *, k: int) -> set[Tuple[int, int]]:
    if cKDTree is None:
        raise RuntimeError("scipy is required for kNN graph. Install with: pip install scipy")
    n = int(xy.shape[0])
    if n <= 1:
        return set()
    k_eff = int(min(max(2, k + 1), n))
    tree = cKDTree(xy)
    _, idx = tree.query(xy, k=k_eff)
    edges: set[Tuple[int, int]] = set()
    for i in range(n):
        for j in idx[i, 1:]:  # skip self
            edges.add(_edge_key(i, int(j)))
    return edges


def _prune_to_k_max(
    edges: Dict[Tuple[int, int], Dict[str, Any]],
    *,
    n_nodes: int,
    k_max: int,
) -> int:
    if k_max <= 0:
        return 0

    adj: Dict[int, set[int]] = {i: set() for i in range(n_nodes)}
    for (i, j) in edges.keys():
        adj[i].add(j)
        adj[j].add(i)

    removed = 0
    while True:
        over = [i for i in range(n_nodes) if len(adj[i]) > k_max]
        if not over:
            break
        removed_this_round = 0
        for i in over:
            if len(adj[i]) <= k_max:
                continue
            neighbors = list(adj[i])
            if not neighbors:
                continue
            # Drop kNN-only edges first; never remove Delaunay edges.
            knn_only = [
                j
                for j in neighbors
                if edges[_edge_key(i, j)].get("knn", False)
                and (not edges[_edge_key(i, j)].get("delaunay", False))
            ]
            if not knn_only:
                continue
            farthest = max(knn_only, key=lambda j: edges[_edge_key(i, j)]["dist_km"])

            key = _edge_key(i, farthest)
            if key in edges:
                del edges[key]
                removed += 1
                removed_this_round += 1
            adj[i].discard(farthest)
            adj[farthest].discard(i)
        if removed_this_round == 0:
            break
    return removed


def run_step1_build_graph(
    *,
    wells_gr_path: Path,
    out_dir: Path,
    cfg: Step1GraphConfig,
    overwrite: bool = False,
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = Step1GraphPaths(out_dir=out_dir)

    if not overwrite and (paths.graph_nodes_csv.exists() or paths.graph_edges_csv.exists()):
        raise FileExistsError(
            f"Step1 outputs already exist under {out_dir}. Use overwrite=true or pick a new output dir."
        )

    wells = load_wells_gr(Path(wells_gr_path), require_latlon=True)
    if wells.empty:
        raise RuntimeError("No wells with valid lat/lon in wells_gr dataset.")

    wells = wells.reset_index(drop=True).copy()
    wells["node_id"] = np.arange(len(wells), dtype="int64")

    lat = wells["lat"].to_numpy(dtype="float64")
    lon = wells["lon"].to_numpy(dtype="float64")
    x_km, y_km = latlon_to_xy_km(lat, lon)
    wells["x_km"] = x_km
    wells["y_km"] = y_km

    xy = np.column_stack([x_km, y_km]).astype("float64", copy=False)

    delaunay_edges = build_delaunay_edges(xy)
    k_candidates = int(max(cfg.knn_candidates, cfg.k_max * 2))
    knn_edges = _build_knn_edges(xy, k=k_candidates)

    edges: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for key in delaunay_edges:
        edges.setdefault(key, {"delaunay": True, "knn": False})

    for key in knn_edges:
        if key in edges:
            edges[key]["knn"] = True
        else:
            edges[key] = {"delaunay": False, "knn": True}

    # Compute distances
    if edges:
        src_idx = np.array([k[0] for k in edges.keys()], dtype="int64")
        dst_idx = np.array([k[1] for k in edges.keys()], dtype="int64")
        dist = haversine_km(lat[src_idx], lon[src_idx], lat[dst_idx], lon[dst_idx])
        for (k, d) in zip(list(edges.keys()), dist.tolist()):
            edges[k]["dist_km"] = float(d)

    # Apply max distance cutoff
    n_cut = 0
    if cfg.r_max_km is not None and np.isfinite(cfg.r_max_km) and cfg.r_max_km > 0:
        for key in list(edges.keys()):
            if float(edges[key]["dist_km"]) > float(cfg.r_max_km):
                del edges[key]
                n_cut += 1

    # Enforce max neighbors
    n_pruned = _prune_to_k_max(edges, n_nodes=int(len(wells)), k_max=int(cfg.k_max))

    # Build edges dataframe
    rows = []
    for (i, j), meta in edges.items():
        edge_type = "both" if (meta.get("delaunay") and meta.get("knn")) else "delaunay" if meta.get("delaunay") else "knn"
        rows.append(
            {
                "src_id": int(i),
                "dst_id": int(j),
                "dist_km": float(meta.get("dist_km", np.nan)),
                "edge_type": edge_type,
            }
        )

    edges_df = pd.DataFrame(rows)
    nodes_df = wells.copy()

    nodes_df.to_csv(paths.graph_nodes_csv, index=False)
    edges_df.to_csv(paths.graph_edges_csv, index=False)

    diag = {
        "counts": {
            "n_nodes": int(len(nodes_df)),
            "n_edges": int(len(edges_df)),
            "n_delaunay": int(sum(1 for v in edges.values() if v.get("delaunay"))),
            "n_knn": int(sum(1 for v in edges.values() if v.get("knn"))),
            "n_both": int(sum(1 for v in edges.values() if v.get("delaunay") and v.get("knn"))),
            "n_cut_dist": int(n_cut),
            "n_pruned_k": int(n_pruned),
        },
        "cfg": asdict(cfg),
        "paths": {
            "graph_nodes_csv": str(paths.graph_nodes_csv),
            "graph_edges_csv": str(paths.graph_edges_csv),
        },
    }
    paths.diagnostics_json.write_text(json.dumps(diag, indent=2))
    paths.manifest_json.write_text(json.dumps({"step": "step1_build_graph", **diag}, indent=2))
    return diag


def _p(path: str) -> Path:
    return Path(path).expanduser().resolve()


def main() -> None:
    ap = argparse.ArgumentParser(description="Step1: build graph = Delaunay + kNN with max neighbors.")
    ap.add_argument("--wells-gr", required=True, help="Path to step0 wells_gr.parquet (file or directory) or csv.")
    ap.add_argument("--out-dir", required=True, help="Output directory for graph artifacts.")
    ap.add_argument("--k-max", type=int, default=12)
    ap.add_argument("--r-max-km", type=float, default=8.0)
    ap.add_argument("--knn-candidates", type=int, default=24)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = Step1GraphConfig(
        k_max=int(args.k_max),
        r_max_km=float(args.r_max_km),
        knn_candidates=int(args.knn_candidates),
    )
    diag = run_step1_build_graph(
        wells_gr_path=_p(args.wells_gr),
        out_dir=_p(args.out_dir),
        cfg=cfg,
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(diag.get("counts", {}), indent=2))


if __name__ == "__main__":
    main()
