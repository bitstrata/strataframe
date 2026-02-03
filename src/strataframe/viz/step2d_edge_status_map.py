# src/strataframe/viz/step2d_edge_status_map.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _status_color(status: str) -> str:
    s = str(status).lower()
    if s == "edge":
        return "#4C78A8"
    if s == "ok":
        return "#1b9e77"
    if s == "end_mismatch":
        return "#7570b3"
    if s == "dtw_fail":
        return "#d95f02"
    if s == "no_overlap":
        return "#e7298a"
    if s == "overlap_too_small":
        return "#66a61e"
    if s == "cache_miss":
        return "#a6761d"
    if s == "read_fail":
        return "#e6ab02"
    if s == "missing_node":
        return "#666666"
    return "#999999"


def _load_nodes(nodes_csv: Path) -> pd.DataFrame:
    nodes = pd.read_csv(nodes_csv)
    nodes["node_id"] = pd.to_numeric(nodes["node_id"], errors="coerce").astype("Int64")
    nodes["x_km"] = pd.to_numeric(nodes.get("x_km"), errors="coerce")
    nodes["y_km"] = pd.to_numeric(nodes.get("y_km"), errors="coerce")
    nodes = nodes[nodes["node_id"].notna() & nodes["x_km"].notna() & nodes["y_km"].notna()].copy()
    nodes["node_id"] = nodes["node_id"].astype(int)
    return nodes


def _load_edges(edges_csv: Path) -> pd.DataFrame:
    edges = pd.read_csv(edges_csv)
    edges["src_id"] = pd.to_numeric(edges["src_id"], errors="coerce").astype("Int64")
    edges["dst_id"] = pd.to_numeric(edges["dst_id"], errors="coerce").astype("Int64")
    edges = edges[edges["src_id"].notna() & edges["dst_id"].notna()].copy()
    edges["src_id"] = edges["src_id"].astype(int)
    edges["dst_id"] = edges["dst_id"].astype(int)
    return edges


def main() -> None:
    ap = argparse.ArgumentParser(description="Step2d: map of edge statuses.")
    ap.add_argument("--nodes-csv", required=True)
    ap.add_argument("--edges-csv", required=True)
    ap.add_argument("--graph-edges-csv", default="", help="Optional graph_edges.csv to filter by edge_type")
    ap.add_argument("--edge-type", default="", help="Filter edges by edge_type (e.g., delaunay, knn, both)")
    ap.add_argument(
        "--edge-list-csv",
        default="",
        help="Optional explicit edge list (csv with src_id,dst_id) to filter, e.g. delaunay_cut_edges.csv",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-edges", type=int, default=250000)
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--lw", type=float, default=0.6)
    ap.add_argument("--figsize", type=str, default="7.5,6.0")
    ap.add_argument("--dpi", type=int, default=200)

    args = ap.parse_args()
    nodes = _load_nodes(Path(args.nodes_csv))
    edges = _load_edges(Path(args.edges_csv))
    if "status" not in edges.columns:
        edges["status"] = "edge"
    if str(args.edge_list_csv).strip():
        base = _load_edges(Path(args.edge_list_csv))
        base["edge_key"] = base.apply(
            lambda r: (r.src_id, r.dst_id) if r.src_id < r.dst_id else (r.dst_id, r.src_id),
            axis=1,
        )
        edges["edge_key"] = edges.apply(
            lambda r: (r.src_id, r.dst_id) if r.src_id < r.dst_id else (r.dst_id, r.src_id),
            axis=1,
        )
        edges = edges[edges["edge_key"].isin(set(base["edge_key"]))].copy()
    elif str(args.graph_edges_csv).strip():
        g = pd.read_csv(args.graph_edges_csv)
        g["src_id"] = pd.to_numeric(g["src_id"], errors="coerce").astype("Int64")
        g["dst_id"] = pd.to_numeric(g["dst_id"], errors="coerce").astype("Int64")
        g = g[g["src_id"].notna() & g["dst_id"].notna()].copy()
        g["src_id"] = g["src_id"].astype(int)
        g["dst_id"] = g["dst_id"].astype(int)
        if str(args.edge_type).strip():
            et = str(args.edge_type).lower()
            g = g[g["edge_type"].astype(str).str.lower() == et].copy()
        g["edge_key"] = g.apply(lambda r: (r.src_id, r.dst_id) if r.src_id < r.dst_id else (r.dst_id, r.src_id), axis=1)
        edges["edge_key"] = edges.apply(lambda r: (r.src_id, r.dst_id) if r.src_id < r.dst_id else (r.dst_id, r.src_id), axis=1)
        edges = edges[edges["edge_key"].isin(set(g["edge_key"]))].copy()

    if int(args.max_edges) > 0 and edges.shape[0] > int(args.max_edges):
        edges = edges.sample(n=int(args.max_edges), random_state=42).reset_index(drop=True)

    idx: Dict[int, Tuple[float, float]] = {
        int(r.node_id): (float(r.x_km), float(r.y_km)) for r in nodes.itertuples(index=False)
    }

    edges["sx"] = edges["src_id"].map(lambda i: idx.get(int(i), (np.nan, np.nan))[0])
    edges["sy"] = edges["src_id"].map(lambda i: idx.get(int(i), (np.nan, np.nan))[1])
    edges["dx"] = edges["dst_id"].map(lambda i: idx.get(int(i), (np.nan, np.nan))[0])
    edges["dy"] = edges["dst_id"].map(lambda i: idx.get(int(i), (np.nan, np.nan))[1])

    edges = edges[np.isfinite(edges["sx"]) & np.isfinite(edges["sy"]) & np.isfinite(edges["dx"]) & np.isfinite(edges["dy"])].copy()

    import matplotlib.pyplot as plt

    fig_w, fig_h = [float(x.strip()) for x in str(args.figsize).split(",")]
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=int(args.dpi))

    # draw edges by status
    for status, g in edges.groupby("status"):
        color = _status_color(status)
        for r in g.itertuples(index=False):
            ax.plot([r.sx, r.dx], [r.sy, r.dy], color=color, alpha=float(args.alpha), linewidth=float(args.lw))

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_title("Step2d Edge Status Map")

    # legend
    legend_status = ["ok", "end_mismatch", "dtw_fail", "no_overlap", "overlap_too_small", "cache_miss", "read_fail"]
    handles = []
    labels = []
    for s in legend_status:
        if s in edges["status"].unique():
            h = ax.plot([], [], color=_status_color(s), linewidth=2, label=s)[0]
            handles.append(h)
            labels.append(s)
    if handles:
        ax.legend(handles=handles, labels=labels, loc="upper right", frameon=False)

    fig.tight_layout()
    fig.savefig(Path(args.out))
    plt.close(fig)


if __name__ == "__main__":
    main()
