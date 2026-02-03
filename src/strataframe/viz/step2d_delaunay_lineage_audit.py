# src/strataframe/viz/step2d_delaunay_lineage_audit.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _load_edges(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["src_id"] = pd.to_numeric(df["src_id"], errors="coerce").astype("Int64")
    df["dst_id"] = pd.to_numeric(df["dst_id"], errors="coerce").astype("Int64")
    df = df[df["src_id"].notna() & df["dst_id"].notna()].copy()
    df["src_id"] = df["src_id"].astype(int)
    df["dst_id"] = df["dst_id"].astype(int)
    df["edge_key"] = df.apply(
        lambda r: (r.src_id, r.dst_id) if r.src_id < r.dst_id else (r.dst_id, r.src_id), axis=1
    )
    return df


def _load_nodes(nodes_csv: Path) -> pd.DataFrame:
    nodes = pd.read_csv(nodes_csv)
    nodes["node_id"] = pd.to_numeric(nodes["node_id"], errors="coerce").astype("Int64")
    nodes["x_km"] = pd.to_numeric(nodes.get("x_km"), errors="coerce")
    nodes["y_km"] = pd.to_numeric(nodes.get("y_km"), errors="coerce")
    nodes = nodes[nodes["node_id"].notna() & nodes["x_km"].notna() & nodes["y_km"].notna()].copy()
    nodes["node_id"] = nodes["node_id"].astype(int)
    return nodes


def _edge_map(ax, edges: pd.DataFrame, nodes: pd.DataFrame, color: str, alpha: float, lw: float) -> None:
    idx = {int(r.node_id): (float(r.x_km), float(r.y_km)) for r in nodes.itertuples(index=False)}
    for r in edges.itertuples(index=False):
        s = idx.get(int(r.src_id))
        d = idx.get(int(r.dst_id))
        if s is None or d is None:
            continue
        ax.plot([s[0], d[0]], [s[1], d[1]], color=color, alpha=alpha, linewidth=lw)


def main() -> None:
    ap = argparse.ArgumentParser(description="Delaunay lineage audit (Step1 → Step2d → Step3).")
    ap.add_argument("--nodes-csv", required=True)
    ap.add_argument("--delaunay-edges-csv", required=True, help="Canonical Delaunay list (e.g., delaunay_cut_edges.csv)")
    ap.add_argument("--graph-edges-csv", required=True)
    ap.add_argument("--dtw-edges-csv", required=True)
    ap.add_argument("--dtw-paths-jsonl", default="", help="Optional: DTW paths jsonl for tiepoint presence")
    ap.add_argument("--exclude-nodes-csv", default="", help="Optional: excluded nodes to explain missing edges")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--map", action="store_true", help="Write a lineage map PNG")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--max-edges", type=int, default=250000)

    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes = _load_nodes(Path(args.nodes_csv))
    delaunay = _load_edges(Path(args.delaunay_edges_csv))
    graph = _load_edges(Path(args.graph_edges_csv))
    dtw = _load_edges(Path(args.dtw_edges_csv))

    if int(args.max_edges) > 0 and delaunay.shape[0] > int(args.max_edges):
        delaunay = delaunay.sample(n=int(args.max_edges), random_state=42).reset_index(drop=True)

    delaunay_set = set(delaunay.edge_key)
    graph_set = set(graph.edge_key)
    dtw_set = set(dtw.edge_key)

    missing_graph = sorted(delaunay_set - graph_set)
    missing_dtw = sorted(delaunay_set - dtw_set)

    pd.DataFrame(missing_graph, columns=["src_id", "dst_id"]).to_csv(out_dir / "delaunay_missing_in_graph.csv", index=False)
    pd.DataFrame(missing_dtw, columns=["src_id", "dst_id"]).to_csv(out_dir / "delaunay_missing_in_dtw.csv", index=False)

    # classify missing dtw edges by excluded nodes
    if str(args.exclude_nodes_csv).strip():
        ex = pd.read_csv(args.exclude_nodes_csv)
        ex_ids = set(pd.to_numeric(ex.get("node_id"), errors="coerce").dropna().astype(int).tolist())
        missing = pd.DataFrame(missing_dtw, columns=["src_id", "dst_id"])
        missing["excluded_node"] = missing["src_id"].isin(ex_ids) | missing["dst_id"].isin(ex_ids)
        missing[missing["excluded_node"]].to_csv(out_dir / "delaunay_missing_excluded.csv", index=False)
        missing[~missing["excluded_node"]].to_csv(out_dir / "delaunay_missing_other.csv", index=False)

    # status breakdown for dtw Delaunay edges
    if "status" in dtw.columns:
        delaunay_dtw = dtw[dtw.edge_key.isin(delaunay_set)].copy()
        delaunay_dtw["status"].value_counts().to_csv(out_dir / "delaunay_dtw_status_counts.csv")

    # Map (optional)
    if bool(args.map):
        import matplotlib.pyplot as plt

        delaunay_df = delaunay[["src_id", "dst_id"]].copy()
        in_dtw = dtw[dtw.edge_key.isin(delaunay_set)][["src_id", "dst_id"]].copy()

        fig, ax = plt.subplots(figsize=(7.6, 6.2), dpi=int(args.dpi))
        _edge_map(ax, delaunay_df, nodes, color="#cccccc", alpha=0.15, lw=0.5)
        _edge_map(ax, in_dtw, nodes, color="#1b9e77", alpha=0.25, lw=0.7)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
        ax.set_title("Delaunay lineage: grey=all, green=DTW‑processed")
        fig.tight_layout()
        fig.savefig(out_dir / "delaunay_lineage_map.png")
        plt.close(fig)

    # summary
    summary = {
        "delaunay_total": int(len(delaunay_set)),
        "in_graph": int(len(delaunay_set & graph_set)),
        "in_dtw": int(len(delaunay_set & dtw_set)),
        "missing_in_graph": int(len(missing_graph)),
        "missing_in_dtw": int(len(missing_dtw)),
    }
    pd.Series(summary).to_csv(out_dir / "delaunay_lineage_summary.csv")


if __name__ == "__main__":
    main()
