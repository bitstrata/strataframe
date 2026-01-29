# src/strataframe/viz/step1_graph_compare.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from strataframe.chronolog.geo import haversine_km, latlon_to_xy_km
from strataframe.chronolog.graph import build_delaunay_edges


def _load_nodes(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    lat_col = cols.get("lat")
    lon_col = cols.get("lon")
    if lat_col is None or lon_col is None:
        raise ValueError(f"graph_nodes.csv missing lat/lon columns. Found: {list(df.columns)}")
    if "node_id" not in df.columns:
        df["node_id"] = np.arange(len(df), dtype="int64")
    df["node_id"] = pd.to_numeric(df["node_id"], errors="coerce").astype("Int64")
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df[df[lat_col].notna() & df[lon_col].notna() & df["node_id"].notna()].copy()
    df.rename(columns={lat_col: "lat", lon_col: "lon"}, inplace=True)
    return df


def _edge_key(i: int, j: int) -> Tuple[int, int]:
    return (i, j) if i < j else (j, i)


def _edges_to_df(
    edges_idx: Iterable[Tuple[int, int]],
    *,
    node_ids: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for i, j in edges_idx:
        src_id = int(node_ids[int(i)])
        dst_id = int(node_ids[int(j)])
        dist = float(haversine_km(lat[int(i)], lon[int(i)], lat[int(j)], lon[int(j)]))
        rows.append({"src_id": src_id, "dst_id": dst_id, "dist_km": dist})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["src_id"] = pd.to_numeric(df["src_id"], errors="coerce").astype("Int64")
        df["dst_id"] = pd.to_numeric(df["dst_id"], errors="coerce").astype("Int64")
    return df


def _edges_set(df: pd.DataFrame) -> set[Tuple[int, int]]:
    return {_edge_key(int(r.src_id), int(r.dst_id)) for r in df.itertuples(index=False)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare raw Delaunay edges to final graph edges.")
    ap.add_argument("--nodes-csv", required=True, help="graph_nodes.csv")
    ap.add_argument("--edges-csv", required=True, help="graph_edges.csv")
    ap.add_argument("--out-dir", required=True, help="Output directory for comparison tables")
    ap.add_argument("--r-max-km", type=float, default=None, help="Optional distance cutoff for Delaunay comparison")
    args = ap.parse_args()

    nodes = _load_nodes(Path(args.nodes_csv))
    nodes = nodes.sort_values("node_id").reset_index(drop=True)

    lat = nodes["lat"].to_numpy(dtype="float64")
    lon = nodes["lon"].to_numpy(dtype="float64")
    node_ids = nodes["node_id"].to_numpy(dtype="int64")

    x_km, y_km = latlon_to_xy_km(lat, lon)
    xy = np.column_stack([x_km, y_km]).astype("float64", copy=False)

    raw_edges = build_delaunay_edges(xy)
    raw_df = _edges_to_df(raw_edges, node_ids=node_ids, lat=lat, lon=lon)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "delaunay_raw_edges.csv"
    raw_df.to_csv(raw_path, index=False)

    if args.r_max_km is not None and np.isfinite(args.r_max_km):
        cut_df = raw_df[raw_df["dist_km"] <= float(args.r_max_km)].copy()
    else:
        cut_df = raw_df.copy()
    cut_path = out_dir / "delaunay_cut_edges.csv"
    cut_df.to_csv(cut_path, index=False)

    edges = pd.read_csv(Path(args.edges_csv))
    graph_d = edges[edges["edge_type"].astype(str).isin(["delaunay", "both"])].copy()
    graph_d["src_id"] = pd.to_numeric(graph_d["src_id"], errors="coerce").astype("Int64")
    graph_d["dst_id"] = pd.to_numeric(graph_d["dst_id"], errors="coerce").astype("Int64")
    graph_path = out_dir / "graph_delaunay_edges.csv"
    graph_d.to_csv(graph_path, index=False)

    cut_set = _edges_set(cut_df)
    graph_set = _edges_set(graph_d)
    missing = cut_set - graph_set
    extra = graph_set - cut_set

    miss_df = pd.DataFrame([{"src_id": i, "dst_id": j} for (i, j) in sorted(missing)])
    extra_df = pd.DataFrame([{"src_id": i, "dst_id": j} for (i, j) in sorted(extra)])
    miss_path = out_dir / "delaunay_missing_in_graph.csv"
    extra_path = out_dir / "delaunay_extra_in_graph.csv"
    miss_df.to_csv(miss_path, index=False)
    extra_df.to_csv(extra_path, index=False)

    summary = {
        "raw_delaunay": int(len(raw_df)),
        "cut_delaunay": int(len(cut_df)),
        "graph_delaunay": int(len(graph_d)),
        "missing_in_graph": int(len(miss_df)),
        "extra_in_graph": int(len(extra_df)),
        "r_max_km": None if args.r_max_km is None else float(args.r_max_km),
        "paths": {
            "delaunay_raw_edges": str(raw_path),
            "delaunay_cut_edges": str(cut_path),
            "graph_delaunay_edges": str(graph_path),
            "missing_in_graph": str(miss_path),
            "extra_in_graph": str(extra_path),
        },
    }
    (out_dir / "compare_summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
