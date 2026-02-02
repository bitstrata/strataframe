# src/strataframe/viz/step3_rgt_panel.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    # When run as a module
    from strataframe.viz.step3b_panel import plot_step3b_rep_arrays_panel
    from strataframe.viz.step3_common import load_edges_csv
except Exception:  # pragma: no cover
    # Fallback for direct script execution
    from step3b_panel import plot_step3b_rep_arrays_panel  # type: ignore
    from step3_common import load_edges_csv  # type: ignore

try:
    import networkx as nx  # type: ignore
except Exception:
    nx = None  # type: ignore

from strataframe.spatial.geodesy import haversine_km_vec


def _evenly_subsample(order: List[str], k: int) -> List[str]:
    if k <= 0 or len(order) <= k:
        return order
    idx = np.linspace(0, len(order) - 1, int(k), dtype=int)
    seen: set[str] = set()
    out: List[str] = []
    for i in idx.tolist():
        rid = str(order[int(i)])
        if rid not in seen:
            out.append(rid)
            seen.add(rid)
    return out


def _load_gr_vectors(path: Path) -> Tuple[Dict[str, Dict[str, np.ndarray]], int]:
    z = np.load(Path(path), allow_pickle=False)
    node_id = np.asarray(z["node_id"], dtype="int64")
    z_top = np.asarray(z["z_top"], dtype="float64")
    z_base = np.asarray(z["z_base"], dtype="float64")
    x_norm = np.asarray(z["x_norm"], dtype="float64")
    if x_norm.ndim != 2:
        raise RuntimeError("gr_vectors.npz x_norm must be 2D (n_wells, n_samples).")
    n_samples = int(x_norm.shape[1])

    rep_arrays: Dict[str, Dict[str, np.ndarray]] = {}
    for i in range(int(node_id.size)):
        rid = str(int(node_id[i]))
        z0 = float(z_top[i])
        z1 = float(z_base[i])
        if not (np.isfinite(z0) and np.isfinite(z1)) or z1 <= z0:
            continue
        depth = np.linspace(z0, z1, n_samples).astype("float64")
        rep_arrays[rid] = {
            "depth_rs": depth,
            "log_rs": np.asarray(x_norm[i], dtype="float64"),
        }
    if not rep_arrays:
        raise RuntimeError("No valid wells found in gr_vectors cache.")
    return rep_arrays, n_samples


def _load_rgt_shifts(path: Path) -> Dict[str, np.ndarray]:
    z = np.load(Path(path), allow_pickle=False)
    node_id = np.asarray(z["node_id"], dtype="int64")
    shifts = np.asarray(z["shifts"], dtype="float64")
    if shifts.ndim != 2:
        raise RuntimeError("rgt_shifts_resampled.npz shifts must be 2D (n_wells, n_samples).")
    out: Dict[str, np.ndarray] = {}
    for i in range(int(node_id.size)):
        out[str(int(node_id[i]))] = shifts[i]
    if not out:
        raise RuntimeError("No shifts loaded from rgt_shifts_resampled.npz.")
    return out


def _nearest_rep_id(reps: pd.DataFrame, *, lon0: float, lat0: float) -> str:
    lat = reps["lat"].to_numpy(dtype="float64")
    lon = reps["lon"].to_numpy(dtype="float64")
    d = haversine_km_vec(lat, lon, float(lat0), float(lon0))
    d[~np.isfinite(d)] = np.inf
    i = int(np.argmin(d))
    return str(reps.iloc[i]["node_id"])


def _build_graph(reps: pd.DataFrame, edges: pd.DataFrame) -> "nx.Graph":
    if nx is None:
        raise RuntimeError("networkx is required for path ordering. pip install networkx")
    G = nx.Graph()
    reps2 = reps.copy()
    reps2["node_id"] = reps2["node_id"].astype(str)
    for r in reps2.itertuples(index=False):
        rid = str(getattr(r, "node_id"))
        lon = float(getattr(r, "lon"))
        lat = float(getattr(r, "lat"))
        if np.isfinite(lon) and np.isfinite(lat):
            G.add_node(rid, lon=lon, lat=lat)

    for e in edges.itertuples(index=False):
        u = str(getattr(e, "u"))
        v = str(getattr(e, "v"))
        if u == v or u not in G.nodes or v not in G.nodes:
            continue
        lu = float(G.nodes[u]["lon"])
        la = float(G.nodes[u]["lat"])
        lv = float(G.nodes[v]["lon"])
        lb = float(G.nodes[v]["lat"])
        w = float(haversine_km_vec(np.array([la]), np.array([lu]), float(lb), float(lv))[0])
        if np.isfinite(w):
            G.add_edge(u, v, w_km=w)
    return G


def _choose_path(
    reps: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    max_wells: int,
    seed: int,
    min_wells: int = 20,
    max_tries: int = 200,
) -> List[str]:
    reps2 = reps.dropna(subset=["lon", "lat"]).copy()
    reps2["node_id"] = reps2["node_id"].astype(str)
    if reps2.empty:
        raise RuntimeError("No valid lon/lat in nodes CSV for ordering.")

    if nx is None or edges.empty:
        order = reps2.sort_values("lon")["node_id"].astype(str).to_list()
        return _evenly_subsample(order, int(max_wells))

    G = _build_graph(reps2, edges)

    min_lon = float(reps2["lon"].min())
    max_lon = float(reps2["lon"].max())
    min_lat = float(reps2["lat"].min())
    max_lat = float(reps2["lat"].max())

    rng = np.random.default_rng(int(seed))
    for _ in range(int(max_tries)):
        orient = "EW" if float(rng.random()) < 0.5 else "NS"
        if orient == "EW":
            latA = float(rng.uniform(min_lat, max_lat))
            lonA = min_lon
            latB = float(rng.uniform(min_lat, max_lat))
            lonB = max_lon
        else:
            lonA = float(rng.uniform(min_lon, max_lon))
            latA = min_lat
            lonB = float(rng.uniform(min_lon, max_lon))
            latB = max_lat

        s = _nearest_rep_id(reps2, lon0=lonA, lat0=latA)
        t = _nearest_rep_id(reps2, lon0=lonB, lat0=latB)
        if s == t or s not in G.nodes or t not in G.nodes:
            continue
        try:
            path = nx.shortest_path(G, s, t, weight="w_km")
        except Exception:
            continue
        if len(path) < int(min_wells):
            continue
        return _evenly_subsample([str(p) for p in path], int(max_wells))

    # fallback: simple lon ordering
    order = reps2.sort_values("lon")["node_id"].astype(str).to_list()
    return _evenly_subsample(order, int(max_wells))


def main() -> None:
    ap = argparse.ArgumentParser(description="Chronostrat panel (RGT-aligned) from Step3 shifts.")
    ap.add_argument("--nodes-csv", required=True, help="graph_nodes.csv")
    ap.add_argument("--edges-csv", required=True, help="graph_edges.csv")
    ap.add_argument("--gr-vectors-npz", required=True, help="gr_vectors.npz (Step2 cache)")
    ap.add_argument("--rgt-shifts-npz", required=True, help="rgt_shifts_resampled.npz (Step3 output)")
    ap.add_argument("--out", required=True, help="Output image path (.png or .svg)")
    ap.add_argument("--max-wells", type=int, default=60)
    ap.add_argument("--min-wells", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    nodes = pd.read_csv(Path(args.nodes_csv))
    if "node_id" not in nodes.columns:
        raise ValueError("nodes_csv must include node_id.")
    nodes["node_id"] = pd.to_numeric(nodes["node_id"], errors="coerce").astype("Int64")
    nodes = nodes[nodes["node_id"].notna()].copy()
    nodes["node_id"] = nodes["node_id"].astype(str)

    # Normalize lon/lat
    for c in ("lon", "lat"):
        if c not in nodes.columns:
            raise ValueError(f"nodes_csv missing required column '{c}'.")
        nodes[c] = pd.to_numeric(nodes[c], errors="coerce")

    edges = load_edges_csv(Path(args.edges_csv))

    rep_arrays, _ = _load_gr_vectors(Path(args.gr_vectors_npz))
    shifts = _load_rgt_shifts(Path(args.rgt_shifts_npz))

    order = _choose_path(
        reps=nodes,
        edges=edges,
        max_wells=int(args.max_wells),
        seed=int(args.seed),
        min_wells=int(args.min_wells),
    )

    # Filter to wells that exist in cache + shifts
    order = [rid for rid in order if rid in rep_arrays and rid in shifts]
    if len(order) < 4:
        raise RuntimeError("Too few wells with both GR vectors and shifts. Check inputs.")

    plot_step3b_rep_arrays_panel(
        rep_arrays=rep_arrays,
        rep_ids=order,
        out_png=str(args.out),
        max_wells=int(args.max_wells),
        shifts=shifts,
        title="Step 3 — Chronostrat panel (RGT-aligned GR logs)",
        xlabel="Wells ordered along a graph path",
        ylabel="Relative geologic time (RGT; depth + shift)",
        gap_w=1.0,
        show_correlations=True,
        correlation_n=36,
        correlation_color="#2E6FDB",
        correlation_alpha=0.35,
        correlation_lw=0.6,
        map_nodes=nodes,
        map_title="Kansas path A–A′",
    )


if __name__ == "__main__":
    main()
