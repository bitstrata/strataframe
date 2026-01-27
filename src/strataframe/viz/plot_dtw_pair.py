# src/strataframe/viz/plot_graph.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover
    nx = None  # type: ignore

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None  # type: ignore

from strataframe.io.csv import read_csv_rows


def _require() -> None:
    if nx is None:
        raise RuntimeError("networkx is required. Install with: pip install networkx")
    if plt is None:
        raise RuntimeError("matplotlib is required. Install with: pip install matplotlib")


def _as_float(s: object) -> float:
    try:
        v = float(str(s).strip())
        return v if np.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def _project_lonlat(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Simple lon/lat projection to a local Cartesian-ish space for plotting.
    """
    lat0 = float(np.nanmean(lats)) if np.isfinite(np.nanmean(lats)) else 0.0
    deg_to_km = 111.32
    x = (lons * np.cos(np.deg2rad(lat0))) * deg_to_km
    y = lats * deg_to_km
    return np.vstack([x, y]).T


def _load_graph(nodes_csv: Path, edges_csv: Path) -> "nx.Graph":
    nodes = read_csv_rows(nodes_csv)
    edges = read_csv_rows(edges_csv)

    if not nodes:
        raise ValueError(f"No rows in nodes csv: {nodes_csv}")
    if not edges:
        raise ValueError(f"No rows in edges csv: {edges_csv}")

    # Expect "rep_id" for nodes; allow "well_id" fallback
    def _node_id(r: Dict[str, str]) -> str:
        return (r.get("rep_id") or r.get("well_id") or "").strip()

    G = nx.Graph()

    for r in nodes:
        nid = _node_id(r)
        if not nid:
            continue

        lat = _as_float(r.get("lat", ""))
        lon = _as_float(r.get("lon", ""))

        G.add_node(
            nid,
            **{
                "rep_id": nid,
                "lat": lat,
                "lon": lon,
                "h3_cell": (r.get("h3_cell") or "").strip(),
                "h3_res": (r.get("h3_res") or "").strip(),
                "score": (r.get("score") or "").strip(),
                "las_path": (r.get("las_path") or "").strip(),
                "curve_used": (r.get("curve_used") or "").strip(),
            },
        )

    for r in edges:
        a = (r.get("src_rep_id") or r.get("src") or r.get("u") or "").strip()
        b = (r.get("dst_rep_id") or r.get("dst") or r.get("v") or "").strip()
        if not a or not b:
            continue
        if not (G.has_node(a) and G.has_node(b)):
            continue

        # Common edge keys in your pipeline
        dist_km = _as_float(r.get("dist_km", ""))
        dtw_cps = _as_float(r.get("dtw_cost_per_step", ""))
        sim = _as_float(r.get("sim", ""))

        G.add_edge(
            a,
            b,
            **{
                "dist_km": dist_km,
                "dtw_cost_per_step": dtw_cps,
                "sim": sim,
                "curve_src": (r.get("curve_src") or "").strip(),
                "curve_dst": (r.get("curve_dst") or "").strip(),
            },
        )

    if G.number_of_nodes() == 0:
        raise ValueError("No valid nodes were loaded.")
    if G.number_of_edges() == 0:
        raise ValueError("No valid edges were loaded (or none matched loaded nodes).")

    return G


def _positions(G: "nx.Graph", *, prefer_geo: bool) -> Dict[str, Tuple[float, float]]:
    nodes = list(G.nodes)
    lats = np.asarray([G.nodes[n].get("lat", np.nan) for n in nodes], dtype="float64")
    lons = np.asarray([G.nodes[n].get("lon", np.nan) for n in nodes], dtype="float64")

    have_geo = np.isfinite(lats).sum() >= max(3, int(0.7 * len(nodes))) and np.isfinite(lons).sum() >= max(
        3, int(0.7 * len(nodes))
    )

    if prefer_geo and have_geo:
        xy = _project_lonlat(lats, lons)
        return {n: (float(xy[i, 0]), float(xy[i, 1])) for i, n in enumerate(nodes)}

    # Deterministic spring layout (seeded)
    return nx.spring_layout(G, seed=13)  # type: ignore[arg-type]


def _edge_widths(G: "nx.Graph", edges: List[Tuple[str, str]]) -> List[float]:
    sims = []
    for u, v in edges:
        s = float(G.edges[u, v].get("sim", np.nan))
        sims.append(s if np.isfinite(s) else 0.0)

    a = np.asarray(sims, dtype="float64")
    if np.all(a <= 0):
        return [1.0 for _ in edges]

    # Map sim in [min..max] -> width in [0.5..4]
    mn = float(np.min(a))
    mx = float(np.max(a))
    if mx <= mn + 1e-12:
        return [2.0 for _ in edges]

    w = 0.5 + 3.5 * ((a - mn) / (mx - mn))
    return [float(x) for x in w]


def main(argv: List[str] | None = None) -> int:
    _require()

    ap = argparse.ArgumentParser(description="Plot a framework/correlation graph from CSV artifacts.")
    ap.add_argument("--framework-dir", type=str, default="", help="Directory containing framework_nodes.csv + framework_edges.csv")
    ap.add_argument("--nodes-csv", type=str, default="", help="Path to nodes CSV (overrides --framework-dir)")
    ap.add_argument("--edges-csv", type=str, default="", help="Path to edges CSV (overrides --framework-dir)")
    ap.add_argument("--prefer-geo", action="store_true", help="Prefer lat/lon layout when available")
    ap.add_argument("--labels", action="store_true", help="Draw node labels (can be cluttered)")
    ap.add_argument("--out", type=str, default="", help="Optional output image path (png). If omitted, shows interactive window.")
    ap.add_argument("--dpi", type=int, default=160)

    args = ap.parse_args(argv)

    if args.framework_dir:
        fdir = Path(args.framework_dir)
        nodes_csv = Path(args.nodes_csv) if args.nodes_csv else (fdir / "framework_nodes.csv")
        edges_csv = Path(args.edges_csv) if args.edges_csv else (fdir / "framework_edges.csv")
    else:
        if not args.nodes_csv or not args.edges_csv:
            raise SystemExit("Provide either --framework-dir or both --nodes-csv and --edges-csv.")
        nodes_csv = Path(args.nodes_csv)
        edges_csv = Path(args.edges_csv)

    G = _load_graph(nodes_csv, edges_csv)
    pos = _positions(G, prefer_geo=bool(args.prefer_geo))

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    ax.set_axis_off()

    edges = list(G.edges)
    widths = _edge_widths(G, edges)

    nx.draw_networkx_edges(G, pos, ax=ax, width=widths, alpha=0.6)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=60, alpha=0.9)

    if args.labels:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=int(args.dpi), bbox_inches="tight")
        print(f"Wrote: {out}")
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
