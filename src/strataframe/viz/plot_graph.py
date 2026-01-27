# src/strataframe/viz/plot_graph.py
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover
    nx = None  # type: ignore

try:
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.collections import LineCollection  # type: ignore
except Exception:  # pragma: no cover
    plt = None  # type: ignore
    LineCollection = None  # type: ignore

# Prefer your shared CSV reader if present; fall back to a local robust reader.
try:  # pragma: no cover
    from strataframe.io.csv import read_csv_rows  # type: ignore
except Exception:  # pragma: no cover

    def _read_sample_text(path: Path, *, max_bytes: int = 32_000) -> str:
        with path.open("rb") as f:
            blob = f.read(max_bytes)
        return blob.decode("utf-8-sig", errors="replace")

    def _first_nonempty_line(sample: str) -> str:
        for line in sample.splitlines():
            s = line.strip()
            if s and not s.lstrip().startswith("#"):
                return s
        return ""

    def _detect_delimiter(sample: str) -> str:
        first = _first_nonempty_line(sample)
        cands = [",", "\t", ";", "|"]

        has = {d: (d in first) for d in cands}
        if has["\t"] and not (has[","] or has[";"] or has["|"]):
            return "\t"
        if has[","] and not (has["\t"] or has[";"] or has["|"]):
            return ","

        lines = [ln for ln in sample.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
        lines = lines[:50]

        def _score(delim: str) -> float:
            counts = [ln.count(delim) for ln in lines]
            if not counts:
                return 0.0
            mx = max(counts)
            if mx == 0:
                return 0.0
            mean = sum(counts) / len(counts)
            var = sum((c - mean) ** 2 for c in counts) / max(1, len(counts) - 1)
            return float(mean) / (1.0 + float(var))

        scores = {d: _score(d) for d in cands}
        best = max(cands, key=lambda d: scores[d])
        if scores.get(best, 0.0) > 0.0:
            return best

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=cands)
            return dialect.delimiter
        except Exception:
            return ","

    def read_csv_rows(path: Path, *, delimiter: Optional[str] = None) -> List[Dict[str, str]]:
        if delimiter is None:
            sample = _read_sample_text(path)
            delimiter = _detect_delimiter(sample)

        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            rdr = csv.DictReader(f, delimiter=delimiter)
            if rdr.fieldnames is None:
                raise ValueError(f"No header row found in {path}")

            out: List[Dict[str, str]] = []
            for r in rdr:
                if r is None:
                    continue
                out.append({k: (v if v is not None else "") for k, v in r.items()})
            return out


def _require() -> None:
    if nx is None:
        raise RuntimeError("networkx is required. Install with: pip install networkx")
    if plt is None or LineCollection is None:
        raise RuntimeError("matplotlib is required. Install with: pip install matplotlib")


def _as_float(s: object) -> float:
    try:
        v = float(str(s).strip())
        return v if np.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def _get_first_present(row: Dict[str, str], keys: Sequence[str]) -> str:
    for k in keys:
        v = (row.get(k) or "").strip()
        if v:
            return v
    return ""


def _project_lonlat(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Simple lon/lat projection to a local Cartesian-ish space for plotting.
    Uses a local scale ~km (good enough for visualization).
    """
    lat0 = float(np.nanmean(lats)) if np.isfinite(np.nanmean(lats)) else 0.0
    deg_to_km = 111.32
    x = (lons * np.cos(np.deg2rad(lat0))) * deg_to_km
    y = lats * deg_to_km
    return np.vstack([x, y]).T


def _detect_edge_cols(rows: List[Dict[str, str]]) -> Tuple[Optional[str], Optional[str]]:
    """
    Best-effort detection of edge endpoint columns across your pipeline variants.
    """
    if not rows:
        return None, None
    cols = set(rows[0].keys())

    u_cands = ["src_rep_id", "u", "src", "src_id", "src_node", "node_u", "rep_i", "i"]
    v_cands = ["dst_rep_id", "v", "dst", "dst_id", "dst_node", "node_v", "rep_j", "j"]

    cu = next((c for c in u_cands if c in cols), None)
    cv = next((c for c in v_cands if c in cols), None)

    if cu and cv:
        return cu, cv

    if "u" in cols and "v" in cols:
        return "u", "v"

    return None, None


def _load_graph(nodes_csv: Path, edges_csv: Path) -> "nx.Graph":
    nodes = read_csv_rows(nodes_csv)
    edges = read_csv_rows(edges_csv)

    if not nodes:
        raise ValueError(f"No rows in nodes csv: {nodes_csv}")
    if not edges:
        raise ValueError(f"No rows in edges csv: {edges_csv}")

    def _node_id(r: Dict[str, str]) -> str:
        return _get_first_present(r, ["rep_id", "well_id", "node_id", "id"])

    G = nx.Graph()

    # ---- nodes
    for r in nodes:
        nid = _node_id(r)
        if not nid:
            continue

        lat = _as_float(_get_first_present(r, ["lat", "latitude"]))
        lon = _as_float(_get_first_present(r, ["lon", "longitude"]))

        # Preserve common fields but also keep original row for later debugging.
        G.add_node(
            nid,
            **{
                "rep_id": nid,
                "lat": lat,
                "lon": lon,
                "h3_cell": _get_first_present(r, ["h3_cell", "cell", "bin_id"]),
                "h3_res": _get_first_present(r, ["h3_res", "res"]),
                "score": _get_first_present(r, ["score"]),
                "las_path": _get_first_present(r, ["las_path", "las", "path"]),
                "curve_used": _get_first_present(r, ["curve_used", "picked_gr", "curve"]),
                "_row": r,
            },
        )

    if G.number_of_nodes() == 0:
        raise ValueError("No valid nodes were loaded (no usable rep_id/well_id/node_id).")

    # ---- edges
    cu, cv = _detect_edge_cols(edges)
    if cu is None or cv is None:
        raise ValueError(
            f"Edges CSV {edges_csv} missing recognizable endpoint columns. "
            "Expected one of: src_rep_id/u/src/... and dst_rep_id/v/dst/..."
        )

    kept = 0
    for r in edges:
        a = (r.get(cu) or "").strip()
        b = (r.get(cv) or "").strip()
        if not a or not b or a == b:
            continue
        if not (G.has_node(a) and G.has_node(b)):
            continue

        # Common edge keys in your pipeline
        dist_km = _as_float(_get_first_present(r, ["dist_km", "distance_km", "w_km"]))
        dtw_cps = _as_float(_get_first_present(r, ["dtw_cost_per_step", "cps", "cost_per_step"]))
        sim = _as_float(_get_first_present(r, ["sim", "similarity"]))

        G.add_edge(
            a,
            b,
            **{
                "dist_km": dist_km,
                "dtw_cost_per_step": dtw_cps,
                "sim": sim,
                "curve_src": _get_first_present(r, ["curve_src", "curve_u", "src_curve"]),
                "curve_dst": _get_first_present(r, ["curve_dst", "curve_v", "dst_curve"]),
                "_row": r,
            },
        )
        kept += 1

    if G.number_of_edges() == 0:
        raise ValueError("No valid edges were loaded (or none matched loaded nodes).")

    return G


def _positions(G: "nx.Graph", *, prefer_geo: bool, seed: int = 13) -> Dict[str, Tuple[float, float]]:
    nodes = list(G.nodes)
    lats = np.asarray([G.nodes[n].get("lat", np.nan) for n in nodes], dtype="float64")
    lons = np.asarray([G.nodes[n].get("lon", np.nan) for n in nodes], dtype="float64")

    n = len(nodes)
    nlat = int(np.isfinite(lats).sum())
    nlon = int(np.isfinite(lons).sum())
    have_geo = (n >= 3) and (nlat >= max(3, int(0.7 * n))) and (nlon >= max(3, int(0.7 * n)))

    if prefer_geo and have_geo:
        xy = _project_lonlat(lats, lons)
        # Fill any missing coords (if present) to avoid crashing draw
        mx = float(np.nanmean(xy[:, 0])) if np.isfinite(np.nanmean(xy[:, 0])) else 0.0
        my = float(np.nanmean(xy[:, 1])) if np.isfinite(np.nanmean(xy[:, 1])) else 0.0
        xy2 = xy.copy()
        bad = ~np.isfinite(xy2)
        if bad.any():
            xy2[bad[:, 0], 0] = mx
            xy2[bad[:, 1], 1] = my
        return {n0: (float(xy2[i, 0]), float(xy2[i, 1])) for i, n0 in enumerate(nodes)}

    # Deterministic spring layout
    return nx.spring_layout(G, seed=int(seed))  # type: ignore[arg-type]


def _edge_metric(G: "nx.Graph", u: str, v: str, *, by: str) -> float:
    ed = G.edges[u, v]
    if by == "sim":
        x = float(ed.get("sim", np.nan))
        return x if np.isfinite(x) else np.nan
    if by == "dtw":
        x = float(ed.get("dtw_cost_per_step", np.nan))
        return x if np.isfinite(x) else np.nan
    if by == "dist":
        x = float(ed.get("dist_km", np.nan))
        return x if np.isfinite(x) else np.nan
    return np.nan


def _edge_widths(G: "nx.Graph", edges: List[Tuple[str, str]], *, by: str) -> List[float]:
    """
    Width mapping:
      - sim: higher sim -> thicker
      - dtw: lower dtw_cps -> thicker
      - dist: lower dist -> thicker (usually)
    """
    vals = np.asarray([_edge_metric(G, u, v, by=by) for u, v in edges], dtype="float64")
    fin = np.isfinite(vals)
    if int(fin.sum()) == 0:
        return [1.0 for _ in edges]

    v = vals.copy()
    v[~fin] = np.nan

    mn = float(np.nanmin(v))
    mx = float(np.nanmax(v))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn + 1e-12:
        return [2.0 for _ in edges]

    # Normalize to [0,1]
    t = (v - mn) / (mx - mn)
    t = np.clip(t, 0.0, 1.0)

    # Invert for dtw/dist: smaller is "stronger"
    if by in {"dtw", "dist"}:
        t = 1.0 - t

    # Width in [0.5..4.0]
    w = 0.5 + 3.5 * t
    w[~np.isfinite(w)] = 1.0
    return [float(x) for x in w]


def _sample_edges(edges: List[Tuple[str, str]], *, max_edges: int, seed: int) -> List[Tuple[str, str]]:
    if max_edges <= 0 or len(edges) <= max_edges:
        return edges
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(len(edges), size=int(max_edges), replace=False)
    return [edges[int(i)] for i in idx]


def main(argv: List[str] | None = None) -> int:
    _require()

    ap = argparse.ArgumentParser(description="Plot a framework/correlation graph from CSV artifacts.")
    ap.add_argument("--framework-dir", type=str, default="", help="Directory containing framework_nodes.csv + framework_edges.csv")
    ap.add_argument("--nodes-csv", type=str, default="", help="Path to nodes CSV (overrides --framework-dir)")
    ap.add_argument("--edges-csv", type=str, default="", help="Path to edges CSV (overrides --framework-dir)")

    ap.add_argument("--prefer-geo", action="store_true", help="Prefer lat/lon layout when available")
    ap.add_argument("--labels", action="store_true", help="Draw node labels (can be cluttered)")

    # NEW but backward compatible
    ap.add_argument("--max-edges", type=int, default=120_000, help="Max edges to render (random sample if larger)")
    ap.add_argument("--seed", type=int, default=13, help="Seed for layout + edge sampling")
    ap.add_argument("--width-by", type=str, default="sim", choices=["sim", "dtw", "dist"], help="Edge width metric")
    ap.add_argument("--edge-alpha", type=float, default=0.55, help="Edge alpha")
    ap.add_argument("--node-size", type=float, default=18.0, help="Node marker size")

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
    pos = _positions(G, prefer_geo=bool(args.prefer_geo), seed=int(args.seed))

    # Collect + sample edges for rendering
    edges_all = list(G.edges)
    edges = _sample_edges(edges_all, max_edges=int(args.max_edges), seed=int(args.seed))
    widths = _edge_widths(G, edges, by=str(args.width_by))

    # Build fast LineCollection instead of nx.draw_networkx_edges (performance)
    segs: List[np.ndarray] = []
    for (u, v) in edges:
        pu = pos.get(u)
        pv = pos.get(v)
        if pu is None or pv is None:
            continue
        a = np.asarray(pu, dtype="float64")
        b = np.asarray(pv, dtype="float64")
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            continue
        segs.append(np.vstack([a, b]))

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_axis_off()
    ax.set_title(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges (rendered {len(edges):,})")

    if segs:
        # Matplotlib supports per-segment linewidths if list length matches seg count.
        lw = widths[: len(segs)] if len(widths) >= len(segs) else 1.0
        lc = LineCollection(segs, linewidths=lw, alpha=float(args.edge_alpha), colors="0.25")
        ax.add_collection(lc)

    # Nodes
    nodes = list(G.nodes)
    xy = np.asarray([pos[n] for n in nodes], dtype="float64")
    ax.scatter(xy[:, 0], xy[:, 1], s=float(args.node_size), alpha=0.9)

    # Labels (guard for huge graphs)
    if bool(args.labels):
        if len(nodes) > 1500:
            # still draw, but it will be unreadable; keep deterministic and simple
            print(f"Warning: labels requested but graph has {len(nodes)} nodes; output will be cluttered.")
        for n in nodes:
            x, y = pos[n]
            ax.text(float(x), float(y), str(n), fontsize=7, alpha=0.85)

    ax.autoscale_view()
    fig.tight_layout()

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
