# src/strataframe/viz/step1_graph_map.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


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


def _load_edges(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("src_id", "dst_id"):
        if col not in df.columns:
            raise ValueError(f"graph_edges.csv missing {col}")
    df["src_id"] = pd.to_numeric(df["src_id"], errors="coerce").astype("Int64")
    df["dst_id"] = pd.to_numeric(df["dst_id"], errors="coerce").astype("Int64")
    if "edge_type" not in df.columns:
        df["edge_type"] = "edge"
    return df[df["src_id"].notna() & df["dst_id"].notna()].copy()


def _maybe_kansas_outline(ax, *, enabled: bool = True) -> None:
    if not enabled:
        return
    try:
        import cartopy.crs as ccrs  # type: ignore
        import cartopy.io.shapereader as shpreader  # type: ignore
        import cartopy.feature as cfeature  # type: ignore
    except Exception:
        return

    shp = shpreader.natural_earth(
        resolution="50m", category="cultural", name="admin_1_states_provinces_lines"
    )
    reader = shpreader.Reader(shp)
    geometries = [rec.geometry for rec in reader.records() if rec.attributes.get("name") == "Kansas"]
    if geometries:
        ax.add_geometries(
            geometries,
            crs=ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="black",
            linewidth=0.8,
            alpha=0.7,
        )
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.4)


def _build_delaunay_edges(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    Returns edges as (M,2) int array of node indices for raw Delaunay triangulation.
    """
    try:
        from scipy.spatial import Delaunay  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("scipy is required for raw Delaunay visualization") from e

    if lat.size < 3:
        return np.zeros((0, 2), dtype="int64")

    # simple local projection for triangulation
    lat0 = float(np.nanmedian(lat)) if lat.size else 0.0
    x = lon * np.cos(np.deg2rad(lat0))
    y = lat
    xy = np.column_stack([x, y]).astype("float64", copy=False)

    tri = Delaunay(xy)
    edges = set()
    for t in tri.simplices:
        a, b, c = int(t[0]), int(t[1]), int(t[2])
        edges.add((a, b) if a < b else (b, a))
        edges.add((b, c) if b < c else (c, b))
        edges.add((a, c) if a < c else (c, a))
    return np.array(sorted(edges), dtype="int64")


def plot_graph_map(
    *,
    nodes_csv: Path,
    edges_csv: Path,
    out_png: Path,
    max_edges: int = 5000,
    edge_type: Optional[str] = None,
    show_outline: bool = True,
    raw_delaunay: bool = False,
    composite_mode: bool = False,
) -> None:
    nodes = _load_nodes(nodes_csv)
    edges = _load_edges(edges_csv)

    if edge_type:
        et = str(edge_type).strip().lower()
        if et == "delaunay":
            edges = edges[edges["edge_type"].astype(str).isin(["delaunay", "both"])].copy()
        elif et == "knn":
            edges = edges[edges["edge_type"].astype(str).isin(["knn", "both"])].copy()
        else:
            edges = edges[edges["edge_type"].astype(str) == str(edge_type)].copy()

    node_idx = nodes.set_index("node_id")

    if raw_delaunay:
        # Build raw triangulation edges from nodes (no pruning)
        edges_idx = _build_delaunay_edges(nodes["lat"].to_numpy(), nodes["lon"].to_numpy())
        edges = pd.DataFrame(edges_idx, columns=["src_id", "dst_id"])
        edges["edge_type"] = "delaunay_raw"
    else:
        edges = edges[edges["src_id"].isin(node_idx.index) & edges["dst_id"].isin(node_idx.index)].copy()
        edges = edges.reset_index(drop=True)

        if composite_mode:
            # Always keep all Delaunay edges; sample only kNN-only edges if needed.
            dmask = edges["edge_type"].astype(str).isin(["delaunay", "both"])
            edges_d = edges[dmask].copy()
            edges_d["edge_type"] = "delaunay"

            edges_k = edges[edges["edge_type"].astype(str) == "knn"].copy()
            if len(edges_k) > max_edges:
                edges_k = edges_k.sample(n=max_edges, random_state=42)
            edges_k["edge_type"] = "knn"

            edges = pd.concat([edges_k, edges_d], ignore_index=True)
        else:
            if len(edges) > max_edges:
                edges = edges.sample(n=max_edges, random_state=42)

    src_lon = node_idx.loc[edges["src_id"].values, "lon"].to_numpy()
    src_lat = node_idx.loc[edges["src_id"].values, "lat"].to_numpy()
    dst_lon = node_idx.loc[edges["dst_id"].values, "lon"].to_numpy()
    dst_lat = node_idx.loc[edges["dst_id"].values, "lat"].to_numpy()

    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    fig = plt.figure(figsize=(8.5, 7.5), dpi=160)
    ax = plt.axes()
    _maybe_kansas_outline(ax, enabled=bool(show_outline))

    colors: Dict[str, str] = {
        "delaunay": "#4C78A8",
        "knn": "#F58518",
        "both": "#54A24B",
        "delaunay_raw": "#4C78A8",
    }

    draw_order = ["knn", "both", "delaunay", "delaunay_raw"]
    for etype in draw_order:
        group = edges[edges["edge_type"].astype(str) == etype]
        if group.empty:
            continue
        col = colors.get(str(etype), "#7F7F7F")
        mask = group.index.to_numpy()
        lw = 0.45 if str(etype) in {"delaunay", "both", "delaunay_raw"} else 0.2
        alpha = 0.7 if str(etype) in {"delaunay", "both", "delaunay_raw"} else 0.25
        segs = np.stack(
            [
                np.column_stack([src_lon[mask], src_lat[mask]]),
                np.column_stack([dst_lon[mask], dst_lat[mask]]),
            ],
            axis=1,
        )
        lc = LineCollection(segs, colors=col, linewidths=lw, alpha=alpha, rasterized=False)
        ax.add_collection(lc)

    ax.scatter(nodes["lon"], nodes["lat"], s=3, c="black", alpha=0.8, rasterized=False)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    title = "Step1 Graph (Delaunay + kNN)"
    if raw_delaunay:
        title = "Step1 Graph (Raw Delaunay Triangulation)"
    ax.set_title(title)

    # Legend (only for types present)
    from matplotlib.lines import Line2D

    present = set(edges["edge_type"].astype(str).unique().tolist())
    handles = []
    if "delaunay_raw" in present or "delaunay" in present:
        handles.append(Line2D([0], [0], color=colors["delaunay"], lw=1.2, label="Delaunay"))
    if "knn" in present:
        handles.append(Line2D([0], [0], color=colors["knn"], lw=1.2, label="kNN"))
    if "both" in present:
        handles.append(Line2D([0], [0], color=colors["both"], lw=1.2, label="Both"))
    if handles:
        ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, format=out_png.suffix.lstrip("."))
    plt.close(fig)


def _resolve_out_path(base: Path, suffix: str, ext: str) -> Path:
    if base.suffix.lower() in {".png", ".svg", ".pdf"}:
        return base.with_name(f"{base.stem}_{suffix}.{ext}")
    base.mkdir(parents=True, exist_ok=True)
    return base / f"step1_graph_{suffix}.{ext}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot Step1 graph map.")
    ap.add_argument("--nodes-csv", required=True, help="graph_nodes.csv")
    ap.add_argument("--edges-csv", required=True, help="graph_edges.csv")
    ap.add_argument("--out-png", required=True, help="Output image path or directory")
    ap.add_argument("--max-edges", type=int, default=5000)
    ap.add_argument("--edge-type", type=str, default="")
    ap.add_argument("--write-both", action="store_true", help="Write delaunay + composite images")
    ap.add_argument("--no-outline", action="store_true", help="Disable state outline (cartopy)")
    ap.add_argument("--raw-delaunay", action="store_true", help="Draw raw Delaunay from nodes (ignores edges CSV)")
    ap.add_argument("--out-ext", type=str, default="png", choices=["png", "svg", "pdf"], help="Output format")
    args = ap.parse_args()

    out_path = Path(args.out_png)
    if bool(args.write_both):
        plot_graph_map(
            nodes_csv=Path(args.nodes_csv),
            edges_csv=Path(args.edges_csv),
            out_png=_resolve_out_path(out_path, "delaunay", str(args.out_ext)),
            max_edges=int(args.max_edges),
            edge_type="delaunay" if not bool(args.raw_delaunay) else None,
            show_outline=not bool(args.no_outline),
            raw_delaunay=bool(args.raw_delaunay),
        )
        plot_graph_map(
            nodes_csv=Path(args.nodes_csv),
            edges_csv=Path(args.edges_csv),
            out_png=_resolve_out_path(out_path, "composite", str(args.out_ext)),
            max_edges=int(args.max_edges),
            edge_type=None,
            show_outline=not bool(args.no_outline),
            raw_delaunay=False,
            composite_mode=True,
        )
    else:
        plot_graph_map(
            nodes_csv=Path(args.nodes_csv),
            edges_csv=Path(args.edges_csv),
            out_png=out_path,
            max_edges=int(args.max_edges),
            edge_type=str(args.edge_type).strip() or None,
            show_outline=not bool(args.no_outline),
            raw_delaunay=bool(args.raw_delaunay),
        )


if __name__ == "__main__":
    main()
