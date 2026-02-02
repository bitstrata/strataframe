# src/strataframe/viz/step2_dtw_pair_viz.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.collections import LineCollection

from strataframe.chronolog.dtw import compute_overlap_window, pad_for_distance, load_gr_vectors_cache, resample_from_cache
from strataframe.graph.las_utils import read_las_curve_resampled_ascii

try:
    from strataframe.viz.step3_colors import yellow_brown_perceptual_cmap
    _HAS_STEP3_COLORS = True
except Exception:  # pragma: no cover
    _HAS_STEP3_COLORS = False


def _load_step2_config(step2_dir: Path) -> Dict[str, Any]:
    diag = step2_dir / "diagnostics.json"
    if diag.exists():
        try:
            return json.loads(diag.read_text()).get("config", {})
        except Exception:
            return {}
    return {}


def _load_nodes(nodes_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(nodes_csv)
    df["node_id"] = pd.to_numeric(df["node_id"], errors="coerce").astype("Int64")
    df["depth_min"] = pd.to_numeric(df.get("depth_min"), errors="coerce")
    df["depth_max"] = pd.to_numeric(df.get("depth_max"), errors="coerce")
    return df[df["node_id"].notna()].set_index("node_id")


def _load_edge(
    edges_csv: Path,
    *,
    seed: int,
    pair: Optional[Tuple[int, int]] = None,
    status: str = "ok",
) -> pd.Series:
    edges = pd.read_csv(edges_csv)
    edges["src_id"] = pd.to_numeric(edges["src_id"], errors="coerce").astype("Int64")
    edges["dst_id"] = pd.to_numeric(edges["dst_id"], errors="coerce").astype("Int64")
    if "status" in edges.columns:
        if str(status).strip().lower() == "any":
            pass
        else:
            edges = edges[edges["status"] == str(status)].copy()
    edges = edges[edges["src_id"].notna() & edges["dst_id"].notna()].copy()
    if pair is not None:
        s, d = pair
        m = (edges["src_id"].astype(int) == int(s)) & (edges["dst_id"].astype(int) == int(d))
        if not m.any():
            m = (edges["src_id"].astype(int) == int(d)) & (edges["dst_id"].astype(int) == int(s))
        if not m.any():
            raise RuntimeError("Pair not found in dtw_edges.csv (status=ok).")
        return edges[m].iloc[0]
    if edges.empty:
        raise RuntimeError(f"No edges with status='{status}' in dtw_edges.csv")
    return edges.sample(n=1, random_state=int(seed)).iloc[0]


def _load_path(paths_jsonl: Path, src_id: int, dst_id: int, *, allow_missing: bool = False) -> Tuple[np.ndarray, bool]:
    if not paths_jsonl.exists():
        raise FileNotFoundError(paths_jsonl)
    with paths_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            s = int(obj.get("src_id"))
            d = int(obj.get("dst_id"))
            if (s == src_id and d == dst_id) or (s == dst_id and d == src_id):
                p = np.asarray(obj.get("path", []), dtype="int64")
                flip = bool(s == dst_id and d == src_id)
                return p, flip
    if allow_missing:
        return np.zeros((0, 2), dtype="int64"), False
    raise RuntimeError("DTW path not found for selected pair.")


def _resample_mask_from_cache(
    mask_full: np.ndarray,
    *,
    z_top: float,
    z_base: float,
    win_min: float,
    win_max: float,
    n_samples: int,
) -> np.ndarray:
    mask_full = np.asarray(mask_full, dtype="float64").reshape(-1)
    n_full = int(mask_full.size)
    if n_full < 2:
        return np.zeros((int(n_samples),), dtype=bool)
    z_full = np.linspace(float(z_top), float(z_base), n_full, dtype="float64")
    z_win = np.linspace(float(win_min), float(win_max), int(n_samples), dtype="float64")
    m = np.interp(z_win, z_full, mask_full, left=1.0, right=1.0)
    return (m >= 0.5).astype(bool)


def _load_curve(
    *,
    node_id: int,
    las_path: Path,
    win_min: float,
    win_max: float,
    cfg: Dict[str, Any],
    cache: Optional[Dict[str, Any]],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    n_samples = int(cfg.get("n_samples", 512))
    p_lo = float(cfg.get("p_lo", 1.0))
    p_hi = float(cfg.get("p_hi", 99.0))
    min_finite = int(cfg.get("min_finite", 10))
    max_rows = int(cfg.get("max_rows", 0))
    curve_mnemonic = str(cfg.get("curve_mnemonic", "GR"))

    if cache is not None and int(node_id) in cache["index"]:
        i = cache["index"][int(node_id)]
        x = resample_from_cache(
            cache["x_norm"][i],
            cache["z_top"][i],
            cache["z_base"][i],
            win_min,
            win_max,
            n_samples=n_samples,
            p_lo=p_lo,
            p_hi=p_hi,
        )
        imputed = None
        if cache.get("imputed_mask") is not None:
            imputed = _resample_mask_from_cache(
                cache["imputed_mask"][i],
                z_top=cache["z_top"][i],
                z_base=cache["z_base"][i],
                win_min=win_min,
                win_max=win_max,
                n_samples=n_samples,
            )
        return x, imputed

    x_norm, _, _, _, _ = read_las_curve_resampled_ascii(
        las_path,
        n_samples=n_samples,
        curve_candidates=(curve_mnemonic,),
        p_lo=p_lo,
        p_hi=p_hi,
        min_finite=min_finite,
        max_rows=max_rows,
        window_min=win_min,
        window_max=win_max,
    )
    return x_norm, None


def _plot_matrix(
    *,
    x1: np.ndarray,
    x2: np.ndarray,
    path: np.ndarray,
    alpha: float,
    out_path: Path,
    cmap: str = "magma",
) -> None:
    n = int(x1.size)
    C = np.abs(x2[:, None] - x1[None, :]) ** float(alpha)

    fig = plt.figure(figsize=(7.5, 7.5), dpi=160)
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 4], height_ratios=[1, 4], wspace=0.05, hspace=0.05)

    ax_top = fig.add_subplot(gs[0, 1])
    ax_left = fig.add_subplot(gs[1, 0])
    ax_mat = fig.add_subplot(gs[1, 1])

    # Matrix
    im = ax_mat.imshow(C, origin="upper", cmap=cmap)
    if path is not None and path.size > 0:
        ax_mat.plot(path[:, 0], path[:, 1], color="white", linewidth=1.0, alpha=0.9)
    ax_mat.set_xlim(0, n - 1)
    ax_mat.set_ylim(n - 1, 0)
    ax_mat.set_xticks([])
    ax_mat.set_yticks([])
    fig.colorbar(im, ax=ax_mat, fraction=0.046, pad=0.02)

    # Top curve (x1)
    ax_top.plot(np.arange(n), x1, color="black", linewidth=0.8)
    ax_top.set_xlim(0, n - 1)
    ax_top.set_ylim(0, 1)
    ax_top.axis("off")

    # Left curve (x2)
    ax_left.plot(x2, np.arange(n), color="black", linewidth=0.8)
    ax_left.set_ylim(n - 1, 0)
    ax_left.set_xlim(0, 1)
    ax_left.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format=out_path.suffix.lstrip("."), bbox_inches="tight")
    plt.close(fig)


def _plot_fence(
    *,
    x1: np.ndarray,
    x2: np.ndarray,
    z: np.ndarray,
    path: np.ndarray,
    out_path: Path,
    corr_every: int = 16,
    imputed1: Optional[np.ndarray] = None,
    imputed2: Optional[np.ndarray] = None,
    z_actual1: Optional[Tuple[float, float]] = None,
    z_actual2: Optional[Tuple[float, float]] = None,
) -> None:
    n = int(x1.size)
    track_w = 1.0
    gap_w = 1.0
    left_base = 0.0
    right_base = track_w + gap_w

    # Normalize depth to a datum at top
    z0 = z - float(z[0])

    # Track x positions
    x_left = left_base + x1 * track_w
    x_right = right_base + x2 * track_w

    fig, ax = plt.subplots(figsize=(8, 9), dpi=160)

    # Shade imputed intervals
    def _shade_imputed(imputed: Optional[np.ndarray], x_left: float, x_right: float) -> None:
        if imputed is None:
            return
        m = np.asarray(imputed, dtype=bool)
        if m.size != z.size:
            return
        idx = np.where(m)[0]
        if idx.size == 0:
            return
        # group contiguous
        start = idx[0]
        prev = idx[0]
        for i in idx[1:]:
            if i != prev + 1:
                y0 = z0[start]
                y1 = z0[prev]
                ax.fill(
                    [x_left, x_right, x_right, x_left],
                    [y0, y0, y1, y1],
                    color="#8aa1ff",
                    alpha=0.12,
                    linewidth=0,
                    zorder=0,
                )
                start = i
            prev = i
        y0 = z0[start]
        y1 = z0[prev]
        ax.fill(
            [x_left, x_right, x_right, x_left],
            [y0, y0, y1, y1],
            color="#8aa1ff",
            alpha=0.12,
            linewidth=0,
            zorder=0,
        )

    _shade_imputed(imputed1, left_base, left_base + track_w)
    _shade_imputed(imputed2, right_base, right_base + track_w)

    # Fill intervals using path segments
    if _HAS_STEP3_COLORS:
        cmap = yellow_brown_perceptual_cmap()
    else:
        cmap = plt.get_cmap("YlOrBr")

    if path is not None and path.size > 0:
        step = max(1, int(corr_every))
        idxs = np.arange(0, path.shape[0], step, dtype="int64")
        if idxs[-1] != path.shape[0] - 1:
            idxs = np.append(idxs, path.shape[0] - 1)

        for a, b in zip(idxs[:-1], idxs[1:]):
            i0, j0 = path[a]
            i1, j1 = path[b]
            i0 = int(np.clip(i0, 0, n - 1))
            i1 = int(np.clip(i1, 0, n - 1))
            j0 = int(np.clip(j0, 0, n - 1))
            j1 = int(np.clip(j1, 0, n - 1))
            # Build a quad between tiepoints
            poly_x = [x_left[i0], x_left[i1], x_right[j1], x_right[j0]]
            poly_y = [z0[i0], z0[i1], z0[j1], z0[j0]]
            avg = float(np.nanmean([x1[i0], x1[i1], x2[j0], x2[j1]]))
            color = cmap(avg)
            ax.fill(poly_x, poly_y, color=color, alpha=0.55, linewidth=0)

        # Correlation lines
        line_pts = []
        for k in idxs:
            i, j = path[int(k)]
            i = int(np.clip(i, 0, n - 1))
            j = int(np.clip(j, 0, n - 1))
            line_pts.append([(x_left[i], z0[i]), (x_right[j], z0[j])])
        lc = LineCollection(line_pts, colors="#444444", linewidths=0.5, alpha=0.6)
        ax.add_collection(lc)

    # Curves
    ax.plot(x_left, z0, color="black", linewidth=0.8)
    ax.plot(x_right, z0, color="black", linewidth=0.8)

    # Axis formatting
    ax.set_ylim(z0.max(), z0.min())
    ax.set_xlim(left_base - 0.1, right_base + track_w + 0.1)
    ax.set_xticks([])
    ax.set_ylabel("Depth (relative)")
    ax.set_title("DTW Fence (two wells)")

    # Actual GR coverage markers (if provided)
    if z_actual1 is not None:
        a0, a1 = z_actual1
        ax.hlines([a0 - z[0], a1 - z[0]], left_base, left_base + track_w, colors="#d62728", linewidth=0.8)
    if z_actual2 is not None:
        b0, b1 = z_actual2
        ax.hlines([b0 - z[0], b1 - z[0]], right_base, right_base + track_w, colors="#d62728", linewidth=0.8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, format=out_path.suffix.lstrip("."), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Step2 DTW pair visualizations.")
    ap.add_argument("--nodes-csv", required=True)
    ap.add_argument("--edges-csv", required=True)
    ap.add_argument("--paths-jsonl", required=True)
    ap.add_argument("--step2-dir", required=True, help="Step2 output dir containing diagnostics.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gr-vectors-npz", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pair", type=str, default="", help="Override pair as 'src_id,dst_id'")
    ap.add_argument("--status", type=str, default="ok", help="Edge status to sample (ok, overlap_too_small, no_overlap, dtw_fail, any)")
    ap.add_argument("--cmap", type=str, default="magma")
    ap.add_argument("--corr-every", type=int, default=16)
    ap.add_argument("--allow-missing-path", action="store_true")

    args = ap.parse_args()

    step2_dir = Path(args.step2_dir)
    cfg = _load_step2_config(step2_dir)
    nodes = _load_nodes(Path(args.nodes_csv))
    pair = None
    if str(args.pair).strip():
        s, d = str(args.pair).split(",", 1)
        pair = (int(s.strip()), int(d.strip()))

    edge = _load_edge(Path(args.edges_csv), seed=int(args.seed), pair=pair, status=str(args.status))
    src_id = int(edge["src_id"])
    dst_id = int(edge["dst_id"])

    n1 = nodes.loc[src_id]
    n2 = nodes.loc[dst_id]

    zmin1 = float(n1["depth_min"])
    zmax1 = float(n1["depth_max"])
    zmin2 = float(n2["depth_min"])
    zmax2 = float(n2["depth_max"])
    dist_km = float(edge.get("dist_km", 0.0))

    pad = pad_for_distance(
        base_pad_ft=float(cfg.get("base_pad_ft", 10.0)),
        pad_slope_ft_per_km=float(cfg.get("pad_slope_ft_per_km", 0.0)),
        max_pad_ft=float(cfg.get("max_pad_ft", 200.0)),
        dist_km=float(dist_km),
    )
    win_min, win_max, _, _ = compute_overlap_window(zmin1, zmax1, zmin2, zmax2, pad)

    cache = None
    if str(args.gr_vectors_npz).strip():
        cache = load_gr_vectors_cache(Path(args.gr_vectors_npz))

    x1, im1 = _load_curve(
        node_id=src_id,
        las_path=Path(n1["las_path"]),
        win_min=win_min,
        win_max=win_max,
        cfg=cfg,
        cache=cache,
    )
    x2, im2 = _load_curve(
        node_id=dst_id,
        las_path=Path(n2["las_path"]),
        win_min=win_min,
        win_max=win_max,
        cfg=cfg,
        cache=cache,
    )

    path, flip = _load_path(Path(args.paths_jsonl), src_id, dst_id, allow_missing=bool(args.allow_missing_path))
    if flip and path.ndim == 2 and path.shape[1] == 2:
        path = path[:, ::-1]

    n_samples = int(cfg.get("n_samples", x1.size))
    z = np.linspace(win_min, win_max, n_samples, dtype="float64")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_matrix(
        x1=x1,
        x2=x2,
        path=path,
        alpha=float(cfg.get("alpha", 0.15)),
        out_path=out_dir / f"step2_pair_matrix_{src_id}_{dst_id}.png",
        cmap=str(args.cmap),
    )

    _plot_fence(
        x1=x1,
        x2=x2,
        z=z,
        path=path,
        out_path=out_dir / f"step2_pair_fence_{src_id}_{dst_id}.png",
        corr_every=int(args.corr_every),
        imputed1=im1,
        imputed2=im2,
        z_actual1=(float(n1.get("z_gr_top_trim", n1.get("depth_min", np.nan))),
                   float(n1.get("z_gr_base_trim", n1.get("depth_max", np.nan)))),
        z_actual2=(float(n2.get("z_gr_top_trim", n2.get("depth_min", np.nan))),
                   float(n2.get("z_gr_base_trim", n2.get("depth_max", np.nan)))),
    )


if __name__ == "__main__":
    main()
