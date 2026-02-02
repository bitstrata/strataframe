# src/strataframe/viz/step2b_pair_coverage_viz.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PairData:
    src_id: int
    dst_id: int
    src_label: str
    dst_label: str
    src_exp: Tuple[float, float]
    dst_exp: Tuple[float, float]
    src_act: Optional[Tuple[float, float]]
    dst_act: Optional[Tuple[float, float]]
    src_proj_from_dst: Optional[Tuple[float, float]]
    dst_proj_from_src: Optional[Tuple[float, float]]


def _safe_float(x: object) -> Optional[float]:
    try:
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _project_interval(
    *,
    act_top: float,
    act_base: float,
    exp_top: float,
    exp_base: float,
    exp_top_other: float,
    exp_base_other: float,
) -> Optional[Tuple[float, float]]:
    thk = float(exp_base - exp_top)
    if not np.isfinite(thk) or thk <= 0:
        return None
    f_top = (float(act_top) - float(exp_top)) / thk
    f_base = (float(act_base) - float(exp_top)) / thk
    thk_other = float(exp_base_other - exp_top_other)
    if not np.isfinite(thk_other) or thk_other <= 0:
        return None
    zt = float(exp_top_other) + f_top * thk_other
    zb = float(exp_top_other) + f_base * thk_other
    return (zt, zb)


def _pick_pair(
    *,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    pair: Optional[str],
    seed: int,
    include_excluded: bool,
) -> Tuple[int, int]:
    if pair:
        toks = str(pair).replace(":", ",").split(",")
        if len(toks) >= 2:
            try:
                return int(toks[0]), int(toks[1])
            except Exception:
                pass

    keep_nodes = nodes
    if (not include_excluded) and ("exclude_flag" in nodes.columns):
        keep_nodes = nodes[~nodes["exclude_flag"]].copy()
    keep_ids = set(keep_nodes["node_id"].astype(int).tolist())

    e = edges.copy()
    e["src_id"] = pd.to_numeric(e["src_id"], errors="coerce").astype("Int64")
    e["dst_id"] = pd.to_numeric(e["dst_id"], errors="coerce").astype("Int64")
    e = e[e["src_id"].notna() & e["dst_id"].notna()].copy()
    e["src_id"] = e["src_id"].astype(int)
    e["dst_id"] = e["dst_id"].astype(int)
    e = e[e["src_id"].isin(keep_ids) & e["dst_id"].isin(keep_ids)].copy()
    if e.empty:
        raise RuntimeError("No edges available for selection.")

    rng = np.random.default_rng(int(seed))
    row = e.sample(n=1, random_state=rng.integers(0, 1_000_000)).iloc[0]
    return int(row["src_id"]), int(row["dst_id"])


def _get_pair_data(nodes: pd.DataFrame, src_id: int, dst_id: int) -> PairData:
    idx = nodes.set_index("node_id")
    if src_id not in idx.index or dst_id not in idx.index:
        raise RuntimeError("Pair not found in nodes CSV.")
    n1 = idx.loc[src_id]
    n2 = idx.loc[dst_id]

    def _exp(n: pd.Series) -> Tuple[float, float]:
        return float(n["z_exp_top"]), float(n["z_exp_base"])

    def _act(n: pd.Series) -> Optional[Tuple[float, float]]:
        zt = _safe_float(n.get("z_gr_top_trim"))
        zb = _safe_float(n.get("z_gr_base_trim"))
        if zt is None or zb is None:
            return None
        return (float(zt), float(zb))

    exp1 = _exp(n1)
    exp2 = _exp(n2)
    act1 = _act(n1)
    act2 = _act(n2)

    proj1 = None
    if act2 is not None:
        proj1 = _project_interval(
            act_top=act2[0],
            act_base=act2[1],
            exp_top=exp2[0],
            exp_base=exp2[1],
            exp_top_other=exp1[0],
            exp_base_other=exp1[1],
        )
    proj2 = None
    if act1 is not None:
        proj2 = _project_interval(
            act_top=act1[0],
            act_base=act1[1],
            exp_top=exp1[0],
            exp_base=exp1[1],
            exp_top_other=exp2[0],
            exp_base_other=exp2[1],
        )

    label1 = str(n1.get("well_id", src_id))
    label2 = str(n2.get("well_id", dst_id))

    return PairData(
        src_id=src_id,
        dst_id=dst_id,
        src_label=label1,
        dst_label=label2,
        src_exp=exp1,
        dst_exp=exp2,
        src_act=act1,
        dst_act=act2,
        src_proj_from_dst=proj1,
        dst_proj_from_src=proj2,
    )


def _plot_pair(pair: PairData, out_path: Path, title: str = "") -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 6.2), sharey=True)

    def _plot_one(ax, label: str, exp: Tuple[float, float], act: Optional[Tuple[float, float]], proj: Optional[Tuple[float, float]]):
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([])
        ax.set_title(label)
        exp_top, exp_base = exp
        ax.fill_between([0.35, 0.65], [exp_top, exp_top], [exp_base, exp_base], color="#c7c7c7", alpha=0.18)
        ax.plot([0.3, 0.7], [exp_top, exp_top], color="#777777", linewidth=1)
        ax.plot([0.3, 0.7], [exp_base, exp_base], color="#777777", linewidth=1)

        if act is not None:
            ax.plot([0.45, 0.45], [act[0], act[1]], color="#1b9e77", linewidth=6, solid_capstyle="butt")
            ax.plot([0.38, 0.52], [act[0], act[0]], color="#1b9e77", linewidth=2)
            ax.plot([0.38, 0.52], [act[1], act[1]], color="#1b9e77", linewidth=2)

        if proj is not None:
            ax.plot([0.65, 0.65], [proj[0], proj[1]], color="#377eb8", linewidth=4, linestyle="--")
            ax.plot([0.6, 0.72], [proj[0], proj[0]], color="#377eb8", linewidth=1.6, linestyle="--")
            ax.plot([0.6, 0.72], [proj[1], proj[1]], color="#377eb8", linewidth=1.6, linestyle="--")

        ax.invert_yaxis()
        ax.set_ylabel("Depth (ft)")

    _plot_one(axes[0], f"{pair.src_label} ({pair.src_id})", pair.src_exp, pair.src_act, pair.src_proj_from_dst)
    _plot_one(axes[1], f"{pair.dst_label} ({pair.dst_id})", pair.dst_exp, pair.dst_act, pair.dst_proj_from_src)

    if title:
        fig.suptitle(title)

    # Simple legend
    axes[0].plot([], [], color="#1b9e77", linewidth=6, label="Actual data range")
    axes[0].plot([], [], color="#377eb8", linewidth=4, linestyle="--", label="Projected from neighbor")
    axes[0].plot([], [], color="#777777", linewidth=1, label="Expected window")
    axes[0].legend(loc="upper right", frameon=False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Step2b: visualize projected depth ranges for a well pair.")
    ap.add_argument("--nodes-csv", required=True)
    ap.add_argument("--edges-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pair", default="", help="Optional pair 'src_id,dst_id'.")
    ap.add_argument("--seed", type=int, default=1, help="Random seed when pair not provided.")
    ap.add_argument("--include-excluded", action="store_true", help="Allow excluded nodes in random selection.")
    ap.add_argument("--title", default="", help="Optional title.")

    args = ap.parse_args()
    nodes = pd.read_csv(args.nodes_csv)
    edges = pd.read_csv(args.edges_csv)

    src_id, dst_id = _pick_pair(
        nodes=nodes,
        edges=edges,
        pair=str(args.pair) if str(args.pair).strip() else None,
        seed=int(args.seed),
        include_excluded=bool(args.include_excluded),
    )
    pair = _get_pair_data(nodes, src_id, dst_id)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"step2b_pair_projection_{pair.src_id}_{pair.dst_id}.png"
    _plot_pair(pair, out_path, title=str(args.title))


if __name__ == "__main__":
    main()
