# src/strataframe/viz/step4_viz.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from strataframe.spatial.grid_index import cell_id_to_ij
from .step3_common import ensure_dir, load_npz


def _require_pd() -> None:
    if pd is None:
        raise RuntimeError("pandas is required for Step 4 visualizations. Install with: pip install pandas")


def _load_cell_logs(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = load_npz(Path(npz_path))
    if "cell_ids" not in z or "rgt_grid" not in z or "type_log" not in z:
        raise ValueError("cell_type_logs.npz missing required keys: cell_ids, rgt_grid, type_log")
    cell_ids = np.asarray(z["cell_ids"], dtype=np.str_)
    rgt = np.asarray(z["rgt_grid"], dtype="float64")
    tlog = np.asarray(z["type_log"], dtype="float64")

    # Allow 1D rgt grid by broadcasting
    if rgt.ndim == 1:
        rgt = np.tile(rgt.reshape(1, -1), (int(cell_ids.size), 1))
    return cell_ids, rgt, tlog


def _load_tops(npz_path: Optional[Path]) -> Optional[np.ndarray]:
    if not npz_path:
        return None
    p = Path(npz_path)
    if not p.exists():
        return None
    # tops_by_level is stored as an object array; requires allow_pickle=True
    z = np.load(p, allow_pickle=True)
    if "tops_by_level" not in z.files:
        return None
    return np.asarray(z["tops_by_level"], dtype=object)


def _pick_cells(cell_ids: np.ndarray, *, max_cells: int, seed: int) -> List[int]:
    n = int(cell_ids.size)
    if max_cells <= 0 or n <= max_cells:
        return list(range(n))
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(n, size=int(max_cells), replace=False)
    return sorted(int(i) for i in idx.tolist())


def plot_step4_type_logs_panel(
    *,
    cell_type_logs_npz: Path,
    cell_tops_npz: Optional[Path],
    out_png: Path,
    max_cells: int = 12,
    seed: int = 42,
    tops_level: int = 0,
    title: str = "Step 4 — Cell type logs (sample)",
) -> None:
    cell_ids, rgt, tlog = _load_cell_logs(cell_type_logs_npz)
    tops_by_cell = _load_tops(cell_tops_npz)

    idx = _pick_cells(cell_ids, max_cells=int(max_cells), seed=int(seed))
    n = len(idx)
    if n == 0:
        raise ValueError("No cells available to plot.")

    ncols = 4 if n >= 4 else n
    nrows = int(np.ceil(n / max(1, ncols)))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.8 * nrows), sharex=False, sharey=False)
    if nrows == 1 and ncols == 1:
        axes = np.asarray([axes])
    axes = np.asarray(axes).reshape(-1)

    for ax_i, cell_idx in enumerate(idx):
        ax = axes[ax_i]
        cid = str(cell_ids[cell_idx])
        rr = np.asarray(rgt[cell_idx], dtype="float64")
        yy = np.asarray(tlog[cell_idx], dtype="float64")
        m = np.isfinite(rr) & np.isfinite(yy)
        if int(np.count_nonzero(m)) >= 2:
            ax.plot(yy[m], rr[m], color="#111111", lw=1.0)

        # overlay tops (level)
        if tops_by_cell is not None and cell_idx < tops_by_cell.size:
            tops_cell = tops_by_cell[cell_idx]
            try:
                tops_lvl = tops_cell[int(tops_level)]
            except Exception:
                tops_lvl = []
            for ti in tops_lvl or []:
                ti = int(ti)
                if 0 <= ti < rr.size and np.isfinite(rr[ti]):
                    ax.axhline(rr[ti], color="#cc3311", lw=0.6, alpha=0.5)

        ax.set_title(cid, fontsize=8)
        ax.set_xlabel("Type log (norm)")
        ax.set_ylabel("RGT")
        ax.invert_yaxis()

    # Hide unused axes
    for k in range(n, len(axes)):
        axes[k].axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_png = Path(out_png)
    ensure_dir(out_png.parent)
    fig.savefig(out_png, dpi=175)
    plt.close(fig)


def _cell_scalar_from_interval_stats(
    stats_csv: Path,
    *,
    stat: str,
    level: int,
) -> Dict[str, float]:
    _require_pd()
    df = pd.read_csv(stats_csv)
    if df.empty:
        return {}

    if "cell_id" not in df.columns:
        raise ValueError(f"cell_interval_stats.csv missing cell_id column: {stats_csv}")

    if "level" in df.columns:
        df = df[df["level"].astype(int) == int(level)]
    if stat not in df.columns:
        raise ValueError(f"cell_interval_stats.csv missing stat column '{stat}': {stats_csv}")

    df[stat] = pd.to_numeric(df[stat], errors="coerce")
    grp = df.groupby("cell_id")[stat].mean(numeric_only=True)
    return {str(k): float(v) for k, v in grp.items()}


def _cell_scalar_from_type_log(
    cell_ids: np.ndarray,
    tlog: np.ndarray,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for i, cid in enumerate(cell_ids.tolist()):
        y = np.asarray(tlog[i], dtype="float64")
        m = np.isfinite(y)
        out[str(cid)] = float(np.nanmean(y[m])) if m.any() else float("nan")
    return out


def plot_step4_cell_map(
    *,
    cell_type_logs_npz: Path,
    cell_interval_stats_csv: Optional[Path],
    out_png: Path,
    stat: str = "ntg",
    level: int = 0,
    title: str = "Step 4 — Cell map",
) -> None:
    cell_ids, _rgt, tlog = _load_cell_logs(cell_type_logs_npz)

    values: Dict[str, float] = {}
    if cell_interval_stats_csv and Path(cell_interval_stats_csv).exists():
        try:
            values = _cell_scalar_from_interval_stats(Path(cell_interval_stats_csv), stat=str(stat), level=int(level))
        except Exception:
            values = {}

    if not values:
        values = _cell_scalar_from_type_log(cell_ids, tlog)

    # Build grid in i/j space
    ij: List[Tuple[int, int]] = []
    vals: List[float] = []
    for cid in cell_ids.tolist():
        try:
            i, j = cell_id_to_ij(str(cid))
        except Exception:
            continue
        ij.append((int(i), int(j)))
        vals.append(float(values.get(str(cid), np.nan)))

    if not ij:
        raise ValueError("No valid cell IDs for map plotting.")

    ii = np.array([p[0] for p in ij], dtype=int)
    jj = np.array([p[1] for p in ij], dtype=int)

    i0, i1 = int(np.min(ii)), int(np.max(ii))
    j0, j1 = int(np.min(jj)), int(np.max(jj))

    grid = np.full((j1 - j0 + 1, i1 - i0 + 1), np.nan, dtype="float64")
    for (i, j), v in zip(ij, vals):
        grid[j - j0, i - i0] = float(v)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(grid, origin="lower", aspect="equal")
    ax.set_title(f"{title} — {stat} (level {int(level)})")
    ax.set_xlabel("Grid i")
    ax.set_ylabel("Grid j")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(stat)

    fig.tight_layout()
    out_png = Path(out_png)
    ensure_dir(out_png.parent)
    fig.savefig(out_png, dpi=175)
    plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Step 4 visualizations.")
    p.add_argument("--cell-type-logs-npz", required=True)
    p.add_argument("--cell-tops-npz", default="")
    p.add_argument("--cell-interval-stats-csv", default="")
    p.add_argument("--out-dir", required=True)

    p.add_argument("--max-cells", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tops-level", type=int, default=0)

    p.add_argument("--map-stat", type=str, default="ntg")
    p.add_argument("--map-level", type=int, default=0)

    p.add_argument("--no-panel", action="store_true")
    p.add_argument("--no-map", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    cell_logs = Path(args.cell_type_logs_npz)
    tops_npz = Path(args.cell_tops_npz) if str(args.cell_tops_npz).strip() else None
    stats_csv = Path(args.cell_interval_stats_csv) if str(args.cell_interval_stats_csv).strip() else None

    if not bool(args.no_panel):
        plot_step4_type_logs_panel(
            cell_type_logs_npz=cell_logs,
            cell_tops_npz=tops_npz,
            out_png=out_dir / "step4_type_logs_panel.png",
            max_cells=int(args.max_cells),
            seed=int(args.seed),
            tops_level=int(args.tops_level),
        )

    if not bool(args.no_map):
        plot_step4_cell_map(
            cell_type_logs_npz=cell_logs,
            cell_interval_stats_csv=stats_csv,
            out_png=out_dir / "step4_cell_map.png",
            stat=str(args.map_stat),
            level=int(args.map_level),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
