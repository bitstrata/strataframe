# src/strataframe/pipelines/step4_type_columns.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import json
import numpy as np

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover
    nx = None  # type: ignore

from strataframe.rgt.chronostrat import ChronostratConfig, build_chronostrat_diagram
from strataframe.rgt.wavelet_tops import CwtConfig, derive_tops_multiscale
from strataframe.rgt.shifts_npz import load_shifts_npz
from strataframe.pipelines.step3_graph_attach_arrays import attach_rep_arrays
from strataframe.pipelines.step3b_rep_arrays import load_rep_arrays_npz
from strataframe.pipelines.step3_io_framework import FrameworkCsvConfig, load_framework_graph
from strataframe.spatial.grid_index import cell_id_to_ij, ij_to_cell_id, kernel_cells_ij
from strataframe.utils.hash_utils import stable_hash32


def _require() -> None:
    if pd is None:
        raise RuntimeError("pandas is required for Step 4. Install with: pip install pandas")
    if nx is None:
        raise RuntimeError("networkx is required for Step 4. Install with: pip install networkx")


@dataclass(frozen=True)
class IntervalStatsConfig:
    ntg_cutoff: float = 0.40
    min_finite: int = 20


@dataclass(frozen=True)
class Step4Config:
    # Inputs
    framework_nodes_csv: Path
    framework_edges_csv: Path
    rep_arrays_npz: Path
    rgt_shifts_npz: Path

    # Cell kernel controls
    kernel_radius: int = 1
    kernel_radius_max: int = 3
    min_wells_per_cell: int = 5
    max_wells_per_cell: int = 200
    seed: int = 42

    # Chronostrat + CWT
    chronostrat: ChronostratConfig = ChronostratConfig()
    cwt: CwtConfig = CwtConfig()

    # Interval stats
    interval_stats: IntervalStatsConfig = IntervalStatsConfig()

    # Column naming (robust heuristics)
    node_id_col: str = "rep_id"
    bin_id_col: str = "bin_id"
    lat_col: str = "lat"
    lon_col: str = "lon"


@dataclass(frozen=True)
class Step4Paths:
    out_dir: Path

    @property
    def cell_type_logs_npz(self) -> Path:
        return self.out_dir / "cell_type_logs.npz"

    @property
    def cell_tops_npz(self) -> Path:
        return self.out_dir / "cell_tops.npz"

    @property
    def cell_interval_stats_csv(self) -> Path:
        return self.out_dir / "cell_interval_stats.csv"

    @property
    def chronostrat_key_json(self) -> Path:
        return self.out_dir / "chronostrat_key.json"

    @property
    def diagnostics_json(self) -> Path:
        return self.out_dir / "step4_diagnostics.json"


def _subsample_nodes(nodes: List[str], *, cell_id: str, max_n: int, seed: int) -> List[str]:
    if max_n <= 0 or len(nodes) <= max_n:
        return nodes
    rng = np.random.default_rng((int(seed) + stable_hash32(cell_id)) & 0xFFFFFFFF)
    idx = rng.choice(len(nodes), size=int(max_n), replace=False)
    return [nodes[int(i)] for i in np.asarray(idx, dtype=int).tolist()]


def _canon_cell_id(cell_id: str) -> Optional[str]:
    try:
        i0, j0 = cell_id_to_ij(cell_id)
        return ij_to_cell_id(int(i0), int(j0))
    except Exception:
        return None


def _collect_kernel_nodes(
    cell_id: str,
    *,
    by_cell_canon: Dict[str, List[str]],
    by_cell_raw: Dict[str, List[str]],
    cfg: Step4Config,
) -> List[str]:
    canon = _canon_cell_id(cell_id)
    # If grid cell id is invalid, fall back to just that cell.
    if canon is None:
        return list(by_cell_raw.get(cell_id, []))

    i0, j0 = cell_id_to_ij(canon)

    r0 = int(cfg.kernel_radius)
    rmax = int(max(cfg.kernel_radius_max, r0))
    min_n = int(cfg.min_wells_per_cell)

    nodes: List[str] = []
    for r in range(r0, rmax + 1):
        nodes = []
        for (i, j) in kernel_cells_ij(i0, j0, radius=int(r)):
            cid = ij_to_cell_id(int(i), int(j))
            nodes.extend(by_cell_canon.get(cid, []))
        # de-dup while preserving order
        if nodes:
            seen: set[str] = set()
            nodes = [n for n in nodes if not (n in seen or seen.add(n))]
        if len(nodes) >= min_n or r == rmax:
            break

    nodes = _subsample_nodes(nodes, cell_id=cell_id, max_n=int(cfg.max_wells_per_cell), seed=int(cfg.seed))
    if not nodes:
        # Fallback to raw cell id if canonical kernel lookup failed
        nodes = list(by_cell_raw.get(cell_id, []))
    return nodes


def _interval_rows_for_cell(
    *,
    cell_id: str,
    rgt_grid: np.ndarray,
    type_log: np.ndarray,
    tops_by_level: Sequence[Sequence[int]],
    cfg: IntervalStatsConfig,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    keys: List[Dict[str, Any]] = []

    rgt_grid = np.asarray(rgt_grid, dtype="float64")
    type_log = np.asarray(type_log, dtype="float64")

    for lvl, tops in enumerate(tops_by_level):
        if tops is None:
            continue
        tops_i = [int(t) for t in tops if int(t) >= 0]
        if len(tops_i) < 2:
            continue

        for k in range(len(tops_i) - 1):
            i0 = int(tops_i[k])
            i1 = int(tops_i[k + 1])
            if i1 < i0:
                i0, i1 = i1, i0
            if i0 < 0 or i1 >= int(type_log.size):
                continue

            seg = type_log[i0 : i1 + 1]
            fin = np.isfinite(seg)
            n_fin = int(np.count_nonzero(fin))

            if n_fin >= int(cfg.min_finite):
                p05 = float(np.nanpercentile(seg[fin], 5))
                p95 = float(np.nanpercentile(seg[fin], 95))
                p25 = float(np.nanpercentile(seg[fin], 25))
                p75 = float(np.nanpercentile(seg[fin], 75))
                mean = float(np.nanmean(seg[fin]))
                std = float(np.nanstd(seg[fin]))
                iqr = float(p75 - p25)
                rng95 = float(p95 - p05)
                ntg = float(np.nanmean((seg[fin] < float(cfg.ntg_cutoff)).astype("float64")))
            else:
                mean = std = iqr = rng95 = ntg = float("nan")

            rgt_top = float(rgt_grid[i0]) if i0 < rgt_grid.size else float("nan")
            rgt_base = float(rgt_grid[i1]) if i1 < rgt_grid.size else float("nan")

            key = f"{cell_id}|L{lvl}|I{k}"
            keys.append(
                {
                    "key": key,
                    "cell_id": cell_id,
                    "level": int(lvl),
                    "interval_idx": int(k),
                    "idx_top": int(i0),
                    "idx_base": int(i1),
                    "rgt_top": rgt_top,
                    "rgt_base": rgt_base,
                }
            )

            rows.append(
                {
                    "cell_id": cell_id,
                    "level": int(lvl),
                    "interval_idx": int(k),
                    "idx_top": int(i0),
                    "idx_base": int(i1),
                    "rgt_top": rgt_top,
                    "rgt_base": rgt_base,
                    "n_finite": int(n_fin),
                    "mean": float(mean),
                    "std": float(std),
                    "iqr": float(iqr),
                    "range95": float(rng95),
                    "ntg": float(ntg),
                }
            )

    return rows, keys


def run_step4_type_columns(
    *,
    out_dir: Path,
    cfg: Step4Config,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Step 4:
      - For each grid cell, build a chronostrat diagram from reps in a kernel
      - Compute type log + multiscale tops + interval stats
    """
    _require()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = Step4Paths(out_dir=out_dir)

    diag: Dict[str, Any] = {
        "step": "step4_type_columns",
        "counts": {
            "n_cells_total": 0,
            "n_cells_ok": 0,
            "n_cells_skipped": 0,
            "n_cells_fail": 0,
        },
        "errors": [],
    }

    # Load graph + arrays + shifts
    fw_cfg = FrameworkCsvConfig(
        node_id_col=str(cfg.node_id_col),
        lat_col=str(cfg.lat_col),
        lon_col=str(cfg.lon_col),
        bin_id_col=str(cfg.bin_id_col),
    )
    G = load_framework_graph(nodes_csv=cfg.framework_nodes_csv, edges_csv=cfg.framework_edges_csv, cfg=fw_cfg)

    rep_arrays = load_rep_arrays_npz(cfg.rep_arrays_npz)
    attach_rep_arrays(G, rep_arrays, log_key="log_rs", depth_key="depth_rs", node_log_key="log_rs", node_depth_key="depth_rs")

    shifts = load_shifts_npz(cfg.rgt_shifts_npz)

    # Group nodes by cell_id (bin_id), keeping raw ids but indexing kernels by canonical ids
    by_cell_raw: Dict[str, List[str]] = {}
    by_cell_canon: Dict[str, List[str]] = {}
    for n, nd in G.nodes(data=True):
        cid = str(nd.get("bin_id", "") or "").strip()
        if not cid:
            continue
        by_cell_raw.setdefault(cid, []).append(str(n))
        canon = _canon_cell_id(cid)
        if canon is not None:
            by_cell_canon.setdefault(canon, []).append(str(n))
        else:
            by_cell_canon.setdefault(cid, []).append(str(n))

    cell_ids = sorted(by_cell_raw.keys())
    diag["counts"]["n_cells_total"] = int(len(cell_ids))

    n_rgt = int(getattr(cfg.chronostrat, "n_rgt", 0) or 0)
    n_rgt = max(50, n_rgt)

    rgt_grid_mat = np.full((len(cell_ids), n_rgt), np.nan, dtype="float64")
    type_log_mat = np.full((len(cell_ids), n_rgt), np.nan, dtype="float64")
    reversal_frac = np.full((len(cell_ids),), np.nan, dtype="float64")

    tops_by_cell: List[Any] = []
    interval_rows: List[Dict[str, Any]] = []
    key_rows: List[Dict[str, Any]] = []

    for i, cell_id in enumerate(cell_ids):
        nodes = _collect_kernel_nodes(
            cell_id,
            by_cell_canon=by_cell_canon,
            by_cell_raw=by_cell_raw,
            cfg=cfg,
        )
        if not nodes:
            diag["counts"]["n_cells_skipped"] += 1
            tops_by_cell.append([])
            continue

        H = G.subgraph(nodes).copy()
        try:
            diag_cell = build_chronostrat_diagram(
                H,
                shifts,
                cfg=cfg.chronostrat,
                z_key="depth_rs",
                log_key="log_rs",
            )
        except Exception as e:
            diag["counts"]["n_cells_fail"] += 1
            if len(diag["errors"]) < 20:
                diag["errors"].append({"cell_id": cell_id, "error": f"{type(e).__name__}: {e}"})
            tops_by_cell.append([])
            continue

        diag["counts"]["n_cells_ok"] += 1

        rgt_grid = np.asarray(diag_cell["rgt_grid"], dtype="float64")
        type_log = np.asarray(diag_cell["type_log"], dtype="float64")

        if rgt_grid.size != n_rgt:
            # Defensive: pad or truncate to configured length
            rr = np.full((n_rgt,), np.nan, dtype="float64")
            tt = np.full((n_rgt,), np.nan, dtype="float64")
            m = int(min(n_rgt, rgt_grid.size, type_log.size))
            rr[:m] = rgt_grid[:m]
            tt[:m] = type_log[:m]
            rgt_grid = rr
            type_log = tt

        rgt_grid_mat[i, :] = rgt_grid
        type_log_mat[i, :] = type_log

        rev = np.asarray(diag_cell.get("reversal_frac", []), dtype="float64")
        reversal_frac[i] = float(np.nanmean(rev)) if rev.size else float("nan")

        tops = derive_tops_multiscale(type_log, cfg=cfg.cwt)
        tops_by_cell.append(tops.get("tops_by_level", []))

        rows, keys = _interval_rows_for_cell(
            cell_id=cell_id,
            rgt_grid=rgt_grid,
            type_log=type_log,
            tops_by_level=tops.get("tops_by_level", []),
            cfg=cfg.interval_stats,
        )
        interval_rows.extend(rows)
        key_rows.extend(keys)

    # Save cell type logs
    np.savez_compressed(
        paths.cell_type_logs_npz,
        cell_ids=np.asarray(cell_ids, dtype=np.str_),
        rgt_grid=rgt_grid_mat,
        type_log=type_log_mat,
        reversal_frac=reversal_frac,
    )

    # Save tops (object array for variable per-cell lengths)
    np.savez_compressed(
        paths.cell_tops_npz,
        cell_ids=np.asarray(cell_ids, dtype=np.str_),
        widths=np.asarray(getattr(cfg.cwt, "widths", []), dtype=np.int32),
        tops_by_level=np.asarray(tops_by_cell, dtype=object),
    )

    # Save interval stats CSV
    if interval_rows:
        pd.DataFrame(interval_rows).to_csv(paths.cell_interval_stats_csv, index=False)
    else:
        pd.DataFrame(columns=["cell_id", "level", "interval_idx"]).to_csv(paths.cell_interval_stats_csv, index=False)

    # Save chronostrat key
    paths.chronostrat_key_json.write_text(json.dumps(key_rows, indent=2), encoding="utf-8")

    # Diagnostics
    paths.diagnostics_json.write_text(json.dumps(diag, indent=2), encoding="utf-8")

    return diag
