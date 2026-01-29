# src/strataframe/steps/step2_pairwise_correlation_dtw.py
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from strataframe.graph.las_utils import (
    LasReadError,
    read_las_curve_resampled_ascii,
    read_las_header_only,
    dtw_cost_and_path,
)


@dataclass(frozen=True)
class Step2DtwConfig:
    curve_mnemonic: str = "GR"
    n_samples: int = 512
    p_lo: float = 1.0
    p_hi: float = 99.0
    min_finite: int = 10
    alpha: float = 0.15

    base_pad_ft: float = 10.0
    pad_slope_ft_per_km: float = 0.0
    max_pad_ft: float = 200.0
    min_overlap_ft: float = 50.0

    guard_frac: float = 0.1
    guard_min_samples: int = 16
    guard_min_slope: float = 0.5
    guard_max_slope: float = 2.0

    max_las_mb: float = 256.0
    max_curves: int = 0
    cache_max_wells: int = 8
    progress_every: int = 500
    gc_every: int = 200
    max_edges: int = 0  # 0 means no limit
    max_rows: int = 0
    gr_vectors_npz: Optional[str] = None
    cache_only: bool = False


@dataclass(frozen=True)
class Step2DtwPaths:
    out_dir: Path

    @property
    def dtw_edges_csv(self) -> Path:
        return self.out_dir / "dtw_edges.csv"

    @property
    def dtw_paths_jsonl(self) -> Path:
        return self.out_dir / "dtw_paths.jsonl"

    @property
    def diagnostics_json(self) -> Path:
        return self.out_dir / "diagnostics.json"

    @property
    def manifest_json(self) -> Path:
        return self.out_dir / "manifest.json"


def _edge_key(i: int, j: int) -> Tuple[int, int]:
    return (i, j) if i < j else (j, i)


class _WellCache:
    def __init__(self, max_wells: int) -> None:
        self.max_wells = int(max(0, max_wells))
        self._cache: "OrderedDict[Any, np.ndarray]" = OrderedDict()

    def get(self, key: Any) -> Optional[np.ndarray]:
        if self.max_wells <= 0:
            return None
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: Any, value: np.ndarray) -> None:
        if self.max_wells <= 0:
            return
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_wells:
            self._cache.popitem(last=False)


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _compute_overlap(
    zmin1: float, zmax1: float, zmin2: float, zmax2: float, pad_ft: float
) -> Tuple[float, float, float, float]:
    overlap_min = max(zmin1, zmin2)
    overlap_max = min(zmax1, zmax2)
    overlap_ft = max(0.0, overlap_max - overlap_min)

    win_min = overlap_min - float(pad_ft)
    win_max = overlap_max + float(pad_ft)

    len1 = max(0.0, zmax1 - zmin1)
    len2 = max(0.0, zmax2 - zmin2)
    denom = min(len1, len2) if min(len1, len2) > 0 else 0.0
    overlap_frac = float(overlap_ft / denom) if denom > 0 else 0.0
    return win_min, win_max, overlap_ft, overlap_frac


def _guard_slope_ok(path: np.ndarray, n_samples: int, cfg: Step2DtwConfig) -> bool:
    if path.ndim != 2 or path.shape[1] != 2:
        return True
    n = int(n_samples)
    if n <= 0:
        return True
    k = max(int(cfg.guard_min_samples), int(cfg.guard_frac * n))
    k = min(k, path.shape[0])
    if k < 2:
        return True

    def _slope(seg: np.ndarray) -> Optional[float]:
        i0, j0 = seg[0]
        i1, j1 = seg[-1]
        di = int(i1) - int(i0)
        dj = int(j1) - int(j0)
        if di == 0:
            return None
        return float(dj / di)

    start_seg = path[:k]
    end_seg = path[-k:]
    s1 = _slope(start_seg)
    s2 = _slope(end_seg)

    def _ok(s: Optional[float]) -> bool:
        if s is None:
            return True
        return (s >= float(cfg.guard_min_slope)) and (s <= float(cfg.guard_max_slope))

    return _ok(s1) and _ok(s2)


def _cache_key(node_id: int, win_min: float, win_max: float) -> Tuple[int, float, float]:
    return (int(node_id), round(float(win_min), 3), round(float(win_max), 3))

def _load_curve_resampled_windowed(
    node_id: int,
    las_path: Path,
    *,
    win_min: float,
    win_max: float,
    cfg: Step2DtwConfig,
    cache: _WellCache,
    header_cache: Dict[int, int],
) -> np.ndarray:
    key = _cache_key(node_id, win_min, win_max)
    cached = cache.get(key)
    if cached is not None:
        return cached

    if cfg.max_curves and cfg.max_curves > 0:
        if int(node_id) in header_cache:
            n_curves = header_cache[int(node_id)]
        else:
            hdr = read_las_header_only(las_path)
            n_curves = int(len(hdr.get("curves", []) or []))
            header_cache[int(node_id)] = n_curves
        if n_curves > int(cfg.max_curves):
            raise RuntimeError(f"LAS has too many curves: {n_curves} > {int(cfg.max_curves)}")

    if cfg.max_las_mb and cfg.max_las_mb > 0:
        try:
            size_mb = float(os.path.getsize(las_path)) / (1024.0 * 1024.0)
            if size_mb > float(cfg.max_las_mb):
                raise RuntimeError(f"LAS too large: {size_mb:.1f} MB > {cfg.max_las_mb} MB")
        except FileNotFoundError:
            raise
        except Exception:
            # If size check fails, continue to attempt read.
            pass

    x_norm, _, _, _, _ = read_las_curve_resampled_ascii(
        las_path,
        n_samples=int(cfg.n_samples),
        curve_candidates=(str(cfg.curve_mnemonic),),
        p_lo=float(cfg.p_lo),
        p_hi=float(cfg.p_hi),
        min_finite=int(cfg.min_finite),
        max_rows=int(cfg.max_rows),
        window_min=float(win_min),
        window_max=float(win_max),
    )
    cache.put(key, x_norm)
    return x_norm


def _load_gr_vectors_cache(path: Path) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=False)
    required = {"node_id", "z_top", "z_base", "x_norm"}
    if not required.issubset(set(data.files)):
        raise ValueError(f"gr_vectors cache missing required arrays: {required}")
    node_id = data["node_id"].astype("int64", copy=False)
    z_top = data["z_top"].astype("float64", copy=False)
    z_base = data["z_base"].astype("float64", copy=False)
    x_norm = data["x_norm"]
    if x_norm.ndim != 2:
        raise ValueError("x_norm must be 2D (n_wells, n_samples)")
    index = {int(n): i for i, n in enumerate(node_id.tolist())}
    meta = {}
    if "meta_json" in data.files:
        try:
            meta = json.loads(str(data["meta_json"].item()))
        except Exception:
            meta = {}
    return {"node_id": node_id, "z_top": z_top, "z_base": z_base, "x_norm": x_norm, "index": index, "meta": meta}


def _resample_from_cache(
    x_full: np.ndarray,
    z_top: float,
    z_base: float,
    win_min: float,
    win_max: float,
    *,
    n_samples: int,
    p_lo: float,
    p_hi: float,
) -> np.ndarray:
    n_full = int(x_full.shape[0])
    if n_full < 2:
        raise RuntimeError("Cached vector too short")
    z_full = np.linspace(float(z_top), float(z_base), n_full, dtype="float64")
    z_win = np.linspace(float(win_min), float(win_max), int(n_samples), dtype="float64")
    x_win = np.interp(z_win, z_full, x_full, left=float(x_full[0]), right=float(x_full[-1]))
    fin = np.isfinite(x_win)
    if not np.any(fin):
        return np.zeros((int(n_samples),), dtype="float64")
    plo = float(np.percentile(x_win[fin], float(p_lo)))
    phi = float(np.percentile(x_win[fin], float(p_hi)))
    if (not np.isfinite(plo)) or (not np.isfinite(phi)) or (phi <= plo):
        plo = float(np.nanmin(x_win[fin]))
        phi = float(np.nanmax(x_win[fin]))
        if (not np.isfinite(plo)) or (not np.isfinite(phi)) or (phi <= plo):
            return np.zeros((int(n_samples),), dtype="float64")
    x_norm = (x_win - plo) / (phi - plo)
    return np.clip(x_norm, 0.0, 1.0).astype("float64", copy=False)


def run_step2_pairwise_correlation_dtw(
    *,
    nodes_csv: Path,
    edges_csv: Path,
    out_dir: Path,
    cfg: Step2DtwConfig,
    overwrite: bool = False,
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = Step2DtwPaths(out_dir=out_dir)

    if not overwrite and (paths.dtw_edges_csv.exists() or paths.dtw_paths_jsonl.exists()):
        raise FileExistsError(
            f"Step2 outputs already exist under {out_dir}. Use overwrite=true or pick a new output dir."
        )

    nodes = pd.read_csv(nodes_csv)
    edges = pd.read_csv(edges_csv)

    if "node_id" not in nodes.columns:
        raise ValueError("graph_nodes.csv missing node_id column")
    if "las_path" not in nodes.columns:
        raise ValueError("graph_nodes.csv missing las_path column")

    nodes["node_id"] = pd.to_numeric(nodes["node_id"], errors="coerce").astype("Int64")
    nodes = nodes[nodes["node_id"].notna()].copy()
    nodes = nodes.set_index("node_id")

    for col in ("src_id", "dst_id"):
        if col not in edges.columns:
            raise ValueError(f"graph_edges.csv missing {col}")
        edges[col] = pd.to_numeric(edges[col], errors="coerce").astype("Int64")
    edges = edges[edges["src_id"].notna() & edges["dst_id"].notna()].copy()

    if cfg.max_edges and cfg.max_edges > 0 and len(edges) > int(cfg.max_edges):
        edges = edges.sample(n=int(cfg.max_edges), random_state=42).reset_index(drop=True)

    # Optional gr_vectors cache
    cache_data: Optional[Dict[str, Any]] = None
    if cfg.gr_vectors_npz:
        cache_data = _load_gr_vectors_cache(Path(cfg.gr_vectors_npz))

    # Stream writers
    edges_f = paths.dtw_edges_csv.open("w", encoding="utf-8", newline="")
    edges_cols = [
        "src_id",
        "dst_id",
        "dist_km",
        "pad_ft",
        "overlap_ft",
        "overlap_frac",
        "status",
        "dtw_cost",
        "dtw_cost_per_step",
    ]
    edges_f.write(",".join(edges_cols) + "\n")

    paths_f = paths.dtw_paths_jsonl.open("w", encoding="utf-8")

    cache = _WellCache(int(cfg.cache_max_wells))
    header_cache: Dict[int, int] = {}

    counts = {
        "n_edges": int(len(edges)),
        "n_ok": 0,
        "n_no_overlap": 0,
        "n_overlap_small": 0,
        "n_no_depth": 0,
        "n_read_fail": 0,
        "n_curve_fail": 0,
        "n_dtw_fail": 0,
        "n_guard_fail": 0,
    }

    def _write_edge_row(row: Dict[str, Any]) -> None:
        out = [row.get(c, "") for c in edges_cols]
        edges_f.write(",".join("" if v is None else str(v) for v in out) + "\n")

    for idx, e in edges.iterrows():
        src_id = int(e["src_id"])
        dst_id = int(e["dst_id"])
        dist_km = _safe_float(e.get("dist_km")) or 0.0

        if src_id not in nodes.index or dst_id not in nodes.index:
            _write_edge_row(
                {
                    "src_id": src_id,
                    "dst_id": dst_id,
                    "dist_km": dist_km,
                    "pad_ft": "",
                    "overlap_ft": "",
                    "overlap_frac": "",
                    "status": "missing_node",
                    "dtw_cost": "",
                    "dtw_cost_per_step": "",
                }
            )
            continue

        n1 = nodes.loc[src_id]
        n2 = nodes.loc[dst_id]

        zmin1 = _safe_float(n1.get("depth_min"))
        zmax1 = _safe_float(n1.get("depth_max"))
        zmin2 = _safe_float(n2.get("depth_min"))
        zmax2 = _safe_float(n2.get("depth_max"))
        if zmin1 is None or zmax1 is None or zmin2 is None or zmax2 is None:
            counts["n_no_depth"] += 1
            _write_edge_row(
                {
                    "src_id": src_id,
                    "dst_id": dst_id,
                    "dist_km": dist_km,
                    "pad_ft": "",
                    "overlap_ft": "",
                    "overlap_frac": "",
                    "status": "no_depth",
                    "dtw_cost": "",
                    "dtw_cost_per_step": "",
                }
            )
            continue

        pad_ft = float(cfg.base_pad_ft) + float(cfg.pad_slope_ft_per_km) * float(dist_km)
        pad_ft = max(0.0, min(float(cfg.max_pad_ft), pad_ft))

        win_min, win_max, overlap_ft, overlap_frac = _compute_overlap(
            zmin1, zmax1, zmin2, zmax2, pad_ft
        )

        if overlap_ft <= 0.0:
            counts["n_no_overlap"] += 1
            _write_edge_row(
                {
                    "src_id": src_id,
                    "dst_id": dst_id,
                    "dist_km": dist_km,
                    "pad_ft": pad_ft,
                    "overlap_ft": overlap_ft,
                    "overlap_frac": overlap_frac,
                    "status": "no_overlap",
                    "dtw_cost": "",
                    "dtw_cost_per_step": "",
                }
            )
            continue

        if overlap_ft < float(cfg.min_overlap_ft):
            counts["n_overlap_small"] += 1
            _write_edge_row(
                {
                    "src_id": src_id,
                    "dst_id": dst_id,
                    "dist_km": dist_km,
                    "pad_ft": pad_ft,
                    "overlap_ft": overlap_ft,
                    "overlap_frac": overlap_frac,
                    "status": "overlap_too_small",
                    "dtw_cost": "",
                    "dtw_cost_per_step": "",
                }
            )
            continue

        try:
            if cache_data is not None:
                idx_map = cache_data["index"]
                if (src_id in idx_map) and (dst_id in idx_map):
                    i1 = idx_map[src_id]
                    i2 = idx_map[dst_id]
                    x1 = _resample_from_cache(
                        cache_data["x_norm"][i1],
                        cache_data["z_top"][i1],
                        cache_data["z_base"][i1],
                        win_min,
                        win_max,
                        n_samples=int(cfg.n_samples),
                        p_lo=float(cfg.p_lo),
                        p_hi=float(cfg.p_hi),
                    )
                    x2 = _resample_from_cache(
                        cache_data["x_norm"][i2],
                        cache_data["z_top"][i2],
                        cache_data["z_base"][i2],
                        win_min,
                        win_max,
                        n_samples=int(cfg.n_samples),
                        p_lo=float(cfg.p_lo),
                        p_hi=float(cfg.p_hi),
                    )
                else:
                    if bool(cfg.cache_only):
                        counts["n_read_fail"] += 1
                        _write_edge_row(
                            {
                                "src_id": src_id,
                                "dst_id": dst_id,
                                "dist_km": dist_km,
                                "pad_ft": pad_ft,
                                "overlap_ft": overlap_ft,
                                "overlap_frac": overlap_frac,
                                "status": "cache_miss",
                                "dtw_cost": "",
                                "dtw_cost_per_step": "",
                            }
                        )
                        continue
                    x1 = _load_curve_resampled_windowed(
                        src_id,
                        Path(n1["las_path"]),
                        win_min=win_min,
                        win_max=win_max,
                        cfg=cfg,
                        cache=cache,
                        header_cache=header_cache,
                    )
                    x2 = _load_curve_resampled_windowed(
                        dst_id,
                        Path(n2["las_path"]),
                        win_min=win_min,
                        win_max=win_max,
                        cfg=cfg,
                        cache=cache,
                        header_cache=header_cache,
                    )
            else:
                x1 = _load_curve_resampled_windowed(
                    src_id,
                    Path(n1["las_path"]),
                    win_min=win_min,
                    win_max=win_max,
                    cfg=cfg,
                    cache=cache,
                    header_cache=header_cache,
                )
                x2 = _load_curve_resampled_windowed(
                    dst_id,
                    Path(n2["las_path"]),
                    win_min=win_min,
                    win_max=win_max,
                    cfg=cfg,
                    cache=cache,
                    header_cache=header_cache,
                )
        except (FileNotFoundError, LasReadError, RuntimeError, ValueError) as ex:
            counts["n_read_fail"] += 1
            _write_edge_row(
                {
                    "src_id": src_id,
                    "dst_id": dst_id,
                    "dist_km": dist_km,
                    "pad_ft": pad_ft,
                    "overlap_ft": overlap_ft,
                    "overlap_frac": overlap_frac,
                    "status": "read_fail",
                    "dtw_cost": "",
                    "dtw_cost_per_step": "",
                }
            )
            continue

        try:
            cost_total, cost_per_step, path = dtw_cost_and_path(
                x1, x2, alpha=float(cfg.alpha), backtrack=True
            )
        except Exception:
            counts["n_dtw_fail"] += 1
            _write_edge_row(
                {
                    "src_id": src_id,
                    "dst_id": dst_id,
                    "dist_km": dist_km,
                    "pad_ft": pad_ft,
                    "overlap_ft": overlap_ft,
                    "overlap_frac": overlap_frac,
                    "status": "dtw_fail",
                    "dtw_cost": "",
                    "dtw_cost_per_step": "",
                }
            )
            continue

        if path is not None and (not _guard_slope_ok(path, int(cfg.n_samples), cfg)):
            counts["n_guard_fail"] += 1
            _write_edge_row(
                {
                    "src_id": src_id,
                    "dst_id": dst_id,
                    "dist_km": dist_km,
                    "pad_ft": pad_ft,
                    "overlap_ft": overlap_ft,
                    "overlap_frac": overlap_frac,
                    "status": "end_mismatch",
                    "dtw_cost": cost_total,
                    "dtw_cost_per_step": cost_per_step,
                }
            )
            # Still write path for diagnostics
        else:
            counts["n_ok"] += 1
            _write_edge_row(
                {
                    "src_id": src_id,
                    "dst_id": dst_id,
                    "dist_km": dist_km,
                    "pad_ft": pad_ft,
                    "overlap_ft": overlap_ft,
                    "overlap_frac": overlap_frac,
                    "status": "ok",
                    "dtw_cost": cost_total,
                    "dtw_cost_per_step": cost_per_step,
                }
            )

        if path is not None:
            paths_f.write(
                json.dumps(
                    {
                        "src_id": src_id,
                        "dst_id": dst_id,
                        "n_samples": int(cfg.n_samples),
                        "path": path.tolist(),
                    }
                )
                + "\n"
            )

        if cfg.progress_every and (idx + 1) % int(cfg.progress_every) == 0:
            print(f"[step2] processed {idx+1}/{len(edges)} edges ok={counts['n_ok']}")
        if cfg.gc_every and (idx + 1) % int(cfg.gc_every) == 0:
            gc.collect()

    edges_f.flush()
    edges_f.close()
    paths_f.flush()
    paths_f.close()

    diag = {"counts": counts, "config": asdict(cfg)}
    paths.diagnostics_json.write_text(json.dumps(diag, indent=2), encoding="utf-8")
    paths.manifest_json.write_text(
        json.dumps(
            {
                "step": "step2_pairwise_correlation_dtw",
                "inputs": {"nodes_csv": str(nodes_csv), "edges_csv": str(edges_csv)},
                "outputs": {
                    "dtw_edges_csv": str(paths.dtw_edges_csv),
                    "dtw_paths_jsonl": str(paths.dtw_paths_jsonl),
                },
                "config": asdict(cfg),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return diag


def main() -> None:
    ap = argparse.ArgumentParser(description="Step2: depth-windowed DTW over graph edges.")
    ap.add_argument("--nodes-csv", required=True, help="graph_nodes.csv")
    ap.add_argument("--edges-csv", required=True, help="graph_edges.csv")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--overwrite", action="store_true")

    ap.add_argument("--curve-mnemonic", default="GR")
    ap.add_argument("--n-samples", type=int, default=512)
    ap.add_argument("--p-lo", type=float, default=1.0)
    ap.add_argument("--p-hi", type=float, default=99.0)
    ap.add_argument("--min-finite", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.15)

    ap.add_argument("--base-pad-ft", type=float, default=10.0)
    ap.add_argument("--pad-slope-ft-per-km", type=float, default=0.0)
    ap.add_argument("--max-pad-ft", type=float, default=200.0)
    ap.add_argument("--min-overlap-ft", type=float, default=50.0)

    ap.add_argument("--guard-frac", type=float, default=0.1)
    ap.add_argument("--guard-min-samples", type=int, default=16)
    ap.add_argument("--guard-min-slope", type=float, default=0.5)
    ap.add_argument("--guard-max-slope", type=float, default=2.0)

    ap.add_argument("--max-las-mb", type=float, default=256.0)
    ap.add_argument("--cache-max-wells", type=int, default=8)
    ap.add_argument("--progress-every", type=int, default=500)
    ap.add_argument("--gc-every", type=int, default=200)
    ap.add_argument("--max-edges", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--max-curves", type=int, default=0)
    ap.add_argument("--gr-vectors-npz", type=str, default="")
    ap.add_argument("--cache-only", action="store_true")

    args = ap.parse_args()

    cfg = Step2DtwConfig(
        curve_mnemonic=str(args.curve_mnemonic),
        n_samples=int(args.n_samples),
        p_lo=float(args.p_lo),
        p_hi=float(args.p_hi),
        min_finite=int(args.min_finite),
        alpha=float(args.alpha),
        base_pad_ft=float(args.base_pad_ft),
        pad_slope_ft_per_km=float(args.pad_slope_ft_per_km),
        max_pad_ft=float(args.max_pad_ft),
        min_overlap_ft=float(args.min_overlap_ft),
        guard_frac=float(args.guard_frac),
        guard_min_samples=int(args.guard_min_samples),
        guard_min_slope=float(args.guard_min_slope),
        guard_max_slope=float(args.guard_max_slope),
        max_las_mb=float(args.max_las_mb),
        max_curves=int(args.max_curves),
        cache_max_wells=int(args.cache_max_wells),
        progress_every=int(args.progress_every),
        gc_every=int(args.gc_every),
        max_edges=int(args.max_edges),
        max_rows=int(args.max_rows),
        gr_vectors_npz=str(args.gr_vectors_npz) if str(args.gr_vectors_npz).strip() else None,
        cache_only=bool(args.cache_only),
    )

    run_step2_pairwise_correlation_dtw(
        nodes_csv=Path(args.nodes_csv),
        edges_csv=Path(args.edges_csv),
        out_dir=Path(args.out_dir),
        cfg=cfg,
        overwrite=bool(args.overwrite),
    )


if __name__ == "__main__":
    main()
