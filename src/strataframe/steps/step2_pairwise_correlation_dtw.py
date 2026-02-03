# src/strataframe/steps/step2_pairwise_correlation_dtw.py
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from collections import OrderedDict
import heapq
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from strataframe.graph.las_utils import (
    LasReadError,
    read_las_curve_resampled_ascii,
    read_las_header_only,
    dtw_cost_and_path,
)
from strataframe.chronolog.dtw import (
    compute_overlap_window,
    load_gr_vectors_cache,
    pad_for_distance,
    resample_from_cache,
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
    band_max_frac: float = 0.2
    band_min_samples: int = 16
    nan_cost: float = 1.0e6
    imputed_penalty: float = 0.35
    scan_windows: bool = True
    scan_scales: Tuple[float, ...] = (0.7, 0.85, 1.0, 1.15, 1.3, 1.5)
    scan_stride_frac: float = 0.25
    scan_top_k: int = 4
    scan_downsample: int = 128
    scan_min_finite_frac: float = 0.9
    scan_min_samples: int = 16
    scan_corr_min: float = 0.0
    scan_min_len_frac: float = 0.85
    scan_min_len_frac_imputed: float = 0.95
    scan_imputed_frac: float = 0.2
    scan_window_max_imputed_frac: float = 0.95

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


def _longest_finite_segment(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    if mask.size == 0:
        return None
    best_start = 0
    best_len = 0
    cur_start: Optional[int] = None
    for i, m in enumerate(mask.tolist()):
        if m and cur_start is None:
            cur_start = int(i)
        elif (not m) and cur_start is not None:
            cur_len = int(i) - int(cur_start)
            if cur_len > best_len:
                best_len = cur_len
                best_start = int(cur_start)
            cur_start = None
    if cur_start is not None:
        cur_len = int(mask.size) - int(cur_start)
        if cur_len > best_len:
            best_len = cur_len
            best_start = int(cur_start)
    if best_len < 2:
        return None
    return int(best_start), int(best_start + best_len)


def _fill_nan_linear(x: np.ndarray) -> Optional[np.ndarray]:
    x = np.asarray(x, dtype="float64").reshape(-1)
    if np.all(np.isfinite(x)):
        return x
    idx = np.arange(x.size, dtype="float64")
    m = np.isfinite(x)
    if int(m.sum()) < 2:
        return None
    return np.interp(idx, idx[m], x[m]).astype("float64", copy=False)


def _resample_to_n(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype="float64").reshape(-1)
    n = int(n)
    if n <= 0:
        return x
    if x.size == n:
        return x
    t_src = np.linspace(0.0, 1.0, x.size, dtype="float64")
    t_dst = np.linspace(0.0, 1.0, n, dtype="float64")
    return np.interp(t_dst, t_src, x).astype("float64", copy=False)


def _corrcoef_fast(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    a = np.asarray(a, dtype="float64").reshape(-1)
    b = np.asarray(b, dtype="float64").reshape(-1)
    if a.size != b.size or a.size < 2:
        return None
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa <= 1.0e-12 or sb <= 1.0e-12:
        return None
    return float(np.dot(a, b) / (float(a.size) * sa * sb))


def _scan_long_for_short(
    *,
    x_short: np.ndarray,
    x_long: np.ndarray,
    long_mask: np.ndarray,
    long_imputed: Optional[np.ndarray],
    min_len_frac: float,
    max_imputed_frac: float,
    cfg: Step2DtwConfig,
) -> List[Tuple[int, int, float]]:
    n_short = int(x_short.size)
    if n_short < int(cfg.scan_min_samples):
        return []
    if not cfg.scan_scales:
        return []

    stride = max(1, int(round(float(cfg.scan_stride_frac) * float(n_short))))
    fin = long_mask.astype(np.int32, copy=False)
    imp = None
    if long_imputed is not None:
        imp = np.asarray(long_imputed, dtype=bool).reshape(-1)
    fin_cum = np.concatenate([[0], np.cumsum(fin)], axis=0)
    n_ds = int(min(int(cfg.scan_downsample), n_short))
    if n_ds < 8:
        n_ds = n_short

    x_short_fill = _fill_nan_linear(x_short)
    if x_short_fill is None:
        return []
    x_short_ds = _resample_to_n(x_short_fill, n_ds)

    heap: List[Tuple[float, int, int]] = []
    min_frac = float(cfg.scan_min_finite_frac)
    corr_min = float(cfg.scan_corr_min)
    min_len = int(math.ceil(float(min_len_frac) * float(n_short)))
    min_len = max(int(cfg.scan_min_samples), int(min_len))
    max_imp = float(max_imputed_frac)

    for s in cfg.scan_scales:
        try:
            scale = float(s)
        except Exception:
            continue
        if scale <= 0.0:
            continue
        Lw = int(round(float(n_short) * scale))
        if Lw < int(min_len) or Lw > int(x_long.size):
            continue

        for start in range(0, int(x_long.size) - Lw + 1, stride):
            fin_cnt = int(fin_cum[start + Lw] - fin_cum[start])
            if fin_cnt < int(math.ceil(min_frac * Lw)):
                continue
            if imp is not None:
                imp_seg = imp[start : start + Lw]
                if imp_seg.size and float(np.mean(imp_seg)) > max_imp:
                    continue
            seg = x_long[start : start + Lw]
            seg_fill = _fill_nan_linear(seg)
            if seg_fill is None:
                continue
            seg_ds = _resample_to_n(seg_fill, n_ds)
            corr = _corrcoef_fast(x_short_ds, seg_ds)
            if corr is None:
                continue
            if corr < corr_min:
                continue
            if len(heap) < int(cfg.scan_top_k):
                heapq.heappush(heap, (float(corr), int(start), int(Lw)))
            else:
                if corr > heap[0][0]:
                    heapq.heapreplace(heap, (float(corr), int(start), int(Lw)))

    if not heap:
        return []
    heap.sort(key=lambda t: t[0], reverse=True)
    return [(int(s), int(L), float(c)) for (c, s, L) in heap]


def _scan_window_candidates(
    *,
    x1: np.ndarray,
    x2: np.ndarray,
    imputed1: Optional[np.ndarray],
    imputed2: Optional[np.ndarray],
    cfg: Step2DtwConfig,
) -> List[Tuple[int, int, int, int, float]]:
    min_len_frac = float(cfg.scan_min_len_frac)
    max_imputed = float(cfg.scan_window_max_imputed_frac)
    imp_thresh = float(cfg.scan_imputed_frac)
    if imputed1 is not None:
        imp1 = float(np.mean(np.asarray(imputed1, dtype=bool))) if imputed1.size else 0.0
        if imp1 >= imp_thresh:
            min_len_frac = max(min_len_frac, float(cfg.scan_min_len_frac_imputed))
            max_imputed = 1.01
    if imputed2 is not None:
        imp2 = float(np.mean(np.asarray(imputed2, dtype=bool))) if imputed2.size else 0.0
        if imp2 >= imp_thresh:
            min_len_frac = max(min_len_frac, float(cfg.scan_min_len_frac_imputed))
            max_imputed = 1.01

    m1 = np.isfinite(x1)
    m2 = np.isfinite(x2)
    seg1 = _longest_finite_segment(m1)
    seg2 = _longest_finite_segment(m2)
    if seg1 is None or seg2 is None:
        return []
    s1, e1 = seg1
    s2, e2 = seg2
    L1 = int(e1 - s1)
    L2 = int(e2 - s2)
    if L1 < int(cfg.scan_min_samples) or L2 < int(cfg.scan_min_samples):
        return []

    if L1 <= L2:
        x_short = x1[s1:e1]
        long_candidates = _scan_long_for_short(
            x_short=x_short,
            x_long=x2,
            long_mask=m2,
            long_imputed=imputed2,
            min_len_frac=min_len_frac,
            max_imputed_frac=max_imputed,
            cfg=cfg,
        )
        return [(int(s1), int(L1), int(s), int(Lw), float(c)) for (s, Lw, c) in long_candidates]

    x_short = x2[s2:e2]
    long_candidates = _scan_long_for_short(
        x_short=x_short,
        x_long=x1,
        long_mask=m1,
        long_imputed=imputed1,
        min_len_frac=min_len_frac,
        max_imputed_frac=max_imputed,
        cfg=cfg,
    )
    return [(int(s), int(Lw), int(s2), int(L2), float(c)) for (s, Lw, c) in long_candidates]

def _mask_outside_valid_range(
    x: np.ndarray,
    *,
    win_min: float,
    win_max: float,
    valid_min: Optional[float],
    valid_max: Optional[float],
) -> np.ndarray:
    if valid_min is None or valid_max is None:
        return x
    vmin = float(valid_min)
    vmax = float(valid_max)
    if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmax <= vmin:
        return x
    z_win = np.linspace(float(win_min), float(win_max), int(x.size), dtype="float64")
    m = (z_win < vmin) | (z_win > vmax)
    if not np.any(m):
        return x
    out = np.asarray(x, dtype="float64").copy()
    out[m] = np.nan
    return out


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

    x_norm, _, _, _, _, z_valid_min, z_valid_max = read_las_curve_resampled_ascii(
        las_path,
        n_samples=int(cfg.n_samples),
        curve_candidates=(str(cfg.curve_mnemonic),),
        p_lo=float(cfg.p_lo),
        p_hi=float(cfg.p_hi),
        min_finite=int(cfg.min_finite),
        max_rows=int(cfg.max_rows),
        window_min=float(win_min),
        window_max=float(win_max),
        return_valid_depths=True,
    )
    x_norm = _mask_outside_valid_range(
        x_norm,
        win_min=float(win_min),
        win_max=float(win_max),
        valid_min=z_valid_min,
        valid_max=z_valid_max,
    )
    cache.put(key, x_norm)
    return x_norm


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


def run_step2_pairwise_correlation_dtw(
    *,
    nodes_csv: Path,
    edges_csv: Path,
    out_dir: Path,
    cfg: Step2DtwConfig,
    overwrite: bool = False,
    exclude_nodes_csv: Optional[Path] = None,
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

    if exclude_nodes_csv is not None and Path(exclude_nodes_csv).exists():
        ex = pd.read_csv(exclude_nodes_csv)
        if "node_id" in ex.columns:
            ex_ids = pd.to_numeric(ex["node_id"], errors="coerce").astype("Int64")
            ex_ids = ex_ids[ex_ids.notna()].astype(int).tolist()
            if ex_ids:
                edges = edges[~edges["src_id"].isin(ex_ids) & ~edges["dst_id"].isin(ex_ids)].copy()

    if cfg.max_edges and cfg.max_edges > 0 and len(edges) > int(cfg.max_edges):
        edges = edges.sample(n=int(cfg.max_edges), random_state=42).reset_index(drop=True)

    # Optional gr_vectors cache
    cache_data: Optional[Dict[str, Any]] = None
    if cfg.gr_vectors_npz:
        cache_data = load_gr_vectors_cache(Path(cfg.gr_vectors_npz))

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
    cache_z_gr_top = None
    cache_z_gr_base = None
    cache_imputed = None
    if cache_data is not None:
        cache_z_gr_top = cache_data.get("z_gr_top", cache_data.get("z_top"))
        cache_z_gr_base = cache_data.get("z_gr_base", cache_data.get("z_base"))
        cache_imputed = cache_data.get("imputed_mask")

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

    for pos, (_, e) in enumerate(edges.iterrows(), start=1):
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

        pad_ft = pad_for_distance(
            base_pad_ft=float(cfg.base_pad_ft),
            pad_slope_ft_per_km=float(cfg.pad_slope_ft_per_km),
            max_pad_ft=float(cfg.max_pad_ft),
            dist_km=float(dist_km),
        )

        win_min, win_max, overlap_ft, overlap_frac = compute_overlap_window(
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

        imputed1 = None
        imputed2 = None
        try:
            if cache_data is not None:
                idx_map = cache_data["index"]
                if (src_id in idx_map) and (dst_id in idx_map):
                    i1 = idx_map[src_id]
                    i2 = idx_map[dst_id]
                    use_mask_outside = cache_imputed is None
                    x1 = resample_from_cache(
                        cache_data["x_norm"][i1],
                        cache_data["z_top"][i1],
                        cache_data["z_base"][i1],
                        win_min,
                        win_max,
                        n_samples=int(cfg.n_samples),
                        p_lo=float(cfg.p_lo),
                        p_hi=float(cfg.p_hi),
                        valid_min=cache_z_gr_top[i1] if cache_z_gr_top is not None else None,
                        valid_max=cache_z_gr_base[i1] if cache_z_gr_base is not None else None,
                        mask_outside=bool(use_mask_outside),
                    )
                    x2 = resample_from_cache(
                        cache_data["x_norm"][i2],
                        cache_data["z_top"][i2],
                        cache_data["z_base"][i2],
                        win_min,
                        win_max,
                        n_samples=int(cfg.n_samples),
                        p_lo=float(cfg.p_lo),
                        p_hi=float(cfg.p_hi),
                        valid_min=cache_z_gr_top[i2] if cache_z_gr_top is not None else None,
                        valid_max=cache_z_gr_base[i2] if cache_z_gr_base is not None else None,
                        mask_outside=bool(use_mask_outside),
                    )
                    if cache_imputed is not None:
                        imputed1 = _resample_mask_from_cache(
                            cache_imputed[i1],
                            z_top=cache_data["z_top"][i1],
                            z_base=cache_data["z_base"][i1],
                            win_min=win_min,
                            win_max=win_max,
                            n_samples=int(cfg.n_samples),
                        )
                        imputed2 = _resample_mask_from_cache(
                            cache_imputed[i2],
                            z_top=cache_data["z_top"][i2],
                            z_base=cache_data["z_base"][i2],
                            win_min=win_min,
                            win_max=win_max,
                            n_samples=int(cfg.n_samples),
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

        def _band_for_len(n1: int, n2: int) -> Optional[int]:
            if float(cfg.band_max_frac) <= 0.0 and int(cfg.band_min_samples) <= 0:
                return None
            band = int(math.ceil(float(cfg.band_max_frac) * float(max(n1, n2))))
            if int(cfg.band_min_samples) > 0:
                band = max(int(cfg.band_min_samples), int(band))
            return int(band)

        cost_total = float("nan")
        cost_per_step = float("nan")
        path = None

        candidates: List[Tuple[int, int, int, int, float]] = []
        if bool(cfg.scan_windows):
            candidates = _scan_window_candidates(x1=x1, x2=x2, imputed1=imputed1, imputed2=imputed2, cfg=cfg)

        tried = 0
        if candidates:
            for s1, L1, s2, L2, _ in candidates:
                try:
                    xs = x1[s1 : s1 + L1]
                    ys = x2[s2 : s2 + L2]
                    imp_x = imputed1[s1 : s1 + L1] if imputed1 is not None else None
                    imp_y = imputed2[s2 : s2 + L2] if imputed2 is not None else None
                    band_rad = _band_for_len(int(xs.size), int(ys.size))
                    ct, cps, p = dtw_cost_and_path(
                        xs,
                        ys,
                        alpha=float(cfg.alpha),
                        backtrack=True,
                        band_rad=band_rad,
                        nan_cost=float(cfg.nan_cost),
                        trim_invalid=True,
                        imputed_x=imp_x,
                        imputed_y=imp_y,
                        imputed_penalty=float(cfg.imputed_penalty),
                    )
                except Exception:
                    continue
                tried += 1
                if p is not None:
                    p = np.asarray(p, dtype="int64")
                    if p.ndim == 2 and p.shape[1] == 2:
                        p[:, 0] += int(s1)
                        p[:, 1] += int(s2)
                if (not np.isfinite(cost_per_step)) or (float(cps) < float(cost_per_step)):
                    cost_total = float(ct)
                    cost_per_step = float(cps)
                    path = p

        if not np.isfinite(cost_per_step):
            try:
                band_rad = _band_for_len(int(x1.size), int(x2.size))
                cost_total, cost_per_step, path = dtw_cost_and_path(
                    x1,
                    x2,
                    alpha=float(cfg.alpha),
                    backtrack=True,
                    band_rad=band_rad,
                    nan_cost=float(cfg.nan_cost),
                    trim_invalid=True,
                    imputed_x=imputed1,
                    imputed_y=imputed2,
                    imputed_penalty=float(cfg.imputed_penalty),
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

        if cfg.progress_every and (pos % int(cfg.progress_every) == 0 or pos == len(edges)):
            print(f"[step2] processed {pos}/{len(edges)} edges ok={counts['n_ok']}")
        if cfg.gc_every and (pos % int(cfg.gc_every) == 0):
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
                "step": "step2d_pairwise_correlation_dtw",
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
    ap = argparse.ArgumentParser(description="Step2d: depth-windowed DTW over graph edges.")
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
    ap.add_argument("--band-max-frac", type=float, default=0.2)
    ap.add_argument("--band-min-samples", type=int, default=16)
    ap.add_argument("--nan-cost", type=float, default=1.0e6)
    ap.add_argument("--imputed-penalty", type=float, default=0.35)
    ap.add_argument("--no-scan-windows", action="store_true", help="Disable sliding-window scan.")
    ap.add_argument("--scan-scales", type=str, default="0.7,0.85,1.0,1.15,1.3,1.5")
    ap.add_argument("--scan-stride-frac", type=float, default=0.25)
    ap.add_argument("--scan-top-k", type=int, default=4)
    ap.add_argument("--scan-downsample", type=int, default=128)
    ap.add_argument("--scan-min-finite-frac", type=float, default=0.9)
    ap.add_argument("--scan-min-samples", type=int, default=16)
    ap.add_argument("--scan-corr-min", type=float, default=0.0)
    ap.add_argument("--scan-min-len-frac", type=float, default=0.85)
    ap.add_argument("--scan-min-len-frac-imputed", type=float, default=0.95)
    ap.add_argument("--scan-imputed-frac", type=float, default=0.2)
    ap.add_argument("--scan-window-max-imputed-frac", type=float, default=0.95)

    ap.add_argument("--max-las-mb", type=float, default=256.0)
    ap.add_argument("--cache-max-wells", type=int, default=8)
    ap.add_argument("--progress-every", type=int, default=500)
    ap.add_argument("--gc-every", type=int, default=200)
    ap.add_argument("--max-edges", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--max-curves", type=int, default=0)
    ap.add_argument("--gr-vectors-npz", type=str, default="")
    ap.add_argument("--cache-only", action="store_true")
    ap.add_argument("--exclude-nodes-csv", type=str, default="", help="Optional list of node_id to exclude.")

    args = ap.parse_args()

    def _parse_scales(s: str) -> Tuple[float, ...]:
        out: List[float] = []
        for tok in str(s).replace(";", ",").split(","):
            t = tok.strip()
            if not t:
                continue
            try:
                out.append(float(t))
            except Exception:
                continue
        return tuple(out)

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
        band_max_frac=float(args.band_max_frac),
        band_min_samples=int(args.band_min_samples),
        nan_cost=float(args.nan_cost),
        imputed_penalty=float(args.imputed_penalty),
        scan_windows=not bool(args.no_scan_windows),
        scan_scales=_parse_scales(args.scan_scales),
        scan_stride_frac=float(args.scan_stride_frac),
        scan_top_k=int(args.scan_top_k),
        scan_downsample=int(args.scan_downsample),
        scan_min_finite_frac=float(args.scan_min_finite_frac),
        scan_min_samples=int(args.scan_min_samples),
        scan_corr_min=float(args.scan_corr_min),
        scan_min_len_frac=float(args.scan_min_len_frac),
        scan_min_len_frac_imputed=float(args.scan_min_len_frac_imputed),
        scan_imputed_frac=float(args.scan_imputed_frac),
        scan_window_max_imputed_frac=float(args.scan_window_max_imputed_frac),
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
        exclude_nodes_csv=Path(args.exclude_nodes_csv) if str(args.exclude_nodes_csv).strip() else None,
    )


if __name__ == "__main__":
    main()
