# src/strataframe/typewell/local_typewell.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from strataframe.graph.las_utils import (
    read_las_normal,
    extract_depth_and_curve,
    resample_and_normalize_curve,
)
from strataframe.graph.select_representatives import resolve_las_path_from_url
from strataframe.spatial.grid import grid_cell_id
from strataframe.typewell.subsequence_dtw import subsequence_dtw


@dataclass(frozen=True)
class TypeWellConfig:
    # template / vectors
    n_template: int = 256
    alpha: float = 0.15

    # kernel behavior
    kernel_radius: int = 1            # 3x3
    kernel_radius_max: int = 3        # adaptive expansion upper bound
    min_kernel_wells: int = 20
    max_kernel_wells: int = 200       # sample cap per kernel

    # QC hard gates (units are depth units from LAS; typically ft)
    qc_min_thickness: float = 50.0
    qc_max_thickness: float = 50_000.0
    qc_min_finite: int = 64
    qc_min_range95: float = 0.08      # on normalized [0,1] vector
    qc_min_iqr: float = 0.04          # on normalized [0,1] vector

    # robust outliers
    z_max: float = 3.5
    shape_z_max: float = 3.5

    # NTG proxy
    ntg_cutoff: float = 0.40

    # placement
    short_frac: float = 0.65          # thickness < short_frac * expected -> attempt placement
    query_n_min: int = 48
    query_n_max: int = 256

    # medoid weights
    w_shape: float = 1.0
    w_thk: float = 0.6
    w_ntg: float = 0.6
    w_iqr: float = 0.3
    w_mean: float = 0.2


def _mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype="float64")
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return float(mad)


def _robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype="float64")
    med = np.nanmedian(x)
    mad = _mad(x)
    if not np.isfinite(mad) or mad <= 1e-12:
        return np.zeros_like(x, dtype="float64")
    return 0.6745 * (x - med) / mad


def _kernel_cells(ix: int, iy: int, r: int) -> List[str]:
    out = []
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            out.append(grid_cell_id(ix + dx, iy + dy))
    return out


def _sample_indices_farthest(lat: np.ndarray, lon: np.ndarray, k: int, seed: int) -> np.ndarray:
    """
    Greedy farthest-point sampling on sphere (haversine).
    Intended only for candidate capping; k <= ~200 typical.
    """
    lat = np.asarray(lat, dtype="float64")
    lon = np.asarray(lon, dtype="float64")
    n = int(lat.size)
    if k <= 0:
        return np.array([], dtype=int)
    if k >= n:
        return np.arange(n, dtype=int)

    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    latr = np.deg2rad(lat)
    lonr = np.deg2rad(lon)

    latc = float(np.mean(latr))
    lonc = float(np.mean(lonr))

    def hav_km(lat1, lon1, lat2, lon2) -> np.ndarray:
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        a = np.clip(a, 0.0, 1.0)
        return 2.0 * 6371.0088 * np.arcsin(np.sqrt(a))

    d0 = hav_km(latr, lonr, latc, lonc)
    mx = float(np.max(d0))
    cand = np.where(np.isclose(d0, mx))[0]
    start = int(rng.choice(cand))

    sel = [start]
    min_d = hav_km(latr, lonr, latr[start], lonr[start])
    for _ in range(1, int(k)):
        nxt = int(np.argmax(min_d))
        sel.append(nxt)
        d_new = hav_km(latr, lonr, latr[nxt], lonr[nxt])
        min_d = np.minimum(min_d, d_new)

    return np.asarray(sel, dtype=int)


def read_gr_features(
    *,
    las_path: Path,
    n: int,
    ntg_cutoff: float,
) -> Dict[str, Any]:
    """
    Read GR from LAS, resample+normalize, and compute basic QC features.
    """
    las = read_las_normal(las_path)
    depth, gr = extract_depth_and_curve(las, curve_mnemonic="GR", depth_preferred=("DEPT", "MD"))

    # basic sampling step
    order = np.argsort(depth)
    d = depth[order]
    step = np.nanmedian(np.diff(d)) if d.size > 3 else np.nan

    x, z_top, z_base = resample_and_normalize_curve(depth, gr, n_samples=int(n))
    fin = np.isfinite(x)
    n_fin = int(np.count_nonzero(fin))

    if n_fin > 0:
        p5 = float(np.nanpercentile(x[fin], 5))
        p95 = float(np.nanpercentile(x[fin], 95))
        iqr = float(np.nanpercentile(x[fin], 75) - np.nanpercentile(x[fin], 25))
        mean = float(np.nanmean(x[fin]))
    else:
        p5, p95, iqr, mean = np.nan, np.nan, np.nan, np.nan

    range95 = p95 - p5 if np.isfinite(p5) and np.isfinite(p95) else np.nan
    thk = float(z_base - z_top)
    ntg = float(np.nanmean((x[fin] < float(ntg_cutoff)).astype("float64"))) if n_fin > 0 else np.nan

    return {
        "x": x.astype("float32", copy=False),
        "z_top": float(z_top),
        "z_base": float(z_base),
        "thickness": float(thk),
        "sample_step": float(step) if np.isfinite(step) else np.nan,
        "n_finite": int(n_fin),
        "p5": float(p5),
        "p95": float(p95),
        "range95": float(range95) if np.isfinite(range95) else np.nan,
        "iqr": float(iqr) if np.isfinite(iqr) else np.nan,
        "mean": float(mean) if np.isfinite(mean) else np.nan,
        "ntg": float(ntg) if np.isfinite(ntg) else np.nan,
    }


def build_cell_typewell(
    *,
    cell_id: str,
    cell_ix: int,
    cell_iy: int,
    kernel_rows: List[Dict[str, str]],
    las_root: Path,
    cfg: TypeWellConfig,
    templates_dir: Path,
    seed: int,
) -> Dict[str, Any]:
    """
    Build a local typewell template for the given cell using wells in its kernel.
    Saves template to templates_dir and returns metadata dict.
    """
    # Candidate cap via farthest-point sampling (uses lat/lon strings)
    lat = np.array([float(r["lat"]) for r in kernel_rows], dtype="float64")
    lon = np.array([float(r["lon"]) for r in kernel_rows], dtype="float64")

    k_cap = int(min(len(kernel_rows), int(cfg.max_kernel_wells)))
    idx = _sample_indices_farthest(lat, lon, k=k_cap, seed=int(seed))
    cand_rows = [kernel_rows[int(i)] for i in idx.tolist()]

    feats = []
    for r in cand_rows:
        url = (r.get("url", "") or "").strip()
        las_path = resolve_las_path_from_url(url, las_root)
        if las_path is None:
            continue
        try:
            f = read_gr_features(las_path=las_path, n=int(cfg.n_template), ntg_cutoff=float(cfg.ntg_cutoff))
        except Exception:
            continue
        f["url"] = url
        f["las_path"] = str(las_path)
        feats.append(f)

    n_read = int(len(feats))
    if n_read <= 0:
        return {
            "cell_id": cell_id,
            "grid_ix": int(cell_ix),
            "grid_iy": int(cell_iy),
            "n_kernel": int(len(kernel_rows)),
            "n_read": 0,
            "n_used": 0,
            "template_path": "",
            "type_url": "",
            "type_las_path": "",
            "error": "no_readable_gr",
        }

    # Hard QC gates
    keep = []
    for f in feats:
        thk = float(f["thickness"])
        if not (cfg.qc_min_thickness <= thk <= cfg.qc_max_thickness):
            continue
        if int(f["n_finite"]) < int(cfg.qc_min_finite):
            continue
        if not np.isfinite(f["range95"]) or float(f["range95"]) < float(cfg.qc_min_range95):
            continue
        if not np.isfinite(f["iqr"]) or float(f["iqr"]) < float(cfg.qc_min_iqr):
            continue
        keep.append(f)

    if len(keep) < 5:
        # too few after QC; return best-effort from raw
        keep = feats

    # Robust outliering on scalar features
    thk = np.array([float(f["thickness"]) for f in keep], dtype="float64")
    ntg = np.array([float(f["ntg"]) for f in keep], dtype="float64")
    iqr = np.array([float(f["iqr"]) for f in keep], dtype="float64")
    mean = np.array([float(f["mean"]) for f in keep], dtype="float64")

    z_thk = np.abs(_robust_z(thk))
    z_ntg = np.abs(_robust_z(ntg))
    z_iqr = np.abs(_robust_z(iqr))
    z_mean = np.abs(_robust_z(mean))

    m0 = (z_thk <= cfg.z_max) & (z_ntg <= cfg.z_max) & (z_iqr <= cfg.z_max) & (z_mean <= cfg.z_max)
    keep2 = [keep[i] for i in range(len(keep)) if bool(m0[i])]
    if len(keep2) < 5:
        keep2 = keep

    X = np.stack([f["x"].astype("float64") for f in keep2], axis=0)  # (n, n_template)

    # Shape outliers: distance to elementwise median vector
    med_vec = np.nanmedian(X, axis=0)
    shape_d = np.nanmean(np.abs(X - med_vec[None, :]), axis=1)
    z_shape = np.abs(_robust_z(shape_d))
    m1 = z_shape <= float(cfg.shape_z_max)
    keep3 = [keep2[i] for i in range(len(keep2)) if bool(m1[i])]
    if len(keep3) < 5:
        keep3 = keep2

    X = np.stack([f["x"].astype("float64") for f in keep3], axis=0)

    # Medoid selection with combined distances
    thk = np.array([float(f["thickness"]) for f in keep3], dtype="float64")
    ntg = np.array([float(f["ntg"]) for f in keep3], dtype="float64")
    iqr = np.array([float(f["iqr"]) for f in keep3], dtype="float64")
    mean = np.array([float(f["mean"]) for f in keep3], dtype="float64")

    thk_s = max(_mad(thk), 1e-6)
    ntg_s = max(_mad(ntg), 1e-6)
    iqr_s = max(_mad(iqr), 1e-6)
    mean_s = max(_mad(mean), 1e-6)

    # Pairwise shape distance (L1 mean)
    n = X.shape[0]
    Dshape = np.zeros((n, n), dtype="float64")
    for i in range(n):
        Dshape[i, :] = np.nanmean(np.abs(X[i, None, :] - X[:, :]), axis=1)

    # Pairwise feature distances
    def pair_abs(a: np.ndarray, s: float) -> np.ndarray:
        return np.abs(a[:, None] - a[None, :]) / float(s)

    D = (
        float(cfg.w_shape) * Dshape
        + float(cfg.w_thk) * pair_abs(thk, thk_s)
        + float(cfg.w_ntg) * pair_abs(ntg, ntg_s)
        + float(cfg.w_iqr) * pair_abs(iqr, iqr_s)
        + float(cfg.w_mean) * pair_abs(mean, mean_s)
    )

    medoid_idx = int(np.argmin(np.sum(D, axis=1)))
    type_f = keep3[medoid_idx]
    template = X[medoid_idx].astype("float32", copy=False)

    templates_dir.mkdir(parents=True, exist_ok=True)
    tpl_path = templates_dir / f"{cell_id}.npz"
    np.savez_compressed(
        tpl_path,
        template=template,
        n_template=int(cfg.n_template),
        cell_id=str(cell_id),
        grid_ix=int(cell_ix),
        grid_iy=int(cell_iy),
    )

    # summary stats (expected thickness, etc.)
    thk_med = float(np.nanmedian(thk))
    ntg_med = float(np.nanmedian(ntg))
    iqr_med = float(np.nanmedian(iqr))

    return {
        "cell_id": cell_id,
        "grid_ix": int(cell_ix),
        "grid_iy": int(cell_iy),
        "n_kernel": int(len(kernel_rows)),
        "n_read": int(n_read),
        "n_used": int(len(keep3)),
        "template_path": str(tpl_path),
        "type_url": str(type_f.get("url", "")),
        "type_las_path": str(type_f.get("las_path", "")),
        "thickness_median": float(thk_med),
        "ntg_median": float(ntg_med),
        "iqr_median": float(iqr_med),
    }


def place_rep_against_template(
    *,
    rep_row: Dict[str, Any],
    cell_meta: Dict[str, Any],
    las_root: Path,
    cfg: TypeWellConfig,
) -> Dict[str, Any]:
    """
    For a representative well, decide whether to attempt placement and compute placement metrics.
    """
    url = (rep_row.get("url", "") or "").strip()
    las_path = resolve_las_path_from_url(url, las_root)
    if las_path is None:
        return {"rep_id": int(rep_row.get("rep_id", 0)), "url": url, "status": "missing_las"}

    # Read rep GR thickness on native interval (do NOT stretch to n_template first)
    try:
        rep_full = read_gr_features(las_path=las_path, n=int(cfg.n_template), ntg_cutoff=float(cfg.ntg_cutoff))
    except Exception as e:
        return {"rep_id": int(rep_row.get("rep_id", 0)), "url": url, "status": "read_fail", "error": str(e)}

    thk = float(rep_full["thickness"])
    expected = float(cell_meta.get("thickness_median", np.nan))
    if not np.isfinite(expected) or expected <= 0:
        expected = thk

    short = bool(np.isfinite(thk) and np.isfinite(expected) and (thk < float(cfg.short_frac) * expected))

    # Load template
    tpl_path = Path(cell_meta.get("template_path", ""))
    if not tpl_path.exists():
        return {"rep_id": int(rep_row.get("rep_id", 0)), "url": url, "status": "missing_template"}

    tpl = np.load(tpl_path, allow_pickle=False)
    y = np.asarray(tpl["template"], dtype="float64")
    m = int(y.size)

    out: Dict[str, Any] = {
        "rep_id": int(rep_row.get("rep_id", 0)),
        "cell_id": str(cell_meta.get("cell_id", "")),
        "url": url,
        "las_path": str(las_path),
        "thickness": float(thk),
        "expected_thickness": float(expected),
        "z_top": float(rep_full["z_top"]),
        "z_base": float(rep_full["z_base"]),
        "short_flag": bool(short),
        "status": "ok",
    }

    if not short:
        # no placement attempt; treat as full interval
        out.update(
            {
                "placed_z_top": float(rep_full["z_top"]),
                "placed_z_base": float(rep_full["z_base"]),
                "missing_top_flag": False,
                "missing_base_flag": False,
                "condensed_flag": False,
                "match_j_start": 0,
                "match_j_end": m - 1,
                "match_s_start": 0.0,
                "match_s_end": 1.0,
                "dtw_cost_per_step": float("nan"),
            }
        )
        return out

    # Build query length proportional to thickness ratio (prevents stretching short intervals)
    ratio = float(np.clip(thk / expected, 0.05, 1.0))
    qn = int(np.clip(int(round(ratio * float(cfg.n_template))), int(cfg.query_n_min), int(cfg.query_n_max)))

    try:
        # re-read with n=qn to avoid stretching
        rep_q = read_gr_features(las_path=las_path, n=qn, ntg_cutoff=float(cfg.ntg_cutoff))
        x = np.asarray(rep_q["x"], dtype="float64")
    except Exception as e:
        out.update({"status": "query_resample_fail", "error": str(e)})
        return out

    # Subsequence DTW
    res = subsequence_dtw(x, y, alpha=float(cfg.alpha))
    j0 = int(res.j_start)
    j1 = int(res.j_end)
    s0 = float(j0 / max(1, m - 1))
    s1 = float(j1 / max(1, m - 1))
    span = float((j1 - j0) / max(1, m - 1))
    warp_ratio = float((j1 - j0 + 1) / max(1, qn))

    # Missing vs condensed heuristic:
    # - condensed if it matches most of the template AND requires pervasive compression
    condensed = bool((span >= 0.85) and (warp_ratio >= 1.25))
    missing_top = bool(not condensed and (s0 > 0.10))
    missing_base = bool(not condensed and (s1 < 0.90))

    miss_top_thk = float(s0 * expected) if missing_top else 0.0
    miss_base_thk = float((1.0 - s1) * expected) if missing_base else 0.0

    placed_top = float(rep_full["z_top"] - miss_top_thk)
    placed_base = float(rep_full["z_base"] + miss_base_thk)

    out.update(
        {
            "placed_z_top": float(placed_top),
            "placed_z_base": float(placed_base),
            "missing_top_flag": bool(missing_top),
            "missing_base_flag": bool(missing_base),
            "condensed_flag": bool(condensed),
            "match_j_start": int(j0),
            "match_j_end": int(j1),
            "match_s_start": float(s0),
            "match_s_end": float(s1),
            "match_span": float(span),
            "warp_ratio": float(warp_ratio),
            "dtw_cost_per_step": float(res.cost_per_step),
        }
    )
    return out
