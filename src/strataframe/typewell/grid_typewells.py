# src/strataframe/typewell/grid_typewells.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from strataframe.graph.select_representatives import resolve_las_path_from_url
from strataframe.spatial.grid_index import (
    GridSpec,
    cell_id_to_ij,
    haversine_km,
    ij_to_cell_id,
    infer_grid_origin,
    kernel_cells_ij,
    latlon_to_xy_m,
    xy_m_to_ij,
)
from strataframe.qc.gr_features import GrFeatureConfig, GrFeatures, compute_gr_features


def _mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype="float64")
    fin = np.isfinite(x)
    if not np.any(fin):
        return float("nan")
    med = float(np.median(x[fin]))
    return float(np.median(np.abs(x[fin] - med)))


def _robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype="float64")
    med = np.nanmedian(x)
    mad = _mad(x)
    if not np.isfinite(mad) or mad <= 1e-12:
        return np.zeros_like(x, dtype="float64")
    return (x - med) / (1.4826 * mad)


@dataclass(frozen=True)
class TypeWellConfig:
    grid_km: float = 10.0
    kernel_radius: int = 1
    kernel_radius_max: int = 3
    n_min_postqc: int = 12
    max_candidates_per_kernel: int = 80

    fallback_nearest_n: int = 40
    fallback_r_max_km: float = 50.0

    use_distance_weights: bool = True
    sigma_km: float = 20.0

    min_finite_frac: float = 0.20
    min_thickness: float = 1.0

    zmax_feature: float = 3.5
    zmax_shape: float = 3.5

    gr_cfg: GrFeatureConfig = GrFeatureConfig()

    w_feat: float = 1.0
    w_shape: float = 2.0


class _FeatureCache:
    def __init__(self, las_root: Path, cfg: TypeWellConfig):
        self.las_root = Path(las_root)
        self.cfg = cfg
        self._cache: Dict[str, GrFeatures] = {}
        self._cache_path: Dict[str, Optional[Path]] = {}

    def las_path_for_url(self, url: str) -> Optional[Path]:
        u = (url or "").strip()
        if u in self._cache_path:
            return self._cache_path[u]
        p = resolve_las_path_from_url(u, self.las_root)
        self._cache_path[u] = p
        return p

    def get(self, url: str) -> GrFeatures:
        u = (url or "").strip()
        if u in self._cache:
            return self._cache[u]
        p = self.las_path_for_url(u)
        if p is None or not p.exists():
            f = GrFeatures(
                status="FAIL",
                error="missing_las",
                z_top=float("nan"),
                z_base=float("nan"),
                thickness=float("nan"),
                n=0,
                finite_frac=0.0,
                p05=float("nan"),
                p50=float("nan"),
                p95=float("nan"),
                iqr=float("nan"),
                rng=float("nan"),
                std=float("nan"),
                sand_frac=float("nan"),
                gr_lowres=None,
            )
            self._cache[u] = f
            return f
        f = compute_gr_features(p, cfg=self.cfg.gr_cfg)
        self._cache[u] = f
        return f


def _hard_qc_pass(f: GrFeatures, *, cfg: TypeWellConfig) -> bool:
    if f.status == "FAIL":
        return False
    if not np.isfinite(f.thickness) or f.thickness <= float(cfg.min_thickness):
        return False
    if not np.isfinite(f.finite_frac) or f.finite_frac < float(cfg.min_finite_frac):
        return False
    if f.gr_lowres is None or np.asarray(f.gr_lowres).size < 8:
        return False
    return True


def _feature_vector(f: GrFeatures) -> np.ndarray:
    return np.asarray(
        [f.thickness, f.p50, f.iqr, f.rng, f.std, f.sand_frac, f.finite_frac],
        dtype="float64",
    )


def _pairwise_L1(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype="float64")
    return np.sum(np.abs(A[:, None, :] - A[None, :, :]), axis=2)


def _distance_weights_km(d_km: np.ndarray, *, sigma_km: float) -> np.ndarray:
    s = float(max(1e-6, sigma_km))
    w = np.exp(-np.power(d_km / s, 2.0))
    return np.clip(w, 1e-6, 1.0)


def _select_medoid(feats: np.ndarray, shapes: np.ndarray, w_i: np.ndarray, *, cfg: TypeWellConfig) -> int:
    Df = _pairwise_L1(feats)
    Ds = _pairwise_L1(shapes)
    D = float(cfg.w_feat) * Df + float(cfg.w_shape) * Ds
    obj = D @ w_i
    return int(np.argmin(obj))


def select_type_wells_grid(
    rows: List[Dict[str, str]],
    *,
    las_root: Path,
    cfg: TypeWellConfig,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Builds one type well per grid cell.

    Output rows match Step-2 representatives schema:
      - 'cell_id' contains the grid cell id
      - 'cell_tag' is a descriptive tag (e.g., 'grid:10km')
      - includes 'las_path' and canonical 'picked_gr'='GR' (others empty)
    """
    lat = np.asarray([float(r.get("lat", "nan")) if (r.get("lat") or "").strip() else np.nan for r in rows], dtype="float64")
    lon = np.asarray([float(r.get("lon", "nan")) if (r.get("lon") or "").strip() else np.nan for r in rows], dtype="float64")
    ok = np.isfinite(lat) & np.isfinite(lon)
    rows_use = [rows[i] for i in np.where(ok)[0].tolist()]
    lat = lat[ok]
    lon = lon[ok]

    origin_lat, origin_lon = infer_grid_origin(lat, lon, grid_km=float(cfg.grid_km))
    spec = GridSpec(grid_km=float(cfg.grid_km), origin_lat=float(origin_lat), origin_lon=float(origin_lon))

    x, y = latlon_to_xy_m(lat, lon, spec=spec)
    ii, jj = xy_m_to_ij(x, y, spec=spec)
    cell_ids = [ij_to_cell_id(int(i), int(j)) for i, j in zip(ii.tolist(), jj.tolist())]

    by_cell: Dict[str, List[int]] = {}
    for idx, cid in enumerate(cell_ids):
        by_cell.setdefault(cid, []).append(int(idx))

    fc = _FeatureCache(las_root=Path(las_root), cfg=cfg)

    reps_out: List[Dict[str, Any]] = []
    rep_id = 0
    per_cell: Dict[str, Any] = {}
    n_fail = 0

    lat_all = lat.copy()
    lon_all = lon.copy()

    cell_tag = f"grid:{float(cfg.grid_km):g}km"

    for cid in sorted(by_cell.keys()):
        idx0 = np.asarray(by_cell[cid], dtype="int64")
        lat0 = float(np.mean(lat_all[idx0]))
        lon0 = float(np.mean(lon_all[idx0]))

        i0, j0 = cell_id_to_ij(cid)

        chosen_radius = None
        cand_idx: np.ndarray = np.zeros((0,), dtype="int64")

        fallback_mode = "kernel"
        for rad in range(int(cfg.kernel_radius), int(cfg.kernel_radius_max) + 1):
            cells_ij = kernel_cells_ij(i0, j0, radius=int(rad))
            cand_list: List[int] = []
            for (i2, j2) in cells_ij:
                c2 = ij_to_cell_id(int(i2), int(j2))
                cand_list.extend(by_cell.get(c2, []))
            cand_idx = np.asarray(sorted(set(cand_list)), dtype="int64")
            if cand_idx.size >= int(cfg.n_min_postqc):
                chosen_radius = int(rad)
                break

        if chosen_radius is None:
            d_km = haversine_km(lat0, lon0, lat_all, lon_all)
            m = np.isfinite(d_km) & (d_km <= float(cfg.fallback_r_max_km))
            idx = np.where(m)[0]
            if idx.size > 0:
                order = np.argsort(d_km[idx])
                idx = idx[order]
                cand_idx = idx[: int(cfg.fallback_nearest_n)]
            chosen_radius = -1
            fallback_mode = "nearestN"

        d_km_cand = haversine_km(lat0, lon0, lat_all[cand_idx], lon_all[cand_idx]) if cand_idx.size else np.zeros((0,))
        if cand_idx.size > int(cfg.max_candidates_per_kernel):
            o = np.argsort(d_km_cand)
            cand_idx = cand_idx[o[: int(cfg.max_candidates_per_kernel)]]
            d_km_cand = d_km_cand[o[: int(cfg.max_candidates_per_kernel)]]

        feats_list: List[np.ndarray] = []
        shape_list: List[np.ndarray] = []
        keep_mask: List[bool] = []

        for wi in cand_idx.tolist():
            r = rows_use[int(wi)]
            url = (r.get("url", "") or "").strip()
            f = fc.get(url)

            if not _hard_qc_pass(f, cfg=cfg):
                keep_mask.append(False)
                continue

            keep_mask.append(True)
            feats_list.append(_feature_vector(f))
            shape_list.append(np.asarray(f.gr_lowres, dtype="float64"))

        keep_mask_a = np.asarray(keep_mask, dtype=bool)
        n_cand = int(cand_idx.size)
        n_pass = int(np.count_nonzero(keep_mask_a))

        if n_pass < int(cfg.n_min_postqc):
            per_cell[cid] = {
                "status": "FAIL",
                "fallback_mode": fallback_mode,
                "kernel_radius_used": chosen_radius,
                "n_candidates": n_cand,
                "n_pass_hardqc": n_pass,
                "reason": "too_few_candidates_postqc",
            }
            n_fail += 1
            continue

        feats = np.stack([feats_list[i] for i in range(len(feats_list))], axis=0)
        shapes = np.stack([shape_list[i] for i in range(len(shape_list))], axis=0)

        z_feat = np.zeros_like(feats, dtype="float64")
        for c in range(feats.shape[1]):
            z_feat[:, c] = _robust_z(feats[:, c])
        out_feat = np.any(np.abs(z_feat) > float(cfg.zmax_feature), axis=1)

        Dshape = _pairwise_L1(shapes)
        medD = np.median(Dshape, axis=1)
        z_shape = _robust_z(medD)
        out_shape = np.abs(z_shape) > float(cfg.zmax_shape)

        keep2 = ~(out_feat | out_shape)
        n_kept2 = int(np.count_nonzero(keep2))
        if n_kept2 < max(3, int(cfg.n_min_postqc) // 2):
            keep2 = np.ones_like(keep2, dtype=bool)
            n_kept2 = int(np.count_nonzero(keep2))

        feats2 = feats[keep2]
        shapes2 = shapes[keep2]

        if bool(cfg.use_distance_weights):
            d2 = d_km_cand[keep_mask_a][keep2]
            w_i = _distance_weights_km(d2, sigma_km=float(cfg.sigma_km))
        else:
            w_i = np.ones((feats2.shape[0],), dtype="float64")

        midx = _select_medoid(feats2, shapes2, w_i, cfg=cfg)

        kept_indices = np.where(keep_mask_a)[0]
        kept2_indices = kept_indices[keep2]
        chosen_local = int(kept2_indices[midx])
        chosen_wi = int(cand_idx[chosen_local])

        chosen_row = rows_use[chosen_wi]
        chosen_url = (chosen_row.get("url", "") or "").strip()
        las_path = fc.las_path_for_url(chosen_url)

        if las_path is None:
            per_cell[cid] = {
                "status": "FAIL",
                "fallback_mode": fallback_mode,
                "kernel_radius_used": chosen_radius,
                "n_candidates": n_cand,
                "n_pass_hardqc": n_pass,
                "n_kept_after_outliers": n_kept2,
                "reason": "chosen_missing_las",
            }
            n_fail += 1
            continue

        rep_id += 1
        reps_out.append(
            {
                "rep_id": int(rep_id),
                "cell_id": str(cid),
                "cell_tag": cell_tag,
                "score": "0.000",
                "url": chosen_row.get("url", ""),
                "kgs_id": chosen_row.get("kgs_id", ""),
                "api": chosen_row.get("api", ""),
                "api_num_nodash": chosen_row.get("api_num_nodash", ""),
                "operator": chosen_row.get("operator", ""),
                "lease": chosen_row.get("lease", ""),
                "lat": chosen_row.get("lat", ""),
                "lon": chosen_row.get("lon", ""),
                "las_path": str(las_path),
                "picked_gr": "GR",
                "picked_por": "",
                "picked_den": "",
                "picked_neu": "",
                "picked_pe": "",
                "picked_dt": "",
            }
        )

        per_cell[cid] = {
            "status": "OK",
            "fallback_mode": fallback_mode,
            "kernel_radius_used": chosen_radius,
            "n_candidates": n_cand,
            "n_pass_hardqc": n_pass,
            "n_kept_after_outliers": n_kept2,
            "chosen_url": chosen_url,
            "chosen_rep_id": int(rep_id),
        }

    diag: Dict[str, Any] = {
        "mode": "grid_typewells",
        "grid_km": float(cfg.grid_km),
        "origin_lat": float(spec.origin_lat),
        "origin_lon": float(spec.origin_lon),
        "kernel_radius": int(cfg.kernel_radius),
        "kernel_radius_max": int(cfg.kernel_radius_max),
        "n_cells_total": int(len(by_cell)),
        "n_typewells_selected": int(len(reps_out)),
        "n_cells_failed": int(n_fail),
        "per_cell": per_cell,
        "feature_cfg": {
            "n_lowres": int(cfg.gr_cfg.n_lowres),
            "p_lo": float(cfg.gr_cfg.p_lo),
            "p_hi": float(cfg.gr_cfg.p_hi),
            "min_finite_raw": int(cfg.gr_cfg.min_finite_raw),
        },
        "cell_tag": cell_tag,
    }

    return reps_out, diag
