# src/strataframe/steps/step2a_surface_fit.py
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from strataframe.chronolog.dtw import load_gr_vectors_cache


@dataclass(frozen=True)
class SurfaceFitConfig:
    order_min: int = 2
    order_max: int = 6
    cv_folds: int = 5
    cv_blocks: int = 6
    ridge_lambda: float = 1.0e-6
    write_error_map: bool = True
    write_median_map: bool = True
    qc_max_median_depth: float = 40000.0
    qc_med_z_max: float = 4.0
    progress_every: int = 500
    grid_km: float = 10.0
    kernel_radius: int = 4  # 9x9 kernel
    kernel_radius_max: int = 6
    min_kernel_wells: int = 20
    error_map_dpi: int = 200
    error_map_point_size: float = 14.0
    error_map_alpha: float = 0.6
    error_map_vmax_pct: float = 95.0
    write_rmse_csv: bool = True
    write_rmse_plot: bool = True
    rmse_plot_dpi: int = 180
    trim_unresp: bool = True
    trim_std_win: int = 15
    trim_std_min: float = 0.02  # on normalized [0,1]
    qc_min_iqr: float = 0.04
    qc_min_range95: float = 0.08
    qc_min_thickness: float = 50.0
    qc_max_thickness: float = 50_000.0
    qc_thk_z_max: float = 3.5


def _design_matrix(x: np.ndarray, y: np.ndarray, order: int) -> np.ndarray:
    cols = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            cols.append((x ** i) * (y ** j))
    return np.stack(cols, axis=1)


def _fit_ridge(A: np.ndarray, z: np.ndarray, lam: float) -> np.ndarray:
    if lam <= 0:
        coef, *_ = np.linalg.lstsq(A, z, rcond=None)
        return coef
    # ridge via augmented system
    n_cols = int(A.shape[1])
    reg = np.sqrt(float(lam)) * np.eye(n_cols, dtype="float64")
    A2 = np.vstack([A, reg])
    z2 = np.concatenate([z, np.zeros((n_cols,), dtype="float64")], axis=0)
    coef, *_ = np.linalg.lstsq(A2, z2, rcond=None)
    return coef


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(d ** 2)))


def _block_ids(x: np.ndarray, y: np.ndarray, n_blocks: int) -> np.ndarray:
    n_blocks = int(max(2, n_blocks))
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    dx = (xmax - xmin) / float(n_blocks)
    dy = (ymax - ymin) / float(n_blocks)
    dx = dx if dx > 0 else 1.0
    dy = dy if dy > 0 else 1.0
    ix = np.floor((x - xmin) / dx).astype("int64")
    iy = np.floor((y - ymin) / dy).astype("int64")
    ix = np.clip(ix, 0, n_blocks - 1)
    iy = np.clip(iy, 0, n_blocks - 1)
    return (iy * n_blocks + ix).astype("int64")


def _cv_splits(block_ids: np.ndarray, k: int, seed: int = 42) -> List[np.ndarray]:
    blocks = np.unique(block_ids)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(blocks)
    k = int(max(2, k))
    folds: List[np.ndarray] = []
    for i in range(k):
        fold_blocks = blocks[i::k]
        folds.append(fold_blocks)
    return folds


def _rolling_std(x: np.ndarray, win: int) -> np.ndarray:
    x = np.asarray(x, dtype="float64")
    if win <= 1 or x.size < win:
        return np.full(x.shape, np.nan, dtype="float64")
    kernel = np.ones((win,), dtype="float64")
    x2 = x * x
    s1 = np.convolve(x, kernel, mode="valid")
    s2 = np.convolve(x2, kernel, mode="valid")
    mean = s1 / float(win)
    var = s2 / float(win) - mean * mean
    var = np.clip(var, 0.0, None)
    std = np.sqrt(var)
    pad = win // 2
    out = np.full(x.shape, np.nan, dtype="float64")
    out[pad:pad + std.size] = std
    return out


def _trim_unresponsive(
    x: np.ndarray,
    z_top: float,
    z_base: float,
    *,
    win: int,
    std_min: float,
) -> Tuple[np.ndarray, float, float]:
    x = np.asarray(x, dtype="float64")
    if x.size < 8:
        return x, float(z_top), float(z_base)
    std = _rolling_std(x, int(win))
    if not np.any(np.isfinite(std)):
        return x, float(z_top), float(z_base)
    good = np.isfinite(std) & (std >= float(std_min))
    if not np.any(good):
        return x, float(z_top), float(z_base)
    idx = np.arange(x.size, dtype="int64")
    i0 = int(idx[good][0])
    i1 = int(idx[good][-1])
    if i1 <= i0:
        return x, float(z_top), float(z_base)
    z_top2 = float(z_top) + (float(i0) / float(max(1, x.size - 1))) * (float(z_base) - float(z_top))
    z_base2 = float(z_top) + (float(i1) / float(max(1, x.size - 1))) * (float(z_base) - float(z_top))
    return x[i0:i1 + 1], float(z_top2), float(z_base2)


def _robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype="float64")
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad <= 1e-12:
        return np.zeros_like(x)
    return 0.6745 * (x - med) / mad


def _assign_cells(df: pd.DataFrame, grid_km: float) -> pd.DataFrame:
    x = df["x_km"].to_numpy(dtype="float64")
    y = df["y_km"].to_numpy(dtype="float64")
    x0 = float(np.nanmin(x))
    y0 = float(np.nanmin(y))
    gx = np.floor((x - x0) / float(grid_km)).astype("int64")
    gy = np.floor((y - y0) / float(grid_km)).astype("int64")
    out = df.copy()
    out["cell_ix"] = gx
    out["cell_iy"] = gy
    out["cell_id"] = [f"g{ix}_{iy}" for ix, iy in zip(gx.tolist(), gy.tolist())]
    return out


def _kernel_cells(ix: int, iy: int, r: int) -> List[str]:
    out = []
    for dx in range(-int(r), int(r) + 1):
        for dy in range(-int(r), int(r) + 1):
            out.append(f"g{int(ix + dx)}_{int(iy + dy)}")
    return out


def _compute_qc(
    x: np.ndarray,
    z_top: float,
    z_base: float,
    cfg: SurfaceFitConfig,
) -> Dict[str, Any]:
    x = np.asarray(x, dtype="float64")
    fin = np.isfinite(x)
    if not np.any(fin):
        return {"qc_ok": False, "reason": "no_finite"}
    p5 = float(np.nanpercentile(x[fin], 5))
    p95 = float(np.nanpercentile(x[fin], 95))
    iqr = float(np.nanpercentile(x[fin], 75) - np.nanpercentile(x[fin], 25))
    range95 = p95 - p5 if np.isfinite(p5) and np.isfinite(p95) else np.nan
    thk = float(z_base - z_top)
    ok = True
    reason = ""
    if not np.isfinite(thk) or thk <= 0:
        ok = False
        reason = "bad_thickness"
    if thk < float(cfg.qc_min_thickness) or thk > float(cfg.qc_max_thickness):
        ok = False
        reason = "thickness_range"
    if not np.isfinite(iqr) or float(iqr) < float(cfg.qc_min_iqr):
        ok = False
        reason = "low_iqr"
    if not np.isfinite(range95) or float(range95) < float(cfg.qc_min_range95):
        ok = False
        reason = "low_range95"
    return {
        "qc_ok": bool(ok),
        "reason": reason,
        "iqr": float(iqr) if np.isfinite(iqr) else float("nan"),
        "range95": float(range95) if np.isfinite(range95) else float("nan"),
        "thickness": float(thk),
    }


def run_surface_fit(
    *,
    nodes_csv: Path,
    gr_vectors_npz: Path,
    out_dir: Path,
    cfg: SurfaceFitConfig,
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes = pd.read_csv(nodes_csv)
    if "node_id" not in nodes.columns:
        raise ValueError("nodes_csv missing node_id")
    for c in ("x_km", "y_km"):
        if c not in nodes.columns:
            raise ValueError(f"nodes_csv missing {c}")
    nodes["node_id"] = pd.to_numeric(nodes["node_id"], errors="coerce").astype("Int64")
    nodes["x_km"] = pd.to_numeric(nodes["x_km"], errors="coerce")
    nodes["y_km"] = pd.to_numeric(nodes["y_km"], errors="coerce")
    nodes = nodes[nodes["node_id"].notna() & nodes["x_km"].notna() & nodes["y_km"].notna()].copy()
    nodes["node_id"] = nodes["node_id"].astype(int)

    cache = load_gr_vectors_cache(Path(gr_vectors_npz))
    idx_map = cache["index"]
    z_top = cache["z_top"].astype("float64", copy=False)
    z_base = cache["z_base"].astype("float64", copy=False)
    x_norm = cache["x_norm"]

    rows = []
    total_nodes = int(nodes.shape[0])
    for i, r in enumerate(nodes.itertuples(index=False), start=1):
        rid = int(getattr(r, "node_id"))
        if rid not in idx_map:
            continue
        i = idx_map[rid]
        x = np.asarray(x_norm[i], dtype="float64")
        zt = float(z_top[i])
        zb = float(z_base[i])
        if bool(cfg.trim_unresp):
            x2, zt2, zb2 = _trim_unresponsive(
                x,
                zt,
                zb,
                win=int(cfg.trim_std_win),
                std_min=float(cfg.trim_std_min),
            )
        else:
            x2, zt2, zb2 = x, zt, zb
        qc = _compute_qc(x2, zt2, zb2, cfg)
        rows.append(
            {
                "node_id": rid,
                "x_km": float(getattr(r, "x_km")),
                "y_km": float(getattr(r, "y_km")),
                "z_gr_top": float(zt),
                "z_gr_base": float(zb),
                "z_gr_top_trim": float(zt2),
                "z_gr_base_trim": float(zb2),
                "qc_ok": bool(qc.get("qc_ok", False)),
                "qc_reason": str(qc.get("reason", "")),
                "qc_iqr": float(qc.get("iqr", np.nan)),
                "qc_range95": float(qc.get("range95", np.nan)),
                "qc_thickness": float(qc.get("thickness", np.nan)),
            }
        )
        if int(cfg.progress_every) > 0 and (i % int(cfg.progress_every) == 0 or i == total_nodes):
            print(f"[step2a] scanned {i}/{total_nodes} wells, kept={len(rows)}")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No nodes with GR cache + valid coordinates.")

    # robust thickness outliers (post-trim)
    thk = df["qc_thickness"].to_numpy(dtype="float64")
    z_thk = _robust_z(thk)
    df["qc_thk_z"] = z_thk
    df["qc_long_flag"] = np.abs(z_thk) > float(cfg.qc_thk_z_max)

    z_med_obs = 0.5 * (
        pd.to_numeric(df.get("z_gr_top_trim"), errors="coerce")
        + pd.to_numeric(df.get("z_gr_base_trim"), errors="coerce")
    )
    z_med_arr = z_med_obs.to_numpy(dtype="float64")
    med_z = np.full(z_med_arr.shape, np.nan, dtype="float64")
    m_med = np.isfinite(z_med_arr)
    if np.any(m_med):
        med_z[m_med] = _robust_z(z_med_arr[m_med])
    df["qc_med_z"] = med_z
    median_outlier = np.zeros(z_med_arr.shape, dtype=bool)
    if np.any(m_med):
        median_outlier |= np.abs(med_z) > float(cfg.qc_med_z_max)
    if float(cfg.qc_max_median_depth) > 0:
        median_outlier |= z_med_arr > float(cfg.qc_max_median_depth)
    df["median_outlier_flag"] = median_outlier
    df["fit_ok"] = df["qc_ok"] & (~df["qc_long_flag"]) & (~df["median_outlier_flag"])

    fit_df = df[df["fit_ok"]].copy()
    if fit_df.shape[0] < 30:
        raise RuntimeError("Too few wells passed QC for surface fitting.")

    # Normalize x/y
    x = fit_df["x_km"].to_numpy(dtype="float64")
    y = fit_df["y_km"].to_numpy(dtype="float64")
    x0 = float(np.mean(x))
    y0 = float(np.mean(y))
    xs = float(np.std(x)) if float(np.std(x)) > 0 else 1.0
    ys = float(np.std(y)) if float(np.std(y)) > 0 else 1.0
    xn = (x - x0) / xs
    yn = (y - y0) / ys

    zt = fit_df["z_gr_top_trim"].to_numpy(dtype="float64")
    zb = fit_df["z_gr_base_trim"].to_numpy(dtype="float64")

    blocks = _block_ids(xn, yn, int(cfg.cv_blocks))
    folds = _cv_splits(blocks, int(cfg.cv_folds), seed=42)

    order_stats: Dict[int, Dict[str, float]] = {}
    best_order = None
    best_score = float("inf")

    for order in range(int(cfg.order_min), int(cfg.order_max) + 1):
        rmses_top = []
        rmses_base = []
        for fold_blocks in folds:
            m_test = np.isin(blocks, fold_blocks)
            m_train = ~m_test
            if not np.any(m_train) or not np.any(m_test):
                continue
            A_train = _design_matrix(xn[m_train], yn[m_train], order)
            A_test = _design_matrix(xn[m_test], yn[m_test], order)
            b_top = _fit_ridge(A_train, zt[m_train], float(cfg.ridge_lambda))
            b_base = _fit_ridge(A_train, zb[m_train], float(cfg.ridge_lambda))
            pred_top = A_test @ b_top
            pred_base = A_test @ b_base
            rmses_top.append(_rmse(pred_top, zt[m_test]))
            rmses_base.append(_rmse(pred_base, zb[m_test]))

        rmse_top = float(np.nanmean(rmses_top)) if rmses_top else float("nan")
        rmse_base = float(np.nanmean(rmses_base)) if rmses_base else float("nan")
        rmse_total = float(np.nanmean([rmse_top, rmse_base]))
        order_stats[order] = {
            "rmse_top": rmse_top,
            "rmse_base": rmse_base,
            "rmse_total": rmse_total,
        }
        if np.isfinite(rmse_total) and rmse_total < best_score:
            best_score = rmse_total
            best_order = order

    if best_order is None:
        raise RuntimeError("Surface fit failed; no valid CV scores.")

    rmse_csv = out_dir / "step2a_surface_rmse.csv"
    if bool(cfg.write_rmse_csv):
        rmse_rows = [
            {"order": int(k), **v} for k, v in sorted(order_stats.items(), key=lambda kv: int(kv[0]))
        ]
        pd.DataFrame(rmse_rows).to_csv(rmse_csv, index=False)

    rmse_plot = out_dir / "step2a_surface_rmse.png"
    if bool(cfg.write_rmse_plot):
        try:
            import matplotlib.pyplot as plt

            orders = [int(k) for k in sorted(order_stats.keys())]
            rmse_vals = [float(order_stats[k]["rmse_total"]) for k in orders]
            fig, ax = plt.subplots(figsize=(5.4, 3.6), dpi=int(cfg.rmse_plot_dpi))
            ax.bar([str(o) for o in orders], rmse_vals, color="#4c78a8")
            ax.set_xlabel("Polynomial order")
            ax.set_ylabel("CV RMSE (ft)")
            ax.set_title("Surface fit CV RMSE")
            fig.tight_layout()
            fig.savefig(rmse_plot)
            plt.close(fig)
        except Exception:
            pass

    # Fit final models
    A_all = _design_matrix(xn, yn, int(best_order))
    coef_top = _fit_ridge(A_all, zt, float(cfg.ridge_lambda))
    coef_base = _fit_ridge(A_all, zb, float(cfg.ridge_lambda))

    # Predict for all nodes with coords (global)
    x_all = df["x_km"].to_numpy(dtype="float64")
    y_all = df["y_km"].to_numpy(dtype="float64")
    xn_all = (x_all - x0) / xs
    yn_all = (y_all - y0) / ys
    A_pred = _design_matrix(xn_all, yn_all, int(best_order))
    zt_pred = A_pred @ coef_top
    zb_pred = A_pred @ coef_base
    df["z_exp_top_global"] = zt_pred
    df["z_exp_base_global"] = zb_pred

    # Local residual adjustment (kernel median)
    A_fit = _design_matrix(xn, yn, int(best_order))
    fit_df["z_exp_top_global"] = A_fit @ coef_top
    fit_df["z_exp_base_global"] = A_fit @ coef_base
    df = _assign_cells(df, float(cfg.grid_km))
    fit_df = _assign_cells(fit_df, float(cfg.grid_km))
    fit_df["r_top"] = fit_df["z_gr_top_trim"] - (fit_df["z_exp_top_global"])
    fit_df["r_base"] = fit_df["z_gr_base_trim"] - (fit_df["z_exp_base_global"])

    by_cell = {cid: g for cid, g in fit_df.groupby("cell_id")}
    cell_resid = {}
    for cell_id, g in by_cell.items():
        ix = int(g["cell_ix"].iloc[0])
        iy = int(g["cell_iy"].iloc[0])
        res_top = None
        res_base = None
        used = 0
        for r in range(int(cfg.kernel_radius), int(cfg.kernel_radius_max) + 1):
            kernel_ids = _kernel_cells(ix, iy, r)
            rows = fit_df[fit_df["cell_id"].isin(kernel_ids)]
            if rows.empty:
                continue
            if rows.shape[0] < int(cfg.min_kernel_wells) and r < int(cfg.kernel_radius_max):
                continue
            res_top = float(np.nanmedian(rows["r_top"].to_numpy(dtype="float64")))
            res_base = float(np.nanmedian(rows["r_base"].to_numpy(dtype="float64")))
            used = int(rows.shape[0])
            break
        cell_resid[cell_id] = {
            "r_top": float(res_top) if res_top is not None and np.isfinite(res_top) else 0.0,
            "r_base": float(res_base) if res_base is not None and np.isfinite(res_base) else 0.0,
            "n_used": int(used),
        }

    r_top_all = []
    r_base_all = []
    r_n_all = []
    for cid in df["cell_id"].astype(str).tolist():
        meta = cell_resid.get(cid)
        if meta is None:
            r_top_all.append(0.0)
            r_base_all.append(0.0)
            r_n_all.append(0)
        else:
            r_top_all.append(float(meta.get("r_top", 0.0)))
            r_base_all.append(float(meta.get("r_base", 0.0)))
            r_n_all.append(int(meta.get("n_used", 0)))

    df["r_top_local"] = np.asarray(r_top_all, dtype="float64")
    df["r_base_local"] = np.asarray(r_base_all, dtype="float64")
    df["r_n_local"] = np.asarray(r_n_all, dtype="int64")

    df["z_exp_top"] = df["z_exp_top_global"] + df["r_top_local"]
    df["z_exp_base"] = df["z_exp_base_global"] + df["r_base_local"]
    df["z_exp_thickness"] = df["z_exp_base"] - df["z_exp_top"]

    z_top_obs = pd.to_numeric(df.get("z_gr_top_trim"), errors="coerce")
    z_base_obs = pd.to_numeric(df.get("z_gr_base_trim"), errors="coerce")
    err_top = np.abs(z_top_obs - df["z_exp_top"])
    err_base = np.abs(z_base_obs - df["z_exp_base"])
    df["err_top"] = err_top
    df["err_base"] = err_base
    df["err_mean"] = 0.5 * (err_top + err_base)
    z_med_obs = 0.5 * (z_top_obs + z_base_obs)
    z_med_exp = 0.5 * (df["z_exp_top"] + df["z_exp_base"])
    df["z_med_obs"] = z_med_obs
    df["z_med_exp"] = z_med_exp
    df["z_med_plot"] = np.where(np.isfinite(z_med_obs), z_med_obs, z_med_exp)

    pred_csv = out_dir / "step2a_surface_predictions.csv"
    df.to_csv(pred_csv, index=False)

    error_map_path = out_dir / "step2a_surface_error_map.png"
    median_map_path = out_dir / "step2a_surface_median_map.png"
    if bool(cfg.write_error_map) or bool(cfg.write_median_map):
        try:
            import matplotlib.pyplot as plt

            x = pd.to_numeric(df.get("x_km"), errors="coerce").to_numpy(dtype="float64")
            y = pd.to_numeric(df.get("y_km"), errors="coerce").to_numpy(dtype="float64")
            outlier = df.get("median_outlier_flag")
            outlier_mask = np.asarray(outlier, dtype=bool) if outlier is not None else np.zeros_like(x, dtype=bool)
            if bool(cfg.write_error_map):
                e = pd.to_numeric(df.get("err_mean"), errors="coerce").to_numpy(dtype="float64")
                m = np.isfinite(x) & np.isfinite(y) & np.isfinite(e) & (~outlier_mask)
                if np.any(m):
                    vmax = np.nanpercentile(e[m], float(cfg.error_map_vmax_pct))
                    if not np.isfinite(vmax) or vmax <= 0:
                        vmax = float(np.nanmax(e[m])) if np.any(np.isfinite(e[m])) else 1.0
                    fig, ax = plt.subplots(figsize=(6.8, 5.2), dpi=int(cfg.error_map_dpi))
                    sc = ax.scatter(
                        x[m],
                        y[m],
                        c=e[m],
                        s=float(cfg.error_map_point_size),
                        cmap="magma",
                        vmin=0.0,
                        vmax=float(vmax),
                        alpha=float(cfg.error_map_alpha),
                        edgecolors="none",
                    )
                    ax.set_aspect("equal", adjustable="box")
                    ax.set_xlabel("X (km)")
                    ax.set_ylabel("Y (km)")
                    cb = fig.colorbar(sc, ax=ax, shrink=0.88)
                    cb.set_label("Mean |error| (ft)")
                    fig.tight_layout()
                    fig.savefig(error_map_path)
                    plt.close(fig)

            if bool(cfg.write_median_map):
                zmed = pd.to_numeric(df.get("z_med_plot"), errors="coerce").to_numpy(dtype="float64")
                m = np.isfinite(x) & np.isfinite(y) & np.isfinite(zmed) & (~outlier_mask)
                if np.any(m):
                    vmin = float(np.nanmin(zmed[m])) if np.any(np.isfinite(zmed[m])) else 0.0
                    vmax = float(np.nanmax(zmed[m])) if np.any(np.isfinite(zmed[m])) else 1.0
                    if not np.isfinite(vmin):
                        vmin = 0.0
                    if not np.isfinite(vmax) or vmax <= vmin:
                        vmax = vmin + 1.0
                    fig, ax = plt.subplots(figsize=(6.8, 5.2), dpi=int(cfg.error_map_dpi))
                    sc = ax.scatter(
                        x[m],
                        y[m],
                        c=zmed[m],
                        s=float(cfg.error_map_point_size),
                        cmap="viridis_r",
                        vmin=float(vmin),
                        vmax=float(vmax),
                        alpha=float(cfg.error_map_alpha),
                        edgecolors="none",
                    )
                    ax.set_aspect("equal", adjustable="box")
                    ax.set_xlabel("X (km)")
                    ax.set_ylabel("Y (km)")
                    cb = fig.colorbar(sc, ax=ax, shrink=0.88)
                    cb.set_label("Median depth (ft)")
                    fig.tight_layout()
                    fig.savefig(median_map_path)
                    plt.close(fig)
        except Exception:
            pass

    out_json = out_dir / "step2a_surface.json"
    out_json.write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "order_selected": int(best_order),
                "order_stats": order_stats,
                "norm": {"x0": x0, "y0": y0, "x_scale": xs, "y_scale": ys},
                "coef_top": coef_top.tolist(),
                "coef_base": coef_base.tolist(),
                "crs": "EPSG:5070",
                "xy_units": "km",
                "error_map_path": str(error_map_path),
                "median_map_path": str(median_map_path),
                "rmse_csv_path": str(rmse_csv),
                "rmse_plot_path": str(rmse_plot),
                "counts": {"n_median_outliers": int(df["median_outlier_flag"].sum())},
                "local_adjustment": {
                    "grid_km": float(cfg.grid_km),
                    "kernel_radius": int(cfg.kernel_radius),
                    "kernel_radius_max": int(cfg.kernel_radius_max),
                    "min_kernel_wells": int(cfg.min_kernel_wells),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    excluded_csv = out_dir / "step2a_excluded_nodes.csv"
    df[df["median_outlier_flag"]].loc[:, ["node_id", "z_med_obs", "qc_med_z"]].to_csv(
        excluded_csv, index=False
    )

    return {
        "pred_csv": str(pred_csv),
        "surface_json": str(out_json),
        "error_map": str(error_map_path),
        "median_map": str(median_map_path),
        "excluded_csv": str(excluded_csv),
        "rmse_csv": str(rmse_csv),
        "rmse_plot": str(rmse_plot),
        "order": int(best_order),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Step2a: fit smooth surface for GR top/base.")
    ap.add_argument("--nodes-csv", required=True)
    ap.add_argument("--gr-vectors-npz", required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--order-min", type=int, default=2)
    ap.add_argument("--order-max", type=int, default=6)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--cv-blocks", type=int, default=6)
    ap.add_argument("--ridge-lambda", type=float, default=1.0e-6)
    ap.add_argument("--progress-every", type=int, default=500)
    ap.add_argument("--error-map", action="store_true")
    ap.add_argument("--no-error-map", dest="error_map", action="store_false")
    ap.set_defaults(error_map=True)
    ap.add_argument("--median-map", action="store_true")
    ap.add_argument("--no-median-map", dest="median_map", action="store_false")
    ap.set_defaults(median_map=True)
    ap.add_argument("--error-map-dpi", type=int, default=200)
    ap.add_argument("--error-map-point-size", type=float, default=14.0)
    ap.add_argument("--error-map-alpha", type=float, default=0.6)
    ap.add_argument("--error-map-vmax-pct", type=float, default=95.0)
    ap.add_argument("--rmse-csv", action="store_true")
    ap.add_argument("--no-rmse-csv", dest="rmse_csv", action="store_false")
    ap.set_defaults(rmse_csv=True)
    ap.add_argument("--rmse-plot", action="store_true")
    ap.add_argument("--no-rmse-plot", dest="rmse_plot", action="store_false")
    ap.set_defaults(rmse_plot=True)
    ap.add_argument("--rmse-plot-dpi", type=int, default=180)

    ap.add_argument("--trim-unresp", action="store_true")
    ap.add_argument("--no-trim-unresp", dest="trim_unresp", action="store_false")
    ap.set_defaults(trim_unresp=True)
    ap.add_argument("--trim-std-win", type=int, default=15)
    ap.add_argument("--trim-std-min", type=float, default=0.02)

    ap.add_argument("--qc-min-iqr", type=float, default=0.04)
    ap.add_argument("--qc-min-range95", type=float, default=0.08)
    ap.add_argument("--qc-min-thickness", type=float, default=50.0)
    ap.add_argument("--qc-max-thickness", type=float, default=50_000.0)
    ap.add_argument("--qc-thk-z-max", type=float, default=3.5)
    ap.add_argument("--qc-max-median-depth", type=float, default=40000.0)
    ap.add_argument("--qc-med-z-max", type=float, default=4.0)
    ap.add_argument("--grid-km", type=float, default=10.0)
    ap.add_argument("--kernel-radius", type=int, default=4)
    ap.add_argument("--kernel-radius-max", type=int, default=6)
    ap.add_argument("--min-kernel-wells", type=int, default=20)

    args = ap.parse_args()
    cfg = SurfaceFitConfig(
        order_min=int(args.order_min),
        order_max=int(args.order_max),
        cv_folds=int(args.cv_folds),
        cv_blocks=int(args.cv_blocks),
        ridge_lambda=float(args.ridge_lambda),
        write_error_map=bool(args.error_map),
        write_median_map=bool(args.median_map),
        error_map_dpi=int(args.error_map_dpi),
        error_map_point_size=float(args.error_map_point_size),
        error_map_alpha=float(args.error_map_alpha),
        error_map_vmax_pct=float(args.error_map_vmax_pct),
        write_rmse_csv=bool(args.rmse_csv),
        write_rmse_plot=bool(args.rmse_plot),
        rmse_plot_dpi=int(args.rmse_plot_dpi),
        trim_unresp=bool(args.trim_unresp),
        trim_std_win=int(args.trim_std_win),
        trim_std_min=float(args.trim_std_min),
        qc_min_iqr=float(args.qc_min_iqr),
        qc_min_range95=float(args.qc_min_range95),
        qc_min_thickness=float(args.qc_min_thickness),
        qc_max_thickness=float(args.qc_max_thickness),
        qc_thk_z_max=float(args.qc_thk_z_max),
        qc_max_median_depth=float(args.qc_max_median_depth),
        qc_med_z_max=float(args.qc_med_z_max),
        progress_every=int(args.progress_every),
        grid_km=float(args.grid_km),
        kernel_radius=int(args.kernel_radius),
        kernel_radius_max=int(args.kernel_radius_max),
        min_kernel_wells=int(args.min_kernel_wells),
    )
    run_surface_fit(
        nodes_csv=Path(args.nodes_csv),
        gr_vectors_npz=Path(args.gr_vectors_npz),
        out_dir=Path(args.out_dir),
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
