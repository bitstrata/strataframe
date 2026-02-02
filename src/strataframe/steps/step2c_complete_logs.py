# src/strataframe/steps/step2c_complete_logs.py
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strataframe.spatial.grid import grid_cell_id
from strataframe.typewell.local_typewell import read_gr_features


@dataclass(frozen=True)
class CompleteLogConfig:
    n_samples: int = 512
    n_template: int = 256
    grid_km: float = 10.0
    kernel_radius: int = 4  # 9x9 kernel
    kernel_radius_max: int = 6
    min_kernel_wells: int = 20
    min_coverage_frac: float = 0.90
    max_imputed_frac: float = 0.10
    global_quantile: float = 0.80
    trim_unresp: bool = True
    trim_std_win: int = 15
    trim_std_min: float = 0.02
    progress_every: int = 200
    template_progress_every: int = 10


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
    out["cell_id"] = [grid_cell_id(ix, iy) for ix, iy in zip(gx.tolist(), gy.tolist())]
    return out


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _kernel_cells(ix: int, iy: int, r: int) -> List[str]:
    out = []
    for dx in range(-int(r), int(r) + 1):
        for dy in range(-int(r), int(r) + 1):
            out.append(grid_cell_id(int(ix + dx), int(iy + dy)))
    return out


def _resample_to_n(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype="float64").reshape(-1)
    n = int(n)
    if n <= 0 or x.size == 0:
        return x
    if x.size == n:
        return x
    t_src = np.linspace(0.0, 1.0, x.size, dtype="float64")
    t_dst = np.linspace(0.0, 1.0, n, dtype="float64")
    return np.interp(t_dst, t_src, x).astype("float64", copy=False)


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


def _place_curve_in_expected(
    x: np.ndarray,
    *,
    z_act_top: float,
    z_act_base: float,
    z_exp_top: float,
    z_exp_base: float,
    n_template: int,
) -> np.ndarray:
    x = np.asarray(x, dtype="float64").reshape(-1)
    n = int(n_template)
    out = np.full((n,), np.nan, dtype="float64")
    thk_act = float(z_act_base - z_act_top)
    thk_exp = float(z_exp_base - z_exp_top)
    if not np.isfinite(thk_act) or not np.isfinite(thk_exp) or thk_act <= 0 or thk_exp <= 0:
        return out

    z_exp = np.linspace(float(z_exp_top), float(z_exp_base), n, dtype="float64")
    m = (z_exp >= float(z_act_top)) & (z_exp <= float(z_act_base))
    if not np.any(m):
        return out
    # normalized coordinate in actual interval
    t = (z_exp[m] - float(z_act_top)) / float(thk_act)
    t_src = np.linspace(0.0, 1.0, x.size, dtype="float64")
    out[m] = np.interp(t, t_src, x)
    return out


def _build_global_template(
    *,
    feats: Dict[int, Dict[str, Any]],
    quantile: float,
    n_template: int,
) -> np.ndarray:
    rows = []
    thk = []
    for f in feats.values():
        x = np.asarray(f["x"], dtype="float64")
        if x.size < 8:
            continue
        rows.append(x)
        thk.append(float(f.get("thickness", np.nan)))
    if not rows:
        raise RuntimeError("No GR features available for global template.")
    thk_arr = np.asarray(thk, dtype="float64")
    q = float(np.nanquantile(thk_arr, float(quantile))) if np.any(np.isfinite(thk_arr)) else np.nan
    if not np.isfinite(q):
        q = np.nanmedian(thk_arr)
    keep = [rows[i] for i in range(len(rows)) if np.isfinite(thk_arr[i]) and thk_arr[i] >= q]
    if len(keep) < 5:
        keep = rows
    X = np.stack([_resample_to_n(x, int(n_template)) for x in keep], axis=0)
    return np.nanmedian(X, axis=0).astype("float64", copy=False)


def _build_local_template(
    *,
    kernel_ids: List[int],
    feats: Dict[int, Dict[str, Any]],
    nodes: pd.DataFrame,
    global_template: np.ndarray,
    n_template: int,
) -> Tuple[np.ndarray, int]:
    rows = []
    idx = nodes.set_index("node_id")
    for nid in kernel_ids:
        f = feats.get(int(nid))
        if f is None:
            continue
        if int(nid) not in idx.index:
            continue
        n = idx.loc[int(nid)]
        z_exp_top = _safe_float(n.get("z_exp_top"))
        z_exp_base = _safe_float(n.get("z_exp_base"))
        if z_exp_top is None or z_exp_base is None:
            continue
        x = np.asarray(f["x"], dtype="float64")
        z_act_top = float(f["z_top"])
        z_act_base = float(f["z_base"])
        placed = _place_curve_in_expected(
            x,
            z_act_top=z_act_top,
            z_act_base=z_act_base,
            z_exp_top=float(z_exp_top),
            z_exp_base=float(z_exp_base),
            n_template=int(n_template),
        )
        rows.append(placed)

    if not rows:
        return global_template.copy(), 0

    X = np.stack(rows, axis=0)
    local = np.nanmedian(X, axis=0)
    # fill gaps with global template
    m = ~np.isfinite(local)
    if np.any(m):
        local = local.copy()
        local[m] = global_template[m]
    return local.astype("float64", copy=False), int(len(rows))


def run_complete_logs(
    *,
    nodes_expected_csv: Path,
    out_dir: Path,
    cfg: CompleteLogConfig,
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes = pd.read_csv(nodes_expected_csv)
    nodes["node_id"] = pd.to_numeric(nodes["node_id"], errors="coerce").astype("Int64")
    nodes["x_km"] = pd.to_numeric(nodes.get("x_km"), errors="coerce")
    nodes["y_km"] = pd.to_numeric(nodes.get("y_km"), errors="coerce")
    nodes = nodes[nodes["node_id"].notna() & nodes["x_km"].notna() & nodes["y_km"].notna()].copy()
    nodes["node_id"] = nodes["node_id"].astype(int)

    if "exclude_flag" not in nodes.columns:
        nodes["exclude_flag"] = False

    nodes = _assign_cells(nodes, float(cfg.grid_km))

    # Read GR features for candidate wells
    feats: Dict[int, Dict[str, Any]] = {}
    read_err: Dict[int, str] = {}
    cand_nodes = nodes[~nodes["exclude_flag"]].copy()
    total_cand = int(cand_nodes.shape[0])
    for i, r in enumerate(cand_nodes.itertuples(index=False), start=1):
        rid = int(getattr(r, "node_id"))
        las_path = Path(str(getattr(r, "las_path")))
        if not las_path.exists():
            read_err[rid] = "missing_las"
            continue
        try:
            f = read_gr_features(
                las_path=las_path,
                n=int(cfg.n_template),
                ntg_cutoff=0.4,
                max_rows=0,
            )
        except Exception:
            read_err[rid] = "read_fail"
            continue

        x_full = np.asarray(f["x"], dtype="float64")
        z_top = float(f["z_top"])
        z_base = float(f["z_base"])
        if bool(cfg.trim_unresp):
            x_full, z_top, z_base = _trim_unresponsive(
                x_full,
                z_top,
                z_base,
                win=int(cfg.trim_std_win),
                std_min=float(cfg.trim_std_min),
            )
        thk = float(z_base - z_top)
        if not np.isfinite(thk) or thk <= 0:
            read_err[rid] = "bad_thickness"
            continue
        feats[rid] = {
            "x": _resample_to_n(x_full, int(cfg.n_template)),
            "z_top": float(z_top),
            "z_base": float(z_base),
            "thickness": float(thk),
        }
        if int(cfg.progress_every) > 0 and (i % int(cfg.progress_every) == 0 or i == total_cand):
            print(f"[step2c] read {i}/{total_cand} GR logs (kept={len(feats)})")

    if not feats:
        raise RuntimeError("No readable GR features for Step2c.")

    # Build global template from upper quantile thickness wells
    global_template = _build_global_template(
        feats=feats,
        quantile=float(cfg.global_quantile),
        n_template=int(cfg.n_template),
    )
    global_template_path = out_dir / "global_template.npz"
    np.savez_compressed(global_template_path, template=global_template.astype("float32", copy=False))

    # Build local templates per cell (expected-window domain)
    templates_dir = out_dir / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    by_cell = {cid: g for cid, g in nodes.groupby("cell_id")}
    total_cells = int(len(by_cell))
    cell_templates: Dict[str, np.ndarray] = {}
    cell_i = 0
    for cell_id, g in by_cell.items():
        cell_i += 1
        ix = int(g["cell_ix"].iloc[0])
        iy = int(g["cell_iy"].iloc[0])
        template = None
        used = 0
        for r in range(int(cfg.kernel_radius), int(cfg.kernel_radius_max) + 1):
            kernel_ids = _kernel_cells(ix, iy, r)
            kernel_rows = cand_nodes[cand_nodes["cell_id"].isin(kernel_ids)]
            kernel_ids_int = kernel_rows["node_id"].astype(int).tolist()
            if len(kernel_ids_int) < int(cfg.min_kernel_wells) and r < int(cfg.kernel_radius_max):
                continue
            template, used = _build_local_template(
                kernel_ids=kernel_ids_int,
                feats=feats,
                nodes=nodes,
                global_template=global_template,
                n_template=int(cfg.n_template),
            )
            break

        if template is None:
            template = global_template.copy()
            used = 0
        cell_templates[str(cell_id)] = template

        tpl_path = templates_dir / f"{cell_id}.npz"
        np.savez_compressed(
            tpl_path,
            template=template.astype("float32", copy=False),
            n_template=int(cfg.n_template),
            cell_id=str(cell_id),
            grid_ix=int(ix),
            grid_iy=int(iy),
            n_used=int(used),
        )
        if int(cfg.template_progress_every) > 0 and (
            cell_i % int(cfg.template_progress_every) == 0 or cell_i == total_cells
        ):
            print(f"[step2c] templates {cell_i}/{total_cells} last_cell={cell_id} used={used}")

    # Build complete logs
    node_ids: List[int] = []
    z_tops: List[float] = []
    z_bases: List[float] = []
    x_list: List[np.ndarray] = []
    imputed_list: List[np.ndarray] = []

    qc_rows: List[Dict[str, Any]] = []

    counts: Dict[str, int] = {}
    total_nodes = int(nodes.shape[0])

    def _bump(status: str) -> None:
        counts[status] = counts.get(status, 0) + 1

    def _log_progress() -> None:
        if int(cfg.progress_every) > 0 and (
            len(qc_rows) % int(cfg.progress_every) == 0 or len(qc_rows) == total_nodes
        ):
            ok = counts.get("full", 0) + counts.get("filled", 0)
            print(
                f"[step2c] processed {len(qc_rows)}/{total_nodes} ok={ok} excluded={counts.get('excluded',0)} "
                f"no_template={counts.get('no_template',0)} read_fail={counts.get('read_fail',0)} "
                f"too_imputed={counts.get('too_imputed',0)}"
            )

    def _record(status: str, extra: Optional[Dict[str, Any]] = None) -> None:
        row = {"node_id": rid, "status": status}
        if extra:
            row.update(extra)
        qc_rows.append(row)
        _bump(status)
        _log_progress()

    for r in nodes.itertuples(index=False):
        rid = int(getattr(r, "node_id"))
        if bool(getattr(r, "exclude_flag")):
            _record("excluded")
            continue
        if rid in read_err and rid not in feats:
            _record(read_err[rid])
            continue
        f = feats.get(rid)
        if f is None:
            _record("read_fail")
            continue

        cell_id = str(getattr(r, "cell_id"))
        template = cell_templates.get(cell_id, global_template)

        expected_top = float(getattr(r, "depth_min"))
        expected_base = float(getattr(r, "depth_max"))
        expected_thk = float(expected_base - expected_top)
        if not np.isfinite(expected_thk) or expected_thk <= 0:
            _record("bad_expected")
            continue
        x_full = np.asarray(f["x"], dtype="float64")
        z_top = float(f["z_top"])
        z_base = float(f["z_base"])
        thk = float(z_base - z_top)
        if not np.isfinite(thk) or thk <= 0:
            _record("bad_thickness")
            continue

        x_exp = _place_curve_in_expected(
            x_full,
            z_act_top=z_top,
            z_act_base=z_base,
            z_exp_top=expected_top,
            z_exp_base=expected_base,
            n_template=int(cfg.n_template),
        )
        imputed = ~np.isfinite(x_exp)
        x_complete_tpl = np.asarray(template, dtype="float64").copy()
        x_complete_tpl[~imputed] = x_exp[~imputed]

        imp_frac = float(np.mean(imputed)) if imputed.size else 0.0
        if imp_frac > float(cfg.max_imputed_frac) or (1.0 - imp_frac) < float(cfg.min_coverage_frac):
            _record(
                "too_imputed",
                {
                    "expected_top": expected_top,
                    "expected_base": expected_base,
                    "expected_thk": expected_thk,
                    "actual_top": z_top,
                    "actual_base": z_base,
                    "actual_thk": thk,
                    "imputed_frac": imp_frac,
                },
            )
            continue

        x_complete = _resample_to_n(x_complete_tpl, int(cfg.n_samples)).astype("float32", copy=False)
        imp_res = _resample_to_n(imputed.astype("float64"), int(cfg.n_samples))
        imp_res = (imp_res >= 0.5).astype(bool)

        node_ids.append(int(rid))
        z_tops.append(float(expected_top))
        z_bases.append(float(expected_base))
        x_list.append(x_complete.astype("float32", copy=False))
        imputed_list.append(imp_res.astype(bool, copy=False))

        status = "full" if imp_frac <= 0.0 else "filled"
        _record(
            status,
            {
                "expected_top": expected_top,
                "expected_base": expected_base,
                "expected_thk": expected_thk,
                "actual_top": z_top,
                "actual_base": z_base,
                "actual_thk": thk,
                "imputed_frac": imp_frac,
            },
        )

    if not node_ids:
        raise RuntimeError("No complete logs produced.")

    x_mat = np.stack(x_list, axis=0).astype("float32", copy=False)
    imp_mat = np.stack(imputed_list, axis=0).astype(bool, copy=False)

    out_npz = out_dir / "step2c_complete_gr.npz"
    np.savez_compressed(
        out_npz,
        node_id=np.asarray(node_ids, dtype="int64"),
        z_top=np.asarray(z_tops, dtype="float32"),
        z_base=np.asarray(z_bases, dtype="float32"),
        x_norm=x_mat,
        imputed_mask=imp_mat,
        meta_json=json.dumps(asdict(cfg)),
    )

    qc_csv = out_dir / "step2c_complete_gr_qc.csv"
    pd.DataFrame(qc_rows).to_csv(qc_csv, index=False)

    # Excluded list
    excluded = pd.DataFrame([r for r in qc_rows if str(r.get("status")) == "too_imputed"])
    if not excluded.empty:
        excluded[["node_id"]].drop_duplicates().to_csv(out_dir / "step2c_excluded_nodes.csv", index=False)

    return {"out_npz": str(out_npz), "qc_csv": str(qc_csv)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Step2c: build complete GR logs using local templates.")
    ap.add_argument("--nodes-expected-csv", required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--n-samples", type=int, default=512)
    ap.add_argument("--n-template", type=int, default=256)
    ap.add_argument("--grid-km", type=float, default=10.0)
    ap.add_argument("--kernel-radius", type=int, default=4)
    ap.add_argument("--kernel-radius-max", type=int, default=6)
    ap.add_argument("--min-kernel-wells", type=int, default=20)
    ap.add_argument("--min-coverage-frac", type=float, default=0.90)
    ap.add_argument("--max-imputed-frac", type=float, default=0.10)
    ap.add_argument("--global-quantile", type=float, default=0.80)

    ap.add_argument("--trim-unresp", action="store_true")
    ap.add_argument("--no-trim-unresp", dest="trim_unresp", action="store_false")
    ap.set_defaults(trim_unresp=True)
    ap.add_argument("--trim-std-win", type=int, default=15)
    ap.add_argument("--trim-std-min", type=float, default=0.02)
    ap.add_argument("--progress-every", type=int, default=200)
    ap.add_argument("--template-progress-every", type=int, default=10)

    args = ap.parse_args()
    cfg = CompleteLogConfig(
        n_samples=int(args.n_samples),
        n_template=int(args.n_template),
        grid_km=float(args.grid_km),
        kernel_radius=int(args.kernel_radius),
        kernel_radius_max=int(args.kernel_radius_max),
        min_kernel_wells=int(args.min_kernel_wells),
        min_coverage_frac=float(args.min_coverage_frac),
        max_imputed_frac=float(args.max_imputed_frac),
        global_quantile=float(args.global_quantile),
        trim_unresp=bool(args.trim_unresp),
        trim_std_win=int(args.trim_std_win),
        trim_std_min=float(args.trim_std_min),
        progress_every=int(args.progress_every),
        template_progress_every=int(args.template_progress_every),
    )
    run_complete_logs(
        nodes_expected_csv=Path(args.nodes_expected_csv),
        out_dir=Path(args.out_dir),
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
