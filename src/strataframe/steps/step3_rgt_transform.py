# src/strataframe/steps/step3_rgt_transform.py
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from strataframe.chronolog.dtw import compute_overlap_window, pad_for_distance


@dataclass(frozen=True)
class Step3RgtConfig:
    n_samples: int = 512
    base_pad_ft: float = 10.0
    pad_slope_ft_per_km: float = 0.0
    max_pad_ft: float = 200.0

    path_stride: int = 4
    max_pairs_per_edge: int = 0  # 0 means no limit

    ref_node_id: Optional[int] = None
    ref_weight: float = 1.0
    well_anchor_weight: float = 0.0
    damping_weight: float = 0.0
    cg_maxiter: int = 200
    cg_tol: float = 1e-6


@dataclass(frozen=True)
class Step3RgtPaths:
    out_dir: Path

    @property
    def rgt_shifts_npz(self) -> Path:
        return self.out_dir / "rgt_shifts_resampled.npz"

    @property
    def rgt_meta_csv(self) -> Path:
        return self.out_dir / "rgt_meta.csv"

    @property
    def diagnostics_json(self) -> Path:
        return self.out_dir / "diagnostics.json"

    @property
    def manifest_json(self) -> Path:
        return self.out_dir / "manifest.json"


def _edge_key(i: int, j: int) -> Tuple[int, int]:
    return (i, j) if i < j else (j, i)


def _load_step2_config(step2_dir: Path) -> Dict[str, Any]:
    diag = Path(step2_dir) / "diagnostics.json"
    if diag.exists():
        try:
            return json.loads(diag.read_text()).get("config", {})
        except Exception:
            return {}
    return {}


def _iter_paths_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def _resample_mask_to_depth(
    mask: np.ndarray,
    *,
    z_top: float,
    z_base: float,
    depth_min: float,
    depth_max: float,
    n_samples: int,
) -> np.ndarray:
    mask = np.asarray(mask, dtype="float64").reshape(-1)
    n = int(mask.size)
    if n <= 1 or int(n_samples) <= 1:
        return np.zeros((int(n_samples),), dtype=bool)
    z_src = np.linspace(float(z_top), float(z_base), n, dtype="float64")
    z_dst = np.linspace(float(depth_min), float(depth_max), int(n_samples), dtype="float64")
    m = np.interp(z_dst, z_src, mask, left=1.0, right=1.0)
    return (m >= 0.5)


def _build_equations(
    *,
    nodes: pd.DataFrame,
    edges_ok: Dict[Tuple[int, int], float],
    paths_jsonl: Path,
    imputed_masks: Optional[Dict[int, np.ndarray]],
    cfg: Step3RgtConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    node_ids = nodes["node_id"].astype(int).to_list()
    node_index = {int(n): i for i, n in enumerate(node_ids)}

    depth_min = nodes["depth_min"].to_numpy(dtype="float64")
    depth_max = nodes["depth_max"].to_numpy(dtype="float64")
    n_nodes = int(len(node_ids))
    n_samples = int(cfg.n_samples)

    eq_i: List[int] = []
    eq_j: List[int] = []
    dz_list: List[float] = []

    n_paths = 0
    n_pairs = 0
    n_skipped = 0
    n_skipped_imputed = 0

    for obj in _iter_paths_jsonl(paths_jsonl):
        try:
            src_id = int(obj.get("src_id"))
            dst_id = int(obj.get("dst_id"))
        except Exception:
            n_skipped += 1
            continue

        key = _edge_key(src_id, dst_id)
        if key not in edges_ok:
            continue

        if src_id not in node_index or dst_id not in node_index:
            continue

        path = np.asarray(obj.get("path", []), dtype="int64")
        if path.ndim != 2 or path.shape[1] != 2 or path.size == 0:
            n_skipped += 1
            continue

        n_paths += 1

        i_node = node_index[src_id]
        j_node = node_index[dst_id]

        zmin1 = float(depth_min[i_node])
        zmax1 = float(depth_max[i_node])
        zmin2 = float(depth_min[j_node])
        zmax2 = float(depth_max[j_node])
        if not (np.isfinite(zmin1) and np.isfinite(zmax1) and np.isfinite(zmin2) and np.isfinite(zmax2)):
            n_skipped += 1
            continue
        if zmax1 <= zmin1 or zmax2 <= zmin2:
            n_skipped += 1
            continue

        dist_km = float(edges_ok[key])
        pad_ft = pad_for_distance(
            base_pad_ft=float(cfg.base_pad_ft),
            pad_slope_ft_per_km=float(cfg.pad_slope_ft_per_km),
            max_pad_ft=float(cfg.max_pad_ft),
            dist_km=float(dist_km),
        )
        win_min, win_max, overlap_ft, _ = compute_overlap_window(zmin1, zmax1, zmin2, zmax2, pad_ft)
        if overlap_ft <= 0.0:
            continue

        # Global grids for this pair mapping
        len1 = zmax1 - zmin1
        len2 = zmax2 - zmin2
        if len1 <= 0 or len2 <= 0:
            continue

        stride = max(1, int(cfg.path_stride))
        max_pairs = int(cfg.max_pairs_per_edge)
        count_this = 0
        mask_i = imputed_masks.get(src_id) if imputed_masks is not None else None
        mask_j = imputed_masks.get(dst_id) if imputed_masks is not None else None

        for p_idx in range(0, path.shape[0], stride):
            i_idx, j_idx = int(path[p_idx, 0]), int(path[p_idx, 1])
            if i_idx < 0 or j_idx < 0:
                continue
            if i_idx >= n_samples or j_idx >= n_samples:
                continue

            # local depths within window
            di = float(win_min + (win_max - win_min) * (i_idx / max(1, n_samples - 1)))
            dj = float(win_min + (win_max - win_min) * (j_idx / max(1, n_samples - 1)))

            # map to global sample indices for each well
            gi = int(round((di - zmin1) / len1 * max(1, n_samples - 1)))
            gj = int(round((dj - zmin2) / len2 * max(1, n_samples - 1)))
            gi = int(np.clip(gi, 0, n_samples - 1))
            gj = int(np.clip(gj, 0, n_samples - 1))

            if mask_i is not None and int(gi) < int(mask_i.size) and bool(mask_i[int(gi)]):
                n_skipped_imputed += 1
                continue
            if mask_j is not None and int(gj) < int(mask_j.size) and bool(mask_j[int(gj)]):
                n_skipped_imputed += 1
                continue

            eq_i.append(i_node * n_samples + gi)
            eq_j.append(j_node * n_samples + gj)
            dz_list.append(float(dj - di))
            n_pairs += 1
            count_this += 1
            if max_pairs > 0 and count_this >= max_pairs:
                break

    meta = {
        "n_paths_used": int(n_paths),
        "n_pairs": int(n_pairs),
        "n_skipped": int(n_skipped),
        "n_skipped_imputed": int(n_skipped_imputed),
        "n_nodes": int(n_nodes),
        "n_samples": int(n_samples),
    }
    return (
        np.asarray(eq_i, dtype="int32"),
        np.asarray(eq_j, dtype="int32"),
        np.asarray(dz_list, dtype="float32"),
        meta,
    )


def run_step3_rgt_transform(
    *,
    nodes_csv: Path,
    dtw_edges_csv: Path,
    dtw_paths_jsonl: Path,
    imputed_mask_npz: Optional[Path],
    out_dir: Path,
    cfg: Step3RgtConfig,
    overwrite: bool = False,
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = Step3RgtPaths(out_dir=out_dir)

    if not overwrite and (paths.rgt_shifts_npz.exists() or paths.rgt_meta_csv.exists()):
        raise FileExistsError(
            f"Step3 outputs already exist under {out_dir}. Use overwrite=true or pick a new output dir."
        )

    nodes = pd.read_csv(nodes_csv)
    nodes["node_id"] = pd.to_numeric(nodes["node_id"], errors="coerce").astype("Int64")
    nodes["depth_min"] = pd.to_numeric(nodes.get("depth_min"), errors="coerce")
    nodes["depth_max"] = pd.to_numeric(nodes.get("depth_max"), errors="coerce")
    nodes = nodes[nodes["node_id"].notna()].copy()
    nodes = nodes.sort_values("node_id").reset_index(drop=True)

    # Optional imputed mask (from step2c_complete_gr.npz)
    imputed_masks: Optional[Dict[int, np.ndarray]] = None
    imputed_arr = None
    if imputed_mask_npz is not None and Path(imputed_mask_npz).exists():
        npz = np.load(Path(imputed_mask_npz), allow_pickle=False)
        if "node_id" in npz.files and "imputed_mask" in npz.files:
            ids = np.asarray(npz["node_id"], dtype="int64")
            masks = np.asarray(npz["imputed_mask"], dtype=bool)
            z_top = np.asarray(npz["z_top"], dtype="float64") if "z_top" in npz.files else None
            z_base = np.asarray(npz["z_base"], dtype="float64") if "z_base" in npz.files else None
            idx_map = {int(n): int(i) for i, n in enumerate(ids.tolist())}
            imputed_masks = {}
            out_masks = []
            for r in nodes.itertuples(index=False):
                nid = int(getattr(r, "node_id"))
                dmin = float(getattr(r, "depth_min"))
                dmax = float(getattr(r, "depth_max"))
                if nid in idx_map:
                    i = idx_map[nid]
                    m = masks[i]
                    if z_top is not None and z_base is not None:
                        m_res = _resample_mask_to_depth(
                            m,
                            z_top=float(z_top[i]),
                            z_base=float(z_base[i]),
                            depth_min=dmin,
                            depth_max=dmax,
                            n_samples=int(cfg.n_samples),
                        )
                    else:
                        m_res = np.asarray(m, dtype=bool).reshape(-1)
                        if int(m_res.size) != int(cfg.n_samples):
                            m_res = _resample_mask_to_depth(
                                m_res.astype("float64"),
                                z_top=dmin,
                                z_base=dmax,
                                depth_min=dmin,
                                depth_max=dmax,
                                n_samples=int(cfg.n_samples),
                            )
                    imputed_masks[nid] = m_res
                    out_masks.append(m_res)
                else:
                    out_masks.append(np.zeros((int(cfg.n_samples),), dtype=bool))
            if out_masks:
                imputed_arr = np.stack(out_masks, axis=0)

    edges = pd.read_csv(dtw_edges_csv)
    edges["src_id"] = pd.to_numeric(edges["src_id"], errors="coerce").astype("Int64")
    edges["dst_id"] = pd.to_numeric(edges["dst_id"], errors="coerce").astype("Int64")
    if "status" in edges.columns:
        edges = edges[edges["status"] == "ok"].copy()
    edges = edges[edges["src_id"].notna() & edges["dst_id"].notna()].copy()

    edges_ok: Dict[Tuple[int, int], float] = {}
    for r in edges.itertuples(index=False):
        edges_ok[_edge_key(int(r.src_id), int(r.dst_id))] = float(getattr(r, "dist_km", 0.0))

    eq_i, eq_j, dz, meta = _build_equations(
        nodes=nodes,
        edges_ok=edges_ok,
        paths_jsonl=Path(dtw_paths_jsonl),
        imputed_masks=imputed_masks,
        cfg=cfg,
    )

    n_nodes = int(meta["n_nodes"])
    n_samples = int(meta["n_samples"])
    n_vars = n_nodes * n_samples

    if eq_i.size == 0:
        raise RuntimeError("No equations generated from DTW paths.")

    # Build rhs = A^T b
    rhs = np.zeros((n_vars,), dtype="float64")
    diag = np.zeros((n_vars,), dtype="float64")
    for i_idx, j_idx, dd in zip(eq_i, eq_j, dz):
        rhs[int(i_idx)] += float(dd)
        rhs[int(j_idx)] -= float(dd)
        diag[int(i_idx)] += 1.0
        diag[int(j_idx)] += 1.0

    # Reference constraint (single sample)
    ref_idx = 0
    if cfg.ref_node_id is not None:
        node_ids = nodes["node_id"].astype(int).to_list()
        if int(cfg.ref_node_id) in node_ids:
            ref_idx = node_ids.index(int(cfg.ref_node_id)) * n_samples
    ref_weight = float(cfg.ref_weight)
    well_anchor_weight = float(cfg.well_anchor_weight)
    damping_weight = float(cfg.damping_weight)

    def matvec(v: np.ndarray) -> np.ndarray:
        out = np.zeros_like(v)
        # A^T A v
        for i_idx, j_idx in zip(eq_i, eq_j):
            diff = v[int(i_idx)] - v[int(j_idx)]
            out[int(i_idx)] += diff
            out[int(j_idx)] -= diff
        if ref_weight > 0:
            out[ref_idx] += ref_weight * v[ref_idx]
        if damping_weight > 0:
            out += damping_weight * v
        if well_anchor_weight > 0:
            # Per-well mean anchor: adds w * (mean / n_samples) to each sample in the well
            v_w = v.reshape((n_nodes, n_samples))
            means = v_w.mean(axis=1)
            out_w = out.reshape((n_nodes, n_samples))
            out_w += (well_anchor_weight / max(1, n_samples)) * means[:, None]
        return out

    # Jacobi preconditioner (diagonal of A^T A + ref)
    if ref_weight > 0:
        diag[ref_idx] += ref_weight
    if damping_weight > 0:
        diag += damping_weight
    if well_anchor_weight > 0:
        diag += well_anchor_weight / float(max(1, n_samples) ** 2)
    diag = np.where(diag <= 0, 1.0, diag)
    inv_diag = 1.0 / diag

    def precond(v: np.ndarray) -> np.ndarray:
        return inv_diag * v

    try:
        from scipy.sparse.linalg import LinearOperator, cg  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("scipy is required for CG solver. Install with: pip install scipy") from e

    Aop = LinearOperator((n_vars, n_vars), matvec=matvec, dtype="float64")
    Mop = LinearOperator((n_vars, n_vars), matvec=precond, dtype="float64")
    x0 = np.zeros((n_vars,), dtype="float64")
    sol, info = cg(
        Aop,
        rhs,
        x0=x0,
        maxiter=int(cfg.cg_maxiter),
        rtol=float(cfg.cg_tol),
        atol=0.0,
        M=Mop,
    )

    if info != 0:
        raise RuntimeError(f"CG did not converge (info={info}). Try increasing maxiter or adjusting ref_weight.")

    shifts = sol.reshape((n_nodes, n_samples)).astype("float32", copy=False)

    if imputed_arr is not None:
        np.savez_compressed(
            paths.rgt_shifts_npz,
            node_id=nodes["node_id"].astype("int64").to_numpy(),
            depth_min=nodes["depth_min"].astype("float32").to_numpy(),
            depth_max=nodes["depth_max"].astype("float32").to_numpy(),
            shifts=shifts,
            imputed_mask=imputed_arr,
            meta_json=json.dumps({"cfg": asdict(cfg), "eq_meta": meta}),
        )
    else:
        np.savez_compressed(
            paths.rgt_shifts_npz,
            node_id=nodes["node_id"].astype("int64").to_numpy(),
            depth_min=nodes["depth_min"].astype("float32").to_numpy(),
            depth_max=nodes["depth_max"].astype("float32").to_numpy(),
            shifts=shifts,
            meta_json=json.dumps({"cfg": asdict(cfg), "eq_meta": meta}),
        )

    nodes[["node_id", "depth_min", "depth_max"]].to_csv(paths.rgt_meta_csv, index=False)

    diag = {
        "counts": {
            "n_nodes": int(n_nodes),
            "n_samples": int(n_samples),
            "n_pairs": int(meta["n_pairs"]),
            "n_paths_used": int(meta["n_paths_used"]),
            "n_skipped_imputed": int(meta.get("n_skipped_imputed", 0)),
        },
        "cg": {"info": int(info), "maxiter": int(cfg.cg_maxiter), "tol": float(cfg.cg_tol)},
        "config": asdict(cfg),
    }
    paths.diagnostics_json.write_text(json.dumps(diag, indent=2), encoding="utf-8")
    paths.manifest_json.write_text(
        json.dumps(
            {
                "step": "step3_rgt_transform",
                "inputs": {
                    "nodes_csv": str(nodes_csv),
                    "dtw_edges_csv": str(dtw_edges_csv),
                    "dtw_paths_jsonl": str(dtw_paths_jsonl),
                    "imputed_mask_npz": str(imputed_mask_npz) if imputed_mask_npz is not None else "",
                },
                "outputs": {
                    "rgt_shifts_npz": str(paths.rgt_shifts_npz),
                    "rgt_meta_csv": str(paths.rgt_meta_csv),
                },
                "config": asdict(cfg),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return diag


def main() -> None:
    ap = argparse.ArgumentParser(description="Step3: RGT transform via least-squares from DTW paths.")
    ap.add_argument("--nodes-csv", required=True)
    ap.add_argument("--dtw-edges-csv", required=True)
    ap.add_argument("--dtw-paths-jsonl", required=True)
    ap.add_argument("--imputed-mask-npz", default="", help="Optional step2c_complete_gr.npz for imputed mask filtering")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--step2-dir", default="")
    ap.add_argument("--overwrite", action="store_true")

    ap.add_argument("--n-samples", type=int, default=512)
    ap.add_argument("--base-pad-ft", type=float, default=10.0)
    ap.add_argument("--pad-slope-ft-per-km", type=float, default=0.0)
    ap.add_argument("--max-pad-ft", type=float, default=200.0)

    ap.add_argument("--path-stride", type=int, default=4)
    ap.add_argument("--max-pairs-per-edge", type=int, default=0)

    ap.add_argument("--ref-node-id", type=int, default=-1)
    ap.add_argument("--ref-weight", type=float, default=1.0)
    ap.add_argument("--well-anchor-weight", type=float, default=0.0)
    ap.add_argument("--damping-weight", type=float, default=0.0)
    ap.add_argument("--cg-maxiter", type=int, default=200)
    ap.add_argument("--cg-tol", type=float, default=1e-6)

    args = ap.parse_args()

    cfg = Step3RgtConfig(
        n_samples=int(args.n_samples),
        base_pad_ft=float(args.base_pad_ft),
        pad_slope_ft_per_km=float(args.pad_slope_ft_per_km),
        max_pad_ft=float(args.max_pad_ft),
        path_stride=int(args.path_stride),
        max_pairs_per_edge=int(args.max_pairs_per_edge),
        ref_node_id=int(args.ref_node_id) if int(args.ref_node_id) >= 0 else None,
        ref_weight=float(args.ref_weight),
        well_anchor_weight=float(args.well_anchor_weight),
        damping_weight=float(args.damping_weight),
        cg_maxiter=int(args.cg_maxiter),
        cg_tol=float(args.cg_tol),
    )

    # Optionally inherit pad and n_samples from step2 diagnostics
    if str(args.step2_dir).strip():
        step2_cfg = _load_step2_config(Path(args.step2_dir))
        if step2_cfg:
            cfg = Step3RgtConfig(
                n_samples=int(step2_cfg.get("n_samples", cfg.n_samples)),
                base_pad_ft=float(step2_cfg.get("base_pad_ft", cfg.base_pad_ft)),
                pad_slope_ft_per_km=float(step2_cfg.get("pad_slope_ft_per_km", cfg.pad_slope_ft_per_km)),
                max_pad_ft=float(step2_cfg.get("max_pad_ft", cfg.max_pad_ft)),
                path_stride=int(cfg.path_stride),
                max_pairs_per_edge=int(cfg.max_pairs_per_edge),
                ref_node_id=cfg.ref_node_id,
                ref_weight=cfg.ref_weight,
                well_anchor_weight=cfg.well_anchor_weight,
                damping_weight=cfg.damping_weight,
                cg_maxiter=cfg.cg_maxiter,
                cg_tol=cfg.cg_tol,
            )

    run_step3_rgt_transform(
        nodes_csv=Path(args.nodes_csv),
        dtw_edges_csv=Path(args.dtw_edges_csv),
        dtw_paths_jsonl=Path(args.dtw_paths_jsonl),
        imputed_mask_npz=Path(args.imputed_mask_npz) if str(args.imputed_mask_npz).strip() else None,
        out_dir=Path(args.out_dir),
        cfg=cfg,
        overwrite=bool(args.overwrite),
    )


if __name__ == "__main__":
    main()
