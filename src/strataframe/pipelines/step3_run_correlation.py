# src/strataframe/pipelines/step3_run_correlation.py
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

try:
    import lasio  # type: ignore
except Exception:  # pragma: no cover
    lasio = None  # type: ignore

from strataframe.correlation.dtw import DtwConfig
from strataframe.correlation.framework import FrameworkConfig

from strataframe.rgt.shifts_npz import save_shifts_npz
from strataframe.rgt.rgt import RgtConfig

from strataframe.pipelines.step3a_candidate_edges import CandidateGraphConfig, build_candidate_edges
from strataframe.pipelines.step3b_rep_arrays import (
    RepArraysConfig,
    build_rep_arrays,
    load_rep_arrays_npz,
    save_meta_table,
    save_rep_arrays_npz,
)
from strataframe.pipelines.step3c_dtw_edges import DtwEdgesConfig, run_dtw_on_candidate_edges, save_dtw_outputs
from strataframe.pipelines.step3d_framework_from_edges import (
    BuildFrameworkGraphConfig,
    build_graph_from_edges,
    prune_framework_graph,
)
from strataframe.pipelines.step3_graph_attach_arrays import attach_rep_arrays
from strataframe.pipelines.step3e_rgt_from_paths import default_component_anchors, solve_rgt_from_framework


def _require() -> None:
    if pd is None:
        raise RuntimeError("pandas is required for Step 3. Install with: pip install pandas")
    if nx is None:
        raise RuntimeError("networkx is required for Step 3. Install with: pip install networkx")
    if lasio is None:
        raise RuntimeError("lasio is required for Step 3. Install with: pip install lasio")


@dataclass(frozen=True)
class Step3Paths:
    out_dir: Path

    @property
    def candidates_csv(self) -> Path:
        return self.out_dir / "rep_edges_candidates.csv"

    @property
    def rep_arrays_npz(self) -> Path:
        return self.out_dir / "rep_arrays.npz"

    @property
    def rep_arrays_meta_csv(self) -> Path:
        return self.out_dir / "rep_arrays_meta.csv"

    @property
    def dtw_edges_csv(self) -> Path:
        return self.out_dir / "dtw_edges.csv"

    @property
    def dtw_paths_npz(self) -> Path:
        return self.out_dir / "dtw_paths.npz"

    @property
    def framework_edges_csv(self) -> Path:
        return self.out_dir / "framework_edges.csv"

    @property
    def framework_nodes_csv(self) -> Path:
        return self.out_dir / "framework_nodes.csv"

    @property
    def framework_diag_json(self) -> Path:
        return self.out_dir / "framework_diagnostics.json"

    @property
    def rgt_shifts_npz(self) -> Path:
        return self.out_dir / "rgt_shifts_resampled.npz"


@dataclass(frozen=True)
class Step3Config:
    # Inputs
    reps_csv: Path
    las_root: Path
    wells_gr_parquet: Optional[Path] = None  # used to resolve LAS paths if reps_csv lacks them

    # Candidate graph (preferred knobs)
    cg_k_max: int = 12
    cg_r_max_km: float = 5.0
    cg_use_quadrants: bool = True
    cg_ensure_one_nn: bool = True

    # Legacy knobs (kept for older configs; mapped onto cg_* if set)
    graph_r_max_m: Optional[float] = None  # if set, overrides cg_r_max_km via /1000
    graph_k_max: Optional[int] = None      # if set, overrides cg_k_max

    # 3b rep arrays
    n_samples: int = 400
    p_lo: float = 1.0
    p_hi: float = 99.0
    fill_nans: bool = True
    min_finite: int = 20
    max_gap_depth: Optional[float] = None

    # 3c DTW
    dtw_alpha: float = 0.15
    dtw_band_rad: Optional[int] = None
    dtw_min_finite: int = 20
    dtw_downsample_path_to: int = 80

    # 3d framework pruning
    fw_mode: str = "mst_plus_topk"
    fw_topk: int = 3
    fw_topk_extra: int = 3
    fw_sim_threshold: float = 0.60
    fw_extra_sim_min: float = 0.0
    fw_sim_scale: float = 0.25

    # 3e RGT solve
    rgt_damping: float = 1e-2
    rgt_maxiter: int = 500
    rgt_tol: float = 1e-6
    rgt_simplified_indexing: bool = True
    rgt_lambda_anchor: float = 1.0

    # Column naming (robust heuristics)
    rep_id_col: str = "rep_id"
    bin_id_col: str = "bin_id"
    lat_col: str = "lat"
    lon_col: str = "lon"


# -------------------------
# LAS loading helpers
# -------------------------

_GR_CANDIDATES = (
    "GR",
    "GRC",
    "GRD",
    "SGR",
    "CGR",
    "GAM",
    "GAMMA",
    "NGAM",
)


def _normalize_reps_columns(df: "pd.DataFrame") -> "pd.DataFrame":
    # Alias: h3_cell -> bin_id
    if "bin_id" not in df.columns:
        if "h3_cell" in df.columns:
            df["bin_id"] = df["h3_cell"]
        elif "cell_id" in df.columns:
            df["bin_id"] = df["cell_id"]

    # Make IDs stable as strings
    if "rep_id" in df.columns:
        df["rep_id"] = df["rep_id"].astype(str)

    # Lat/lon as numeric
    if "lat" in df.columns:
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    if "lon" in df.columns:
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    return df


def _pick_curve_mnemonic(las: "lasio.LASFile", candidates: Sequence[str]) -> Optional[str]:
    names = [str(c.mnemonic).strip() for c in las.curves]
    upper = {n.upper(): n for n in names}
    for c in candidates:
        if str(c).upper() in upper:
            return upper[str(c).upper()]
    for n in names:
        if n.upper().startswith("GR"):
            return n
    for n in names:
        if "GAM" in n.upper():
            return n
    return None


def _read_las_depth_and_gr(las_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    try:
        las = lasio.read(str(las_path), engine="normal")
    except TypeError:
        las = lasio.read(str(las_path))

    # depth axis
    try:
        depth = np.asarray(las.index, dtype="float64")
    except Exception:
        depth = np.array([], dtype="float64")
    if depth.size == 0:
        for nm in ("DEPT", "DEPTH", "Z"):
            if nm in las.curvesdict:
                depth = np.asarray(las[nm], dtype="float64")
                break
    if depth.size == 0:
        raise ValueError(f"no_depth_axis_in_las: {las_path.name}")

    gr_mn = _pick_curve_mnemonic(las, _GR_CANDIDATES)
    if not gr_mn:
        raise ValueError(f"no_gr_curve_in_las: {las_path.name}")

    gr = np.asarray(las[gr_mn], dtype="float64")
    if gr.size != depth.size:
        n = int(min(gr.size, depth.size))
        depth = depth[:n]
        gr = gr[:n]
    return depth, gr


def _resolve_las_paths(
    reps: "pd.DataFrame",
    *,
    las_root: Path,
    wells_gr_parquet: Optional[Path],
) -> "pd.DataFrame":
    """
    Ensure reps has a usable absolute 'las_path' column.

    Priority:
      1) reps already has las_path (abs or rel -> resolved vs las_root)
      2) reps has las_relpath / las_file -> resolved vs las_root
      3) join reps with wells_gr.parquet (step0) to get path-ish column
    """
    reps = reps.copy()
    las_root = Path(las_root)

    def _to_abs(p: object) -> str:
        if p is None:
            return ""
        if pd is not None and pd.isna(p):
            return ""
        s = str(p).strip()
        if not s or s.lower() == "nan":
            return ""
        pp = Path(s)
        if not pp.is_absolute():
            pp = (las_root / pp)
        return str(pp.resolve())

    # direct
    if "las_path" in reps.columns:
        reps["las_path"] = reps["las_path"].apply(_to_abs)
        return reps

    # relpath/file
    for c in ("las_relpath", "las_file", "las_filename", "filename", "path"):
        if c in reps.columns:
            reps["las_path"] = reps[c].apply(_to_abs)
            return reps

    # join with wells_gr parquet if available
    if wells_gr_parquet is None:
        raise ValueError("reps CSV lacks las_path/las_relpath, and wells_gr_parquet not provided")
    if not Path(wells_gr_parquet).exists():
        raise ValueError(f"wells_gr_parquet not found: {wells_gr_parquet}")

    wg = pd.read_parquet(wells_gr_parquet)

    # find join key
    join_keys = ["well_id", "api", "uwi", "UWI", "API"]
    key = next((k for k in join_keys if k in reps.columns and k in wg.columns), None)
    if key is None:
        raise ValueError("Could not find common join key between reps and wells_gr.parquet (tried well_id/api/uwi)")

    # find path column in wells_gr
    path_col = next((c for c in ("las_path", "las_relpath", "las_file", "path", "filename") if c in wg.columns), None)
    if path_col is None:
        raise ValueError("wells_gr.parquet does not appear to contain a LAS path column")

    wg2 = wg[[key, path_col]].copy()
    wg2[key] = wg2[key].astype(str)
    reps[key] = reps[key].astype(str)

    reps = reps.merge(wg2, on=key, how="left", suffixes=("", "_wg"))
    reps["_las_src_col"] = path_col
    reps["las_path"] = reps[path_col].apply(_to_abs)

    return reps


def _edges_graph_to_df(G: "nx.Graph") -> "pd.DataFrame":
    rows = []
    for u, v, ed in G.edges(data=True):
        r = {"src_rep_id": str(u), "dst_rep_id": str(v)}
        r.update({k: ed.get(k) for k in ed.keys()})
        rows.append(r)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["src_rep_id", "dst_rep_id"])


def _nodes_graph_to_df(G: "nx.Graph") -> "pd.DataFrame":
    rows = []
    for n, nd in G.nodes(data=True):
        r = {"rep_id": str(n)}
        for k in ("lat", "lon", "bin_id"):
            if k in nd:
                r[k] = nd.get(k)
        rows.append(r)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["rep_id", "lat", "lon", "bin_id"])


def _save_shifts_npz(shifts: Dict[str, np.ndarray], out_path: Path, *, n_samples_hint: Optional[int] = None) -> None:
    """
    Writes shifts in a simple stacked format:
      rep_ids: (R,) unicode
      shifts:  (R,nS) float64
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rep_ids = np.asarray(sorted(shifts.keys()), dtype=np.str_)
    if rep_ids.size == 0:
        nS = int(n_samples_hint) if n_samples_hint is not None else 0
        np.savez_compressed(out_path, rep_ids=rep_ids, shifts=np.zeros((0, nS), dtype="float64"))
        return
    S = np.stack([np.asarray(shifts[str(r)], dtype="float64") for r in rep_ids.tolist()], axis=0)
    np.savez_compressed(out_path, rep_ids=rep_ids, shifts=S)


def run_step3(
    *,
    out_dir: Path,
    cfg: Step3Config,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Runs Step 3 end-to-end:
      3a) candidate edges
      3b) rep arrays cache (depth_rs, gr_rs normalized)
      3c) DTW on candidates + packed paths
      3d) similarity + prune -> framework graph
      3e) inject paths + solve RGT shifts

    Returns diagnostics dict.
    """
    _require()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = Step3Paths(out_dir=out_dir)

    # -------------------------
    # Load reps
    # -------------------------
    reps = pd.read_csv(cfg.reps_csv)
    reps = _normalize_reps_columns(reps)

    for col in [cfg.rep_id_col, cfg.lat_col, cfg.lon_col]:
        if col not in reps.columns:
            raise ValueError(f"reps CSV missing required column: {col}")

    # Ensure bin_id exists (optional, but used for output/node attrs)
    if cfg.bin_id_col not in reps.columns:
        reps[cfg.bin_id_col] = ""

    reps[cfg.rep_id_col] = reps[cfg.rep_id_col].astype(str)
    reps[cfg.bin_id_col] = reps[cfg.bin_id_col].astype(str)
    reps[cfg.lat_col] = pd.to_numeric(reps[cfg.lat_col], errors="coerce")
    reps[cfg.lon_col] = pd.to_numeric(reps[cfg.lon_col], errors="coerce")

    reps = _resolve_las_paths(reps, las_root=cfg.las_root, wells_gr_parquet=cfg.wells_gr_parquet)

    # Drop rows without lat/lon (cannot build graph meaningfully)
    reps0 = reps.copy()
    reps = reps[np.isfinite(reps[cfg.lat_col].to_numpy()) & np.isfinite(reps[cfg.lon_col].to_numpy())].copy()

    # -------------------------
    # 3a Candidate edges
    # -------------------------
    if overwrite or (not paths.candidates_csv.exists()):
        # Map legacy knobs onto cg_* if present
        k_max = int(cfg.cg_k_max)
        r_max_km = float(cfg.cg_r_max_km)
        if cfg.graph_k_max is not None:
            k_max = int(cfg.graph_k_max)
        if cfg.graph_r_max_m is not None and np.isfinite(cfg.graph_r_max_m):
            r_max_km = float(cfg.graph_r_max_m) / 1000.0

        cg_cfg = CandidateGraphConfig(
            k_max=max(1, k_max),
            r_max_km=r_max_km,
            use_quadrants=bool(cfg.cg_use_quadrants),
            ensure_one_nn=bool(cfg.cg_ensure_one_nn),
            rep_id_col=cfg.rep_id_col,
            lat_col=cfg.lat_col,
            lon_col=cfg.lon_col,
        )
        cand = build_candidate_edges(reps, cfg=cg_cfg, bins_meta=None)
        cand.to_csv(paths.candidates_csv, index=False)
    else:
        cand = pd.read_csv(paths.candidates_csv)

    # -------------------------
    # 3b Build rep arrays
    # -------------------------
    if overwrite or (not paths.rep_arrays_npz.exists()):
        rep_to_path = dict(
            zip(
                reps[cfg.rep_id_col].astype(str).tolist(),
                reps["las_path"].astype(str).tolist(),
            )
        )

        def loader(rep_id: str) -> Tuple[np.ndarray, np.ndarray]:
            p = rep_to_path.get(str(rep_id), "")
            if not p or str(p).lower() == "nan":
                raise ValueError("missing_las_path")
            las_path = Path(p)
            if not las_path.exists():
                raise FileNotFoundError(str(las_path))
            return _read_las_depth_and_gr(las_path)

        ra_cfg = RepArraysConfig(
            n_samples=int(cfg.n_samples),
            p_lo=float(cfg.p_lo),
            p_hi=float(cfg.p_hi),
            fill_nans=bool(cfg.fill_nans),
            min_finite=int(cfg.min_finite),
            max_gap_depth=float(cfg.max_gap_depth) if cfg.max_gap_depth is not None else None,
        )

        res = build_rep_arrays(reps[cfg.rep_id_col].astype(str).tolist(), loader, cfg=ra_cfg, keep_meta=True)
        save_rep_arrays_npz(res, paths.rep_arrays_npz, include_imputed_mask=True, compress=True)
        save_meta_table(res, paths.rep_arrays_meta_csv)

        usable_rep_ids = set(res.rep_ids)
    else:
        rep_arrays = load_rep_arrays_npz(paths.rep_arrays_npz)
        usable_rep_ids = set(rep_arrays.keys())

    # Filter reps to usable
    reps_use = reps[reps[cfg.rep_id_col].astype(str).isin(sorted(usable_rep_ids))].copy()

    # -------------------------
    # 3c DTW over candidate edges
    # -------------------------
    if overwrite or (not paths.dtw_edges_csv.exists()) or (not paths.dtw_paths_npz.exists()):
        rep_arrays = load_rep_arrays_npz(paths.rep_arrays_npz)

        # filter candidate edges to usable endpoints
        cand2 = cand.copy()
        cand2["src_rep_id"] = cand2["src_rep_id"].astype(str)
        cand2["dst_rep_id"] = cand2["dst_rep_id"].astype(str)
        m = cand2["src_rep_id"].isin(usable_rep_ids) & cand2["dst_rep_id"].isin(usable_rep_ids)
        cand2 = cand2[m].copy()

        dtw_cfg = DtwConfig(alpha=float(cfg.dtw_alpha), band_rad=cfg.dtw_band_rad, min_finite=int(cfg.dtw_min_finite))
        dtw_edges_cfg = DtwEdgesConfig(downsample_path_to=int(cfg.dtw_downsample_path_to))

        dtw_edges, edge_ids, dtw_paths = run_dtw_on_candidate_edges(rep_arrays, cand2, dtw_cfg=dtw_cfg, cfg=dtw_edges_cfg)

        # join back dist/source for convenience
        key_cols = ["src_rep_id", "dst_rep_id"]
        extra_cols = [c for c in cand2.columns if c not in key_cols]
        if extra_cols:
            dtw_edges = dtw_edges.merge(cand2[key_cols + extra_cols], on=key_cols, how="left")

        save_dtw_outputs(
            dtw_edges,
            edge_ids,
            dtw_paths,
            dtw_edges_csv=str(paths.dtw_edges_csv),
            dtw_paths_npz=str(paths.dtw_paths_npz),
        )
    else:
        dtw_edges = pd.read_csv(paths.dtw_edges_csv)

    # -------------------------
    # 3d Build graph + prune framework
    # -------------------------
    fw_cfg = FrameworkConfig(
        mode=str(cfg.fw_mode),
        topk=int(cfg.fw_topk),
        topk_extra=int(cfg.fw_topk_extra),
        sim_threshold=float(cfg.fw_sim_threshold),
        extra_sim_min=float(cfg.fw_extra_sim_min),
        sim_scale=float(cfg.fw_sim_scale),
    )

    # Filter edges to usable + OK DTW
    dtw_edges2 = dtw_edges.copy()
    if "dtw_status" in dtw_edges2.columns:
        dtw_edges2 = dtw_edges2[dtw_edges2["dtw_status"].astype(str).str.upper() == "OK"].copy()
    if "dtw_cost_per_step" in dtw_edges2.columns:
        dtw_edges2["dtw_cost_per_step"] = pd.to_numeric(dtw_edges2["dtw_cost_per_step"], errors="coerce")
        dtw_edges2 = dtw_edges2[np.isfinite(dtw_edges2["dtw_cost_per_step"].to_numpy())].copy()

    # Build graph from dtw_edges; include only reps_use nodes for consistency
    reps_for_graph = reps_use.rename(
        columns={
            cfg.rep_id_col: "rep_id",
            cfg.lat_col: "lat",
            cfg.lon_col: "lon",
            cfg.bin_id_col: "bin_id",
        }
    )

    G = build_graph_from_edges(
        reps_for_graph,
        dtw_edges2,
        cfg=BuildFrameworkGraphConfig(node_id_col="rep_id", lat_col="lat", lon_col="lon", bin_id_col="bin_id"),
    )

    # prune
    H = prune_framework_graph(G, fw_cfg=fw_cfg, sim_scale=float(cfg.fw_sim_scale))

    # Save framework outputs
    fw_edges_df = _edges_graph_to_df(H)
    fw_nodes_df = _nodes_graph_to_df(H)

    fw_edges_df.to_csv(paths.framework_edges_csv, index=False)
    fw_nodes_df.to_csv(paths.framework_nodes_csv, index=False)

    fw_diag = {
        "n_nodes": int(H.number_of_nodes()),
        "n_edges": int(H.number_of_edges()),
        "mode": fw_cfg.mode,
        "sim_scale": float(fw_cfg.sim_scale),
        "topk": int(fw_cfg.topk),
        "topk_extra": int(getattr(fw_cfg, "topk_extra", 0)),
        "sim_threshold": float(fw_cfg.sim_threshold),
    }
    paths.framework_diag_json.write_text(json.dumps(fw_diag, indent=2))

    # -------------------------
    # 3e RGT solve on pruned graph
    # -------------------------
    rep_arrays = load_rep_arrays_npz(paths.rep_arrays_npz)
    attach_rep_arrays(H, rep_arrays, log_key="log_rs", depth_key="depth_rs", node_log_key="log_rs", node_depth_key="depth_rs")

    sample_idx = max(0, int(cfg.n_samples) // 2)
    anchors = default_component_anchors(H, sample_idx=sample_idx, target_shift=0.0, weight=1.0)

    rgt_cfg = RgtConfig(
        damping=float(cfg.rgt_damping),
        maxiter=int(cfg.rgt_maxiter),
        tol=float(cfg.rgt_tol),
        simplified_indexing=bool(cfg.rgt_simplified_indexing),
        lambda_anchor=float(cfg.rgt_lambda_anchor),
    )

    shifts = solve_rgt_from_framework(H, rgt_cfg=rgt_cfg, paths_npz=str(paths.dtw_paths_npz), anchors=anchors)
    save_shifts_npz(shifts, paths.rgt_shifts_npz)

    # -------------------------
    # Diagnostics
    # -------------------------
    try:
        n_dtw_ok = int((dtw_edges.get("dtw_status", "") == "OK").sum())  # type: ignore[call-arg]
    except Exception:
        n_dtw_ok = 0

    n_candidates_usable = 0
    try:
        if cand.shape[0] > 0 and "src_rep_id" in cand.columns and "dst_rep_id" in cand.columns:
            n_candidates_usable = int(
                (cand["src_rep_id"].astype(str).isin(usable_rep_ids) & cand["dst_rep_id"].astype(str).isin(usable_rep_ids)).sum()
            )
    except Exception:
        n_candidates_usable = 0

    counts = {
        "n_reps_in_csv": int(reps0.shape[0]),
        "n_reps_with_latlon": int(reps.shape[0]),
        "n_reps_usable": int(len(usable_rep_ids)),
        "n_candidates": int(cand.shape[0]),
        "n_candidates_usable": int(n_candidates_usable),
        "n_dtw_ok": int(n_dtw_ok),
        "n_framework_edges": int(H.number_of_edges()),
        "n_components": int(nx.number_connected_components(H)) if H.number_of_nodes() > 0 else 0,
    }

    out_paths = {
        "out_dir": str(paths.out_dir),
        "candidates_csv": str(paths.candidates_csv),
        "rep_arrays_npz": str(paths.rep_arrays_npz),
        "rep_arrays_meta_csv": str(paths.rep_arrays_meta_csv),
        "dtw_edges_csv": str(paths.dtw_edges_csv),
        "dtw_paths_npz": str(paths.dtw_paths_npz),
        "framework_edges_csv": str(paths.framework_edges_csv),
        "framework_nodes_csv": str(paths.framework_nodes_csv),
        "framework_diag_json": str(paths.framework_diag_json),
        "rgt_shifts_npz": str(paths.rgt_shifts_npz),
    }

    return {"counts": counts, "paths": out_paths}
