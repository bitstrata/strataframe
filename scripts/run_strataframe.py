# scripts/run_strataframe.py
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Import only Step 0/1 at module import time (keeps --steps 1 working even if Step 2 is broken)
from strataframe.steps.step0_index_gr import Step0Config, run_step0_index_gr
from strataframe.steps.step1_build_bins import Step1Config, run_step1_build_bins
from strataframe.utils.config import deep_get, load_yaml

IMPLEMENTED_STEPS = ["0", "1", "2", "3", "4"]


def _parse_steps(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    parts: List[str] = []
    for tok in s.replace(";", ",").split(","):
        t = tok.strip()
        if t:
            parts.append(t)
    # de-dupe preserving order
    seen = set()
    out = [x for x in parts if not (x in seen or seen.add(x))]
    return out


def _preparse_config(argv: List[str]) -> Path:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--config", type=str, default="")
    ns, _ = ap.parse_known_args(argv)
    return Path(ns.config) if ns.config else Path()


def _dataclass_fieldnames(cls: Any) -> set[str]:
    f = getattr(cls, "__dataclass_fields__", None)
    if isinstance(f, dict):
        return set(f.keys())
    return set()


def _build_object_from_dict(cls: Any, d: Any) -> Any:
    """
    Best-effort construction:
      - if cls is a dataclass: pass only matching fields as kwargs
      - else: try kwargs construction; on failure, instantiate default and setattr known attrs
    """
    d = d if isinstance(d, dict) else {}
    fields = _dataclass_fieldnames(cls)
    if fields:
        kwargs = {k: d[k] for k in d.keys() if k in fields}
        return cls(**kwargs)

    try:
        return cls(**dict(d))
    except Exception:
        obj = cls()
        for k, v in d.items():
            if hasattr(obj, k):
                try:
                    setattr(obj, k, v)
                except Exception:
                    pass
        return obj


def _guess_step1_manifest(run_dir: Path, step1_out_subdir: str) -> Optional[Path]:
    """
    Typical locations:
      - <run_dir>/<step1_out_subdir>/manifest.json
      - <run_dir>/<step1_out_subdir>/step1_manifest.json (rare/legacy)
    """
    c1 = run_dir / step1_out_subdir / "manifest.json"
    if c1.exists():
        return c1
    c2 = run_dir / step1_out_subdir / "step1_manifest.json"
    if c2.exists():
        return c2
    return None


def main(argv: List[str] | None = None) -> int:
    import sys

    argv2 = list(sys.argv[1:] if argv is None else argv)

    cfg_path = _preparse_config(argv2)
    cfg: Dict[str, Any] = {}
    if cfg_path and str(cfg_path).strip():
        cfg = load_yaml(cfg_path)

    # Defaults from config (with safe fallbacks)
    d_ks_manifest = deep_get(cfg, "inputs.ks_manifest", "data/ks_las_files.txt")
    d_las_root = deep_get(cfg, "inputs.las_root", "data/las")
    d_run_dir = deep_get(cfg, "run.run_dir", "runs/runA")
    d_overwrite = bool(deep_get(cfg, "run.overwrite", False))

    d_steps = deep_get(cfg, "pipeline.steps", [])
    if isinstance(d_steps, list):
        d_steps_s = ",".join(str(x) for x in d_steps)
    else:
        d_steps_s = str(d_steps or "")

    # -----------------------
    # Step 0 defaults
    # -----------------------
    d_step0_out = deep_get(cfg, "outputs.step0_out_subdir", "00_wells_gr")
    d_curve_family = deep_get(cfg, "step0.curve_family", "GR")
    d_min_finite = int(deep_get(cfg, "step0.min_finite", 200))
    d_pct = deep_get(cfg, "step0.pct", [1.0, 50.0, 99.0])
    d_quiet = bool(deep_get(cfg, "step0.quiet_lasio", True))

    # -----------------------
    # Step 1 defaults (GRID ONLY)
    # -----------------------
    d_step1_out = deep_get(cfg, "outputs.step1_out_subdir", "01_bins")
    d_step1_target_bins = int(deep_get(cfg, "step1.target_bins", 100))
    d_step1_min_bin_size = int(deep_get(cfg, "step1.min_bin_size", 10))
    d_step1_grid_cell_km = deep_get(cfg, "step1.grid_cell_km", None)
    d_step1_grid_pad_frac = float(deep_get(cfg, "step1.grid_pad_frac", 0.01))
    d_step1_filter_csv = deep_get(cfg, "step1.filter_wells_csv", "")

    # -----------------------
    # Step 2 defaults (read config now; import Step2 lazily later)
    # -----------------------
    d_step2_out = deep_get(cfg, "outputs.step2_out_subdir", "02_reps")
    d_step2_n_rep = int(deep_get(cfg, "step2.n_rep", 1000))
    d_step2_quota_mode = str(deep_get(cfg, "step2.quota_mode", "equal"))
    d_step2_q_min = int(deep_get(cfg, "step2.q_min", 5))
    d_step2_candidate_method = str(
        deep_get(cfg, "step2.candidate_method", deep_get(cfg, "step2.method", "farthest"))
    )
    d_step2_seed = int(deep_get(cfg, "step2.seed", 42))
    d_step2_max_cand = int(deep_get(cfg, "step2.max_candidates_per_cell", 25))
    d_step2_require_gr = bool(deep_get(cfg, "step2.require_gr", True))

    # Additional Step2 fields (grid workflow)
    d_step2_grid_km = float(deep_get(cfg, "step2.grid_km", 10.0))
    d_step2_build_typewells = bool(deep_get(cfg, "step2.build_typewells", True))
    d_step2_max_las_mb = int(deep_get(cfg, "step2.max_las_mb", 512))
    d_step2_max_las_curves = int(deep_get(cfg, "step2.max_las_curves", 0))
    d_step2_typewell_max_cells = int(deep_get(cfg, "step2.typewell_max_cells", 0))
    d_step2_typewell_max_kernel_wells = int(deep_get(cfg, "step2.typewell_max_kernel_wells", 200))
    d_step2_typewell_gc_every = int(deep_get(cfg, "step2.typewell_gc_every", 10))
    d_step2_resume = bool(deep_get(cfg, "step2.resume", True))
    d_step2_typewell_dict = deep_get(cfg, "step2.typewell", {})  # dict for TypeWellConfig
    d_step2_typewell_max_rows = int(deep_get(cfg, "step2.typewell.max_las_rows", 0))
    d_step2_typewell_use_subprocess = bool(deep_get(cfg, "step2.typewell.use_subprocess", False))
    d_step2_typewell_subprocess_min_mb = int(deep_get(cfg, "step2.typewell.subprocess_min_las_mb", 0))
    d_step2_typewell_worker_timeout = int(deep_get(cfg, "step2.typewell.worker_timeout_sec", 0))
    d_step2_typewell_worker_mem = int(deep_get(cfg, "step2.typewell.worker_mem_mb", 0))
    d_step2_typewell_gr_cache_dir = str(deep_get(cfg, "step2.typewell.gr_cache_dir", ""))
    d_step2_typewell_gr_cache_read = bool(deep_get(cfg, "step2.typewell.gr_cache_read", True))
    d_step2_typewell_gr_cache_write = bool(deep_get(cfg, "step2.typewell.gr_cache_write", True))

    # -----------------------
    # Step 3 defaults (read config now; import Step3 lazily later)
    # -----------------------
    d_step3_out = deep_get(cfg, "outputs.step3_out_subdir", "03_corr")
    d_step3_mode = str(deep_get(cfg, "step3.mode", "full"))

    d_step3_graph_r_max_m = float(
        deep_get(cfg, "step3.graph_r_max_m", deep_get(cfg, "step3.graph.r_max_m", 5000.0))
    )
    d_step3_graph_k_max = int(deep_get(cfg, "step3.graph_k_max", deep_get(cfg, "step3.graph.k_max", 12)))
    d_step3_graph_ensure_one_nn = bool(
        deep_get(cfg, "step3.graph_ensure_one_nn", deep_get(cfg, "step3.graph.ensure_one_nn", True))
    )

    d_step3_k_intra = int(deep_get(cfg, "step3.k_intra", 6))
    d_step3_k_bin = int(deep_get(cfg, "step3.k_bin", 3))
    d_step3_m_bridge_pairs = int(deep_get(cfg, "step3.m_bridge_pairs", 2))
    d_step3_d_max_km = float(deep_get(cfg, "step3.d_max_km", 80.0))

    d_step3_n_samples = int(deep_get(cfg, "step3.n_samples", 400))
    d_step3_p_lo = float(deep_get(cfg, "step3.p_lo", 1.0))
    d_step3_p_hi = float(deep_get(cfg, "step3.p_hi", 99.0))
    d_step3_fill_nans = bool(deep_get(cfg, "step3.fill_nans", True))
    d_step3_min_finite = int(deep_get(cfg, "step3.min_finite", 20))
    d_step3_max_gap_depth = deep_get(cfg, "step3.max_gap_depth", None)

    d_step3_dtw_alpha = float(deep_get(cfg, "step3.dtw.alpha", 0.15))
    d_step3_dtw_band_rad = deep_get(cfg, "step3.dtw.band_rad", None)
    d_step3_dtw_min_finite = int(deep_get(cfg, "step3.dtw.min_finite", 20))
    d_step3_dtw_downsample = int(deep_get(cfg, "step3.dtw.downsample_path_to", 80))

    d_step3_fw_mode = str(deep_get(cfg, "step3.framework.mode", "mst_plus_topk"))
    d_step3_fw_topk = int(deep_get(cfg, "step3.framework.topk", 3))
    d_step3_fw_topk_extra = int(deep_get(cfg, "step3.framework.topk_extra", 3))
    d_step3_fw_sim_threshold = float(deep_get(cfg, "step3.framework.sim_threshold", 0.60))
    d_step3_fw_extra_sim_min = float(deep_get(cfg, "step3.framework.extra_sim_min", 0.0))
    d_step3_fw_sim_scale = float(deep_get(cfg, "step3.framework.sim_scale", 0.25))

    d_step3_rgt_damping = float(deep_get(cfg, "step3.rgt.damping", 1e-2))
    d_step3_rgt_maxiter = int(deep_get(cfg, "step3.rgt.maxiter", 500))
    d_step3_rgt_tol = float(deep_get(cfg, "step3.rgt.tol", 1e-6))
    d_step3_rgt_simplified = bool(deep_get(cfg, "step3.rgt.simplified_indexing", True))
    d_step3_rgt_lambda_anchor = float(deep_get(cfg, "step3.rgt.lambda_anchor", 1.0))

    # -----------------------
    # Step 4 defaults (read config now; import Step4 lazily later)
    # -----------------------
    d_step4_out = deep_get(cfg, "outputs.step4_out_subdir", "04_type_columns")
    d_step4_kernel_radius = int(deep_get(cfg, "step4.kernel_radius", 1))
    d_step4_kernel_radius_max = int(deep_get(cfg, "step4.kernel_radius_max", 3))
    d_step4_min_wells_per_cell = int(deep_get(cfg, "step4.min_wells_per_cell", 5))
    d_step4_max_wells_per_cell = int(deep_get(cfg, "step4.max_wells_per_cell", 200))
    d_step4_seed = int(deep_get(cfg, "step4.seed", 42))

    d_step4_chron_n_rgt = int(deep_get(cfg, "step4.chronostrat.n_rgt", 800))
    d_step4_chron_pad = float(deep_get(cfg, "step4.chronostrat.rgt_pad_frac", 0.02))
    d_step4_chron_monotonic = str(deep_get(cfg, "step4.chronostrat.monotonic_mode", "nondecreasing"))

    d_step4_cwt_widths = deep_get(cfg, "step4.cwt.widths", [2, 4, 6, 8, 12, 16, 24, 32])
    d_step4_cwt_snap = int(deep_get(cfg, "step4.cwt.snap_window", 25))
    d_step4_cwt_include_endpoints = bool(deep_get(cfg, "step4.cwt.include_endpoints", True))

    d_step4_ntg_cutoff = float(deep_get(cfg, "step4.interval_stats.ntg_cutoff", 0.40))
    d_step4_min_finite = int(deep_get(cfg, "step4.interval_stats.min_finite", 20))

    ap = argparse.ArgumentParser(description="Strataframe runner (configurable; modular steps).")
    ap.add_argument("--config", type=str, default=str(cfg_path) if cfg_path else "", help="Path to YAML config.")
    ap.add_argument("--steps", type=str, default=d_steps_s, help="Comma list of steps to run (default from config).")

    ap.add_argument("--ks-manifest", type=str, default=d_ks_manifest, help="Path to data/ks_las_files.txt")
    ap.add_argument("--las-root", type=str, default=d_las_root, help="Directory with local LAS files")
    ap.add_argument("--run-dir", type=str, default=d_run_dir, help="Run directory root (e.g., runs/runA)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    ap.add_argument("--no-overwrite", action="store_true", help="Force overwrite=false (overrides config/--overwrite)")

    # Step 0 overrides
    ap.add_argument("--step0-out-subdir", type=str, default=d_step0_out, help="Subdir under run-dir for Step0 outputs")
    ap.add_argument("--step0-curve-family", type=str, default=str(d_curve_family), help="Curve family (default GR)")
    ap.add_argument("--step0-min-finite", type=int, default=int(d_min_finite), help="Minimum finite GR samples")
    ap.add_argument("--step0-pct", type=str, default=",".join(str(x) for x in d_pct), help="Percentiles '1,50,99'")
    ap.add_argument("--step0-quiet", action="store_true", help="Quiet lasio output (forces quiet=true)")
    ap.add_argument("--step0-no-quiet", action="store_true", help="Forces quiet=false")
    ap.add_argument("--step0-progress-every", type=int, default=int(deep_get(cfg, "step0.progress_every", 250)))
    ap.add_argument("--step0-no-progress", action="store_true")
    ap.add_argument(
        "--step0-flush-every",
        type=int,
        default=int(deep_get(cfg, "step0.flush_every", 1000)),
        help="Write wells_gr parquet part every N OK wells (default 1000).",
    )

    # Step 1 overrides (GRID ONLY)
    ap.add_argument("--step1-out-subdir", type=str, default=d_step1_out, help="Subdir under run-dir for Step1 outputs")
    ap.add_argument("--step1-target-bins", type=int, default=int(d_step1_target_bins))
    ap.add_argument("--step1-min-bin-size", type=int, default=int(d_step1_min_bin_size))
    ap.add_argument("--step1-filter-wells-csv", type=str, default=str(d_step1_filter_csv or ""))

    ap.add_argument(
        "--step1-grid-cell-km",
        type=float,
        default=float(d_step1_grid_cell_km) if d_step1_grid_cell_km is not None else float("nan"),
        help="Explicit grid cell size (km). NaN/omit lets Step1 choose.",
    )
    ap.add_argument("--step1-grid-pad-frac", type=float, default=float(d_step1_grid_pad_frac))

    # Step 2 overrides
    ap.add_argument("--step2-out-subdir", type=str, default=d_step2_out, help="Subdir under run-dir for Step2 outputs")
    ap.add_argument("--step2-n-rep", type=int, default=int(d_step2_n_rep))
    ap.add_argument("--step2-quota-mode", choices=["equal", "proportional"], default=d_step2_quota_mode)
    ap.add_argument("--step2-q-min", type=int, default=int(d_step2_q_min))
    ap.add_argument("--step2-candidate-method", choices=["farthest", "random"], default=d_step2_candidate_method)
    ap.add_argument("--step2-seed", type=int, default=int(d_step2_seed))
    ap.add_argument("--step2-max-candidates-per-cell", type=int, default=int(d_step2_max_cand))
    ap.add_argument("--step2-no-require-gr", action="store_true", help="Disable GR requirement (not recommended).")

    ap.add_argument("--step2-grid-km", type=float, default=float(d_step2_grid_km))
    ap.add_argument("--step2-no-typewells", action="store_true", help="Skip local typewells + placement.")
    ap.add_argument("--step2-max-las-mb", type=int, default=int(d_step2_max_las_mb))
    ap.add_argument("--step2-max-curves", type=int, default=int(d_step2_max_las_curves))
    ap.add_argument("--step2-typewell-max-cells", type=int, default=int(d_step2_typewell_max_cells))
    ap.add_argument("--step2-typewell-max-kernel-wells", type=int, default=int(d_step2_typewell_max_kernel_wells))
    ap.add_argument("--step2-typewell-gc-every", type=int, default=int(d_step2_typewell_gc_every))
    ap.add_argument("--step2-no-resume", action="store_true", help="Disable resume mode for typewell outputs.")
    ap.add_argument("--step2-typewell-max-rows", type=int, default=int(d_step2_typewell_max_rows))
    ap.add_argument("--step2-typewell-subprocess", action="store_true", default=bool(d_step2_typewell_use_subprocess))
    ap.add_argument("--step2-typewell-subprocess-min-mb", type=int, default=int(d_step2_typewell_subprocess_min_mb))
    ap.add_argument("--step2-typewell-worker-timeout", type=int, default=int(d_step2_typewell_worker_timeout))
    ap.add_argument("--step2-typewell-worker-mem-mb", type=int, default=int(d_step2_typewell_worker_mem))
    ap.add_argument("--step2-typewell-gr-cache-dir", type=str, default=str(d_step2_typewell_gr_cache_dir))
    ap.add_argument("--step2-typewell-no-gr-cache", action="store_true")

    ap.add_argument(
        "--step2-well-to-cell-csv",
        type=str,
        default="",
        help="Optional explicit path to well_to_cell.csv; default is <run-dir>/<step1-out-subdir>/well_to_cell.csv",
    )
    ap.add_argument(
        "--step2-manifest-json",
        type=str,
        default="",
        help="Optional explicit path to Step1 manifest.json (used to rebuild well_to_cell.csv if missing).",
    )
    ap.add_argument(
        "--step2-force-rebuild-well-to-cell",
        action="store_true",
        help="Force rebuild well_to_cell.csv (bins + wells_gr must be discoverable via manifest/paths).",
    )
    ap.add_argument("--step2-dry-run", action="store_true")

    # Step 3 overrides (unchanged)
    ap.add_argument("--step3-out-subdir", type=str, default=d_step3_out, help="Subdir under run-dir for Step3 outputs")
    ap.add_argument(
        "--step3-mode",
        type=str,
        default=str(d_step3_mode),
        choices=["prep", "full"],
        help="Step3 mode: prep (candidates + rep arrays only) or full (DTW + framework + RGT)",
    )

    ap.add_argument("--step3-graph-r-max-m", type=float, default=float(d_step3_graph_r_max_m))
    ap.add_argument("--step3-graph-k-max", type=int, default=int(d_step3_graph_k_max))
    ap.add_argument("--step3-graph-ensure-one-nn", action="store_true")
    ap.add_argument("--step3-graph-no-ensure-one-nn", action="store_true")

    ap.add_argument("--step3-k-intra", type=int, default=int(d_step3_k_intra))
    ap.add_argument("--step3-k-bin", type=int, default=int(d_step3_k_bin))
    ap.add_argument("--step3-m-bridge-pairs", type=int, default=int(d_step3_m_bridge_pairs))
    ap.add_argument("--step3-d-max-km", type=float, default=float(d_step3_d_max_km))

    ap.add_argument("--step3-n-samples", type=int, default=int(d_step3_n_samples))
    ap.add_argument("--step3-p-lo", type=float, default=float(d_step3_p_lo))
    ap.add_argument("--step3-p-hi", type=float, default=float(d_step3_p_hi))
    ap.add_argument("--step3-no-fill-nans", action="store_true")
    ap.add_argument("--step3-min-finite", type=int, default=int(d_step3_min_finite))
    ap.add_argument(
        "--step3-max-gap-depth",
        type=float,
        default=float(d_step3_max_gap_depth) if d_step3_max_gap_depth is not None else float("nan"),
    )

    ap.add_argument("--step3-dtw-alpha", type=float, default=float(d_step3_dtw_alpha))
    ap.add_argument(
        "--step3-dtw-band-rad",
        type=int,
        default=int(d_step3_dtw_band_rad) if d_step3_dtw_band_rad is not None else -1,
    )
    ap.add_argument("--step3-dtw-min-finite", type=int, default=int(d_step3_dtw_min_finite))
    ap.add_argument("--step3-dtw-downsample", type=int, default=int(d_step3_dtw_downsample))

    ap.add_argument("--step3-fw-mode", type=str, default=str(d_step3_fw_mode))
    ap.add_argument("--step3-fw-topk", type=int, default=int(d_step3_fw_topk))
    ap.add_argument("--step3-fw-topk-extra", type=int, default=int(d_step3_fw_topk_extra))
    ap.add_argument("--step3-fw-sim-threshold", type=float, default=float(d_step3_fw_sim_threshold))
    ap.add_argument("--step3-fw-extra-sim-min", type=float, default=float(d_step3_fw_extra_sim_min))
    ap.add_argument("--step3-fw-sim-scale", type=float, default=float(d_step3_fw_sim_scale))

    ap.add_argument("--step3-rgt-damping", type=float, default=float(d_step3_rgt_damping))
    ap.add_argument("--step3-rgt-maxiter", type=int, default=int(d_step3_rgt_maxiter))
    ap.add_argument("--step3-rgt-tol", type=float, default=float(d_step3_rgt_tol))
    ap.add_argument("--step3-rgt-no-simplified", action="store_true")
    ap.add_argument("--step3-rgt-lambda-anchor", type=float, default=float(d_step3_rgt_lambda_anchor))

    # Step 4 overrides
    ap.add_argument("--step4-out-subdir", type=str, default=str(d_step4_out))
    ap.add_argument("--step4-kernel-radius", type=int, default=int(d_step4_kernel_radius))
    ap.add_argument("--step4-kernel-radius-max", type=int, default=int(d_step4_kernel_radius_max))
    ap.add_argument("--step4-min-wells-per-cell", type=int, default=int(d_step4_min_wells_per_cell))
    ap.add_argument("--step4-max-wells-per-cell", type=int, default=int(d_step4_max_wells_per_cell))
    ap.add_argument("--step4-seed", type=int, default=int(d_step4_seed))

    ap.add_argument("--step4-n-rgt", type=int, default=int(d_step4_chron_n_rgt))
    ap.add_argument("--step4-rgt-pad-frac", type=float, default=float(d_step4_chron_pad))
    ap.add_argument("--step4-monotonic-mode", type=str, default=str(d_step4_chron_monotonic))

    ap.add_argument(
        "--step4-cwt-widths",
        type=str,
        default=",".join(str(x) for x in d_step4_cwt_widths),
        help="Comma-separated CWT widths.",
    )
    ap.add_argument("--step4-cwt-snap-window", type=int, default=int(d_step4_cwt_snap))
    ap.add_argument("--step4-cwt-include-endpoints", action="store_true")
    ap.add_argument("--step4-cwt-no-include-endpoints", dest="step4_cwt_include_endpoints", action="store_false")
    ap.set_defaults(step4_cwt_include_endpoints=bool(d_step4_cwt_include_endpoints))

    ap.add_argument("--step4-ntg-cutoff", type=float, default=float(d_step4_ntg_cutoff))
    ap.add_argument("--step4-min-finite", type=int, default=int(d_step4_min_finite))

    ap.add_argument("--step3-reps-csv", type=str, default="")
    ap.add_argument("--step3-wells-gr-parquet", type=str, default="")

    args = ap.parse_args(argv2)

    steps = _parse_steps(args.steps)
    if not steps:
        steps = list(IMPLEMENTED_STEPS)

    steps = [s for s in steps if s in IMPLEMENTED_STEPS]
    if not steps:
        raise SystemExit("No implemented steps requested. Implemented: " + ",".join(IMPLEMENTED_STEPS))

    overwrite = d_overwrite
    if bool(args.overwrite):
        overwrite = True
    if bool(args.no_overwrite):
        overwrite = False

    # Step0 pct parse
    pct_vals: List[float] = []
    for tok in str(args.step0_pct).split(","):
        t = tok.strip()
        if not t:
            continue
        try:
            pct_vals.append(float(t))
        except Exception:
            raise SystemExit(f"Invalid --step0-pct value: {args.step0_pct}")
    if len(pct_vals) != 3:
        raise SystemExit(f"--step0-pct must have 3 numbers, got: {args.step0_pct}")

    quiet = d_quiet
    if bool(args.step0_quiet):
        quiet = True
    if bool(args.step0_no_quiet):
        quiet = False

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 0
    # -------------------------------------------------------------------------
    if "0" in steps:
        out_dir0 = run_dir / str(args.step0_out_subdir)
        cfg0 = Step0Config(
            curve_family=str(args.step0_curve_family),
            min_finite=int(args.step0_min_finite),
            pct=(float(pct_vals[0]), float(pct_vals[1]), float(pct_vals[2])),
            quiet_lasio=bool(quiet),
            progress=(not bool(args.step0_no_progress)),
            progress_every=int(args.step0_progress_every),
            flush_every=int(args.step0_flush_every),
        )

        diag0 = run_step0_index_gr(
            ks_manifest_path=Path(args.ks_manifest),
            las_root=Path(args.las_root),
            out_dir=out_dir0,
            cfg=cfg0,
            overwrite=bool(overwrite),
        )

        c = diag0.get("counts", {})
        print("Step 0 complete:")
        print(f"  keep: {c.get('n_keep', 0)}")
        print(f"  missing_las: {c.get('n_missing_las', 0)}")
        print(f"  no_depth: {c.get('n_no_depth', 0)}")
        print(f"  no_gr: {c.get('n_no_gr', 0)}")
        print(f"  insufficient_gr: {c.get('n_insufficient_gr', 0)}")
        print(f"  out: {out_dir0}")

    # -------------------------------------------------------------------------
    # Step 1 (GRID ONLY)
    # -------------------------------------------------------------------------
    if "1" in steps:
        out_dir1 = run_dir / str(args.step1_out_subdir)

        # Build Step1Config defensively (tolerate field evolution)
        f1 = _dataclass_fieldnames(Step1Config)

        grid_cell_km: Optional[float] = None
        try:
            v = float(args.step1_grid_cell_km)
            if np.isfinite(v) and v > 0:
                grid_cell_km = v
        except Exception:
            grid_cell_km = None

        cfg1_kwargs: Dict[str, Any] = {}
        if "method" in f1:
            cfg1_kwargs["method"] = "grid"
        if "target_bins" in f1:
            cfg1_kwargs["target_bins"] = int(args.step1_target_bins)
        if "min_bin_size" in f1:
            cfg1_kwargs["min_bin_size"] = int(args.step1_min_bin_size)
        if "filter_wells_csv" in f1:
            cfg1_kwargs["filter_wells_csv"] = str(args.step1_filter_wells_csv).strip() or None
        if "grid_cell_km" in f1:
            cfg1_kwargs["grid_cell_km"] = grid_cell_km
        if "grid_pad_frac" in f1:
            cfg1_kwargs["grid_pad_frac"] = float(args.step1_grid_pad_frac)

        cfg1 = Step1Config(**cfg1_kwargs)

        man1 = run_step1_build_bins(
            ks_manifest_path=Path(args.ks_manifest),
            out_dir=out_dir1,
            cfg=cfg1,
            overwrite=bool(overwrite),
        )

        # Ensure a concrete manifest file exists for Step2 rebuilds.
        try:
            (out_dir1 / "manifest.json").write_text(json.dumps(man1, indent=2), encoding="utf-8")
        except Exception:
            pass

        d1 = man1.get("diagnostics", {})
        print("Step 1 complete:")
        print(f"  out: {out_dir1}")
        if isinstance(d1, dict):
            if "n_bins_final" in d1:
                print(f"  bins_final: {d1.get('n_bins_final', 0)}")
            # Prefer explicit chosen/grid values if present
            if "chosen_grid_cell_km" in d1:
                print(f"  grid_cell_km: {d1.get('chosen_grid_cell_km', '')}")
            elif "grid_cell_km" in d1:
                print(f"  grid_cell_km: {d1.get('grid_cell_km', '')}")
            elif "grid_km" in d1:
                print(f"  grid_km: {d1.get('grid_km', '')}")

    # -------------------------------------------------------------------------
    # Step 2 (lazy import here)
    # -------------------------------------------------------------------------
    if "2" in steps:
        # Lazy import: Step2 may pull in optional deps.
        from strataframe.pipelines.step2_reps import Step2RepsConfig
        from strataframe.steps.step2_run import run_step2
        from strataframe.typewell.local_typewell import TypeWellConfig

        out_dir2 = run_dir / str(args.step2_out_subdir)

        # well_to_cell.csv location: explicit override, else Step1 output dir
        if str(args.step2_well_to_cell_csv).strip():
            well_to_cell_csv = Path(args.step2_well_to_cell_csv)
        else:
            well_to_cell_csv = run_dir / str(args.step1_out_subdir) / "well_to_cell.csv"

        # manifest.json: explicit override, else guess from Step1 output
        manifest_json: Optional[Path] = None
        if str(args.step2_manifest_json).strip():
            manifest_json = Path(args.step2_manifest_json)
        else:
            manifest_json = _guess_step1_manifest(run_dir, str(args.step1_out_subdir))

        # Build TypeWellConfig from YAML (step2.typewell.*)
        tw_cfg = _build_object_from_dict(TypeWellConfig, d_step2_typewell_dict)
        if int(args.step2_typewell_max_rows) >= 0:
            tw_cfg = replace(tw_cfg, max_las_rows=int(args.step2_typewell_max_rows))
        # typewell cache dir (auto default unless disabled)
        cache_dir = ""
        if bool(args.step2_typewell_no_gr_cache):
            cache_dir = ""
        elif str(args.step2_typewell_gr_cache_dir).strip():
            cache_dir = str(args.step2_typewell_gr_cache_dir).strip()
        elif str(d_step2_typewell_gr_cache_dir).strip():
            cache_dir = str(d_step2_typewell_gr_cache_dir).strip()
        else:
            cache_dir = str(out_dir2 / "gr_cache")

        tw_cfg = replace(
            tw_cfg,
            use_subprocess=bool(args.step2_typewell_subprocess),
            subprocess_min_las_mb=int(args.step2_typewell_subprocess_min_mb),
            worker_timeout_sec=int(args.step2_typewell_worker_timeout),
            worker_mem_mb=int(args.step2_typewell_worker_mem_mb),
            gr_cache_dir=str(cache_dir),
            gr_cache_read=bool(d_step2_typewell_gr_cache_read),
            gr_cache_write=bool(d_step2_typewell_gr_cache_write),
        )

        # Build Step2 config defensively (tolerate field evolution)
        f2 = _dataclass_fieldnames(Step2RepsConfig)
        cfg2_kwargs: Dict[str, Any] = {
            "n_rep": int(args.step2_n_rep),
            "quota_mode": str(args.step2_quota_mode),
            "q_min": int(args.step2_q_min),
            "candidate_method": str(args.step2_candidate_method),
            "seed": int(args.step2_seed),
            "max_candidates_per_cell": int(args.step2_max_candidates_per_cell),
            "require_gr": (not bool(args.step2_no_require_gr)),
            "grid_km": float(args.step2_grid_km),
            "build_typewells": (not bool(args.step2_no_typewells)) and bool(d_step2_build_typewells),
            "typewell": tw_cfg,
            "max_las_mb": int(args.step2_max_las_mb),
            "max_las_curves": int(args.step2_max_curves),
            "typewell_max_cells": int(args.step2_typewell_max_cells),
            "typewell_max_kernel_wells": int(args.step2_typewell_max_kernel_wells),
            "typewell_gc_every": int(args.step2_typewell_gc_every),
            "resume": (not bool(args.step2_no_resume)) and bool(d_step2_resume),
        }
        if f2:
            cfg2_kwargs = {k: v for k, v in cfg2_kwargs.items() if k in f2}
        cfg2 = Step2RepsConfig(**cfg2_kwargs)

        reps2, diag2 = run_step2(
            well_to_cell_csv=well_to_cell_csv,
            las_root=Path(args.las_root),
            out_dir=out_dir2,
            cfg=cfg2,
            dry_run=bool(args.step2_dry_run),
            manifest_json=manifest_json if (manifest_json and manifest_json.exists()) else None,
            force_rebuild_well_to_cell=bool(args.step2_force_rebuild_well_to_cell),
        )

        print("Step 2 complete:")
        try:
            n_selected = len(reps2)
        except Exception:
            n_selected = int(getattr(diag2, "get", lambda *_: 0)("n_rep_selected", 0))  # type: ignore
        print(f"  reps_selected: {n_selected}")
        print(f"  out: {out_dir2}")

    # -------------------------------------------------------------------------
    # Step 3 (lazy import here)
    # -------------------------------------------------------------------------
    if "3" in steps:
        from strataframe.steps.step3_run import Step3Config, run_step3  # noqa: WPS433

        out_dir3 = run_dir / str(args.step3_out_subdir)

        if str(args.step3_reps_csv).strip():
            reps_csv = Path(args.step3_reps_csv)
        else:
            c1 = run_dir / str(args.step2_out_subdir) / "reps.csv"
            c2 = run_dir / str(args.step2_out_subdir) / "reps_selected.csv"
            c3 = run_dir / str(args.step2_out_subdir) / "representatives.csv"
            reps_csv = c1 if c1.exists() else (c2 if c2.exists() else c3)

        if not reps_csv.exists():
            raise SystemExit(
                "Step 3 requires Step2 reps CSV. "
                f"Not found: {reps_csv}. "
                "Run Step 2 first or pass --step3-reps-csv."
            )

        if str(args.step3_wells_gr_parquet).strip():
            wells_gr_parquet = Path(args.step3_wells_gr_parquet)
        else:
            wells_gr_parquet = run_dir / str(args.step0_out_subdir) / "wells_gr.parquet"

        step3_fill_nans = not bool(args.step3_no_fill_nans)

        max_gap_depth = None
        if hasattr(args, "step3_max_gap_depth"):
            try:
                v = float(args.step3_max_gap_depth)
                if np.isfinite(v):
                    max_gap_depth = v
            except Exception:
                max_gap_depth = None

        band_rad = None
        if int(args.step3_dtw_band_rad) >= 0:
            band_rad = int(args.step3_dtw_band_rad)

        cg_ensure_one_nn = (
            False
            if bool(args.step3_graph_no_ensure_one_nn)
            else True
            if bool(args.step3_graph_ensure_one_nn)
            else bool(d_step3_graph_ensure_one_nn)
        )
        cfg3 = Step3Config(
            reps_csv=reps_csv,
            wells_gr_parquet=wells_gr_parquet if wells_gr_parquet.exists() else None,
            las_root=Path(args.las_root),
            mode=str(args.step3_mode),
            cg_k_max=int(args.step3_graph_k_max),
            cg_r_max_km=float(args.step3_graph_r_max_m) / 1000.0,
            cg_ensure_one_nn=bool(cg_ensure_one_nn),
            n_samples=int(args.step3_n_samples),
            p_lo=float(args.step3_p_lo),
            p_hi=float(args.step3_p_hi),
            fill_nans=bool(step3_fill_nans),
            min_finite=int(args.step3_min_finite),
            max_gap_depth=max_gap_depth,
            dtw_alpha=float(args.step3_dtw_alpha),
            dtw_band_rad=band_rad,
            dtw_min_finite=int(args.step3_dtw_min_finite),
            dtw_downsample_path_to=int(args.step3_dtw_downsample),
            fw_mode=str(args.step3_fw_mode),
            fw_topk=int(args.step3_fw_topk),
            fw_topk_extra=int(args.step3_fw_topk_extra),
            fw_sim_threshold=float(args.step3_fw_sim_threshold),
            fw_extra_sim_min=float(args.step3_fw_extra_sim_min),
            fw_sim_scale=float(args.step3_fw_sim_scale),
            rgt_damping=float(args.step3_rgt_damping),
            rgt_maxiter=int(args.step3_rgt_maxiter),
            rgt_tol=float(args.step3_rgt_tol),
            rgt_simplified_indexing=(not bool(args.step3_rgt_no_simplified)),
            rgt_lambda_anchor=float(args.step3_rgt_lambda_anchor),
        )

        diag3 = run_step3(
            out_dir=out_dir3,
            cfg=cfg3,
            overwrite=bool(overwrite),
        )

        print("Step 3 complete:")
        print(f"  out: {out_dir3}")
        try:
            print(f"  reps_usable: {diag3.get('counts', {}).get('n_reps_usable', 0)}")
            print(f"  candidates: {diag3.get('counts', {}).get('n_candidates', 0)}")
            print(f"  dtw_ok: {diag3.get('counts', {}).get('n_dtw_ok', 0)}")
            print(f"  framework_edges: {diag3.get('counts', {}).get('n_framework_edges', 0)}")
            print(f"  rgt_components: {diag3.get('counts', {}).get('n_components', 0)}")
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Step 4 (lazy import here)
    # -------------------------------------------------------------------------
    if "4" in steps:
        from strataframe.pipelines.step4_type_columns import (  # noqa: WPS433
            IntervalStatsConfig,
            Step4Config,
            run_step4_type_columns,
        )
        from strataframe.rgt.chronostrat import ChronostratConfig  # noqa: WPS433
        from strataframe.rgt.monotonic import MonotonicConfig  # noqa: WPS433
        from strataframe.rgt.wavelet_tops import CwtConfig  # noqa: WPS433

        out_dir4 = run_dir / str(args.step4_out_subdir)

        step3_dir = run_dir / str(args.step3_out_subdir)
        nodes_csv = step3_dir / "framework_nodes.csv"
        edges_csv = step3_dir / "framework_edges.csv"
        rep_arrays_npz = step3_dir / "rep_arrays.npz"
        shifts_npz = step3_dir / "rgt_shifts_resampled.npz"

        if not nodes_csv.exists() or not edges_csv.exists() or not rep_arrays_npz.exists() or not shifts_npz.exists():
            raise SystemExit(
                "Step 4 requires Step 3 outputs (framework_nodes.csv, framework_edges.csv, "
                "rep_arrays.npz, rgt_shifts_resampled.npz). Run Step 3 first."
            )

        widths = [int(x.strip()) for x in str(args.step4_cwt_widths).split(",") if x.strip() != ""]

        chron = ChronostratConfig(
            n_rgt=int(args.step4_n_rgt),
            rgt_pad_frac=float(args.step4_rgt_pad_frac),
            monotonic=MonotonicConfig(mode=str(args.step4_monotonic_mode)),
        )
        cwt = CwtConfig(
            widths=tuple(widths),
            snap_window=int(args.step4_cwt_snap_window),
            include_endpoints=bool(args.step4_cwt_include_endpoints),
        )
        interval = IntervalStatsConfig(ntg_cutoff=float(args.step4_ntg_cutoff), min_finite=int(args.step4_min_finite))

        cfg4 = Step4Config(
            framework_nodes_csv=nodes_csv,
            framework_edges_csv=edges_csv,
            rep_arrays_npz=rep_arrays_npz,
            rgt_shifts_npz=shifts_npz,
            kernel_radius=int(args.step4_kernel_radius),
            kernel_radius_max=int(args.step4_kernel_radius_max),
            min_wells_per_cell=int(args.step4_min_wells_per_cell),
            max_wells_per_cell=int(args.step4_max_wells_per_cell),
            seed=int(args.step4_seed),
            chronostrat=chron,
            cwt=cwt,
            interval_stats=interval,
        )

        diag4 = run_step4_type_columns(out_dir=out_dir4, cfg=cfg4, overwrite=bool(overwrite))
        print("Step 4 complete:")
        print(f"  out: {out_dir4}")
        try:
            print(f"  cells_ok: {diag4.get('counts', {}).get('n_cells_ok', 0)}")
            print(f"  cells_fail: {diag4.get('counts', {}).get('n_cells_fail', 0)}")
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
