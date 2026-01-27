# src/strataframe/run_build_framework.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover
    nx = None  # type: ignore

from strataframe.io.csv import read_csv_rows, write_csv, to_float

from strataframe.graph.las_utils import (
    read_las_normal,
    extract_depth_and_curve,
    resample_and_normalize_curve,
    normalize_mnemonic,
)

from strataframe.correlation.framework import FrameworkConfig, add_similarity_scores, prune_to_framework
from strataframe.rgt.rgt import RgtConfig, solve_rgt_shifts


# -----------------------------------------------------------------------------
# Requirements
# -----------------------------------------------------------------------------

def _require_nx() -> None:
    if nx is None:
        raise RuntimeError("networkx is required. Install with: pip install networkx")


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def _edge_key(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a < b else (b, a)


def _safe_int(x: str, default: int = 0) -> int:
    try:
        return int((x or "").strip())
    except Exception:
        return default


def _safe_float(x: str, default: float = float("nan")) -> float:
    try:
        return float((x or "").strip())
    except Exception:
        return default


def _read_jsonl_paths(path: Path) -> Dict[Tuple[str, str], np.ndarray]:
    """
    Reads dtw_paths_tiepoints.jsonl from graph/build_sparse_dtw_edges.py --emit-paths
    Returns dict keyed by (src_rep_id, dst_rep_id) as strings (sorted undirected),
    value is (K,2) int64 array of (i,j) indices.
    """
    out: Dict[Tuple[str, str], np.ndarray] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            a = str(obj.get("src_rep_id", ""))
            b = str(obj.get("dst_rep_id", ""))
            if not a or not b:
                continue
            tp = obj.get("tiepoints_ij", [])
            try:
                arr = np.asarray(tp, dtype="int64")
                if arr.ndim != 2 or arr.shape[1] != 2:
                    continue
            except Exception:
                continue
            out[_edge_key(a, b)] = arr
    return out


# -----------------------------------------------------------------------------
# Node loading (LAS -> resampled depth/log arrays)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class NodeLoadConfig:
    n_samples: int
    # If reps csv has picked_gr, we prefer it; else fallback_curve.
    fallback_curve: str = "GR"
    # Percentiles for robust scaling (must match las_utils default if you want consistency)
    p_lo: float = 1.0
    p_hi: float = 99.0


def _load_node_arrays_from_las(
    las_path: Path,
    *,
    curve_mnemonic: str,
    cfg: NodeLoadConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      depth_rs: (n_samples,) float64 (in original depth units)
      log_rs:   (n_samples,) float64 (normalized to [0,1])
    """
    las = read_las_normal(las_path)
    depth, y = extract_depth_and_curve(las, curve_mnemonic=curve_mnemonic)
    x_norm, z_top, z_base = resample_and_normalize_curve(
        depth,
        y,
        n_samples=int(cfg.n_samples),
        p_lo=float(cfg.p_lo),
        p_hi=float(cfg.p_hi),
    )
    depth_rs = np.linspace(float(z_top), float(z_base), int(cfg.n_samples), dtype="float64")
    return depth_rs, x_norm


# -----------------------------------------------------------------------------
# Build graph from reps + dtw edges (+ optional paths)
# -----------------------------------------------------------------------------

def build_graph_from_artifacts(
    reps_csv: Path,
    dtw_edges_csv: Path,
    *,
    dtw_paths_jsonl: Optional[Path],
    fallback_curve: str,
    n_samples: Optional[int],
    p_lo: float,
    p_hi: float,
) -> Tuple["nx.Graph", Dict[str, Any]]:
    """
    Constructs a networkx graph where:
      node_id == rep_id (string)
      node has: lat, lon, las_path, depth_rs, log_rs, curve_used, h3_cell, score
      edge has: dist_km, dtw_cost, dtw_cost_per_step, curve_src, curve_dst, dtw_path(optional)
    """
    _require_nx()

    reps = read_csv_rows(reps_csv)
    edges_rows = read_csv_rows(dtw_edges_csv)

    if not reps:
        raise ValueError(f"No rows in reps csv: {reps_csv}")
    if not edges_rows:
        raise ValueError(f"No rows in dtw edges csv: {dtw_edges_csv}")

    # Determine n_samples (prefer user-provided, else infer from dtw_edges.csv)
    inferred_ns: Optional[int] = None
    for r in edges_rows:
        ns = _safe_int(r.get("n_samples", ""))
        if ns > 0:
            inferred_ns = ns
            break
    ns_final = int(n_samples) if (n_samples is not None and int(n_samples) > 0) else int(inferred_ns or 0)
    if ns_final <= 0:
        raise ValueError(
            "Could not determine n_samples. Provide --n-samples, or ensure dtw_edges.csv has n_samples."
        )

    node_cfg = NodeLoadConfig(n_samples=ns_final, fallback_curve=str(fallback_curve), p_lo=p_lo, p_hi=p_hi)

    # Load optional paths
    paths_by_edge: Dict[Tuple[str, str], np.ndarray] = {}
    if dtw_paths_jsonl is not None:
        if not dtw_paths_jsonl.exists():
            raise FileNotFoundError(f"dtw paths jsonl not found: {dtw_paths_jsonl}")
        paths_by_edge = _read_jsonl_paths(dtw_paths_jsonl)

    # Build quick lookup for reps by rep_id
    rep_by_id: Dict[str, Dict[str, str]] = {}
    for r in reps:
        rid = str(r.get("rep_id", "") or "").strip()
        if not rid:
            continue
        rep_by_id[rid] = r

    if len(rep_by_id) < 2:
        raise ValueError("Need at least 2 representatives with rep_id.")

    # Build curve hint per rep_id from edges file (matches what DTW actually used)
    curve_hint: Dict[str, str] = {}
    for er in edges_rows:
        a = str(er.get("src_rep_id", "") or "").strip()
        b = str(er.get("dst_rep_id", "") or "").strip()
        if a:
            cs = str(er.get("curve_src", "") or "").strip()
            if cs:
                curve_hint[a] = normalize_mnemonic(cs)
        if b:
            cd = str(er.get("curve_dst", "") or "").strip()
            if cd:
                curve_hint[b] = normalize_mnemonic(cd)

    # Graph
    G = nx.Graph()

    diag: Dict[str, Any] = {
        "inputs": {
            "reps_csv": str(reps_csv),
            "dtw_edges_csv": str(dtw_edges_csv),
            "dtw_paths_jsonl": str(dtw_paths_jsonl) if dtw_paths_jsonl is not None else "",
        },
        "n_samples": int(ns_final),
        "p_lo": float(p_lo),
        "p_hi": float(p_hi),
        "n_reps_rows": int(len(reps)),
        "n_reps_unique": int(len(rep_by_id)),
        "n_edges_rows": int(len(edges_rows)),
        "nodes_loaded": 0,
        "nodes_failed": 0,
        "node_failures": [],
        "edges_added": 0,
        "edges_skipped_missing_node": 0,
        "edges_with_paths": 0,
        "edges_without_paths": 0,
    }

    # Load nodes (LAS once per rep)
    for rid, r in sorted(rep_by_id.items(), key=lambda t: int(t[0]) if t[0].isdigit() else t[0]):
        las_path_s = str(r.get("las_path", "") or "").strip()
        las_path = Path(las_path_s) if las_path_s else Path()
        if not las_path_s or not las_path.exists():
            diag["nodes_failed"] += 1
            diag["node_failures"].append({"rep_id": rid, "stage": "las_path", "error": f"Missing LAS: {las_path_s}"})
            continue

        lat = to_float(r.get("lat", "") or "")
        lon = to_float(r.get("lon", "") or "")
        if lat is None or lon is None:
            # Still load arrays; coords used mainly for plotting / ordering later
            lat_v = float("nan")
            lon_v = float("nan")
        else:
            lat_v = float(lat)
            lon_v = float(lon)

        # Choose curve mnemonic:
        # 1) what DTW build actually used (curve_hint),
        # 2) picked_gr from reps,
        # 3) fallback_curve.
        picked_gr = normalize_mnemonic(str(r.get("picked_gr", "") or ""))
        curve_used = curve_hint.get(rid) or picked_gr or normalize_mnemonic(fallback_curve)

        try:
            depth_rs, log_rs = _load_node_arrays_from_las(las_path, curve_mnemonic=curve_used, cfg=node_cfg)
        except Exception as e:
            diag["nodes_failed"] += 1
            diag["node_failures"].append({"rep_id": rid, "stage": "load_arrays", "curve": curve_used, "error": str(e)})
            continue

        G.add_node(
            rid,
            rep_id=rid,
            h3_cell=str(r.get("h3_cell", "") or ""),
            h3_res=str(r.get("h3_res", "") or ""),
            score=str(r.get("score", "") or ""),
            url=str(r.get("url", "") or ""),
            kgs_id=str(r.get("kgs_id", "") or ""),
            api=str(r.get("api", "") or ""),
            api_num_nodash=str(r.get("api_num_nodash", "") or ""),
            operator=str(r.get("operator", "") or ""),
            lease=str(r.get("lease", "") or ""),
            lat=float(lat_v),
            lon=float(lon_v),
            las_path=str(las_path),
            curve_used=str(curve_used),
            depth_rs=depth_rs,
            log_rs=log_rs,
        )
        diag["nodes_loaded"] += 1

    # Add edges from dtw_edges.csv
    for er in edges_rows:
        a = str(er.get("src_rep_id", "") or "").strip()
        b = str(er.get("dst_rep_id", "") or "").strip()
        if not a or not b:
            continue
        if not G.has_node(a) or not G.has_node(b):
            diag["edges_skipped_missing_node"] += 1
            continue

        dist_km = _safe_float(er.get("dist_km", ""))
        dtw_cost = _safe_float(er.get("dtw_cost", ""))
        dtw_cps = _safe_float(er.get("dtw_cost_per_step", ""))

        attrs: Dict[str, Any] = {
            "dist_km": float(dist_km) if np.isfinite(dist_km) else np.nan,
            "dtw_cost": float(dtw_cost) if np.isfinite(dtw_cost) else np.nan,
            "dtw_cost_per_step": float(dtw_cps) if np.isfinite(dtw_cps) else np.nan,
            "curve_src": str(er.get("curve_src", "") or ""),
            "curve_dst": str(er.get("curve_dst", "") or ""),
            "n_samples": int(ns_final),
            "alpha": _safe_float(er.get("alpha", "")),
        }

        k = _edge_key(a, b)
        wp = paths_by_edge.get(k)
        if wp is not None:
            attrs["dtw_path"] = wp
            attrs["dtw_steps"] = int(wp.shape[0])
            diag["edges_with_paths"] += 1
        else:
            attrs["dtw_path"] = None
            attrs["dtw_steps"] = 0
            diag["edges_without_paths"] += 1

        # Dedup: keep the edge with smaller dtw_cost_per_step if repeated
        if G.has_edge(a, b):
            prev = G.edges[a, b]
            prev_cps = float(prev.get("dtw_cost_per_step", np.nan))
            new_cps = float(attrs.get("dtw_cost_per_step", np.nan))
            if np.isfinite(new_cps) and (not np.isfinite(prev_cps) or new_cps < prev_cps):
                G.edges[a, b].update(attrs)
        else:
            G.add_edge(a, b, **attrs)

        diag["edges_added"] += 1

    return G, diag


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Build ChronoLog-style framework + RGT shifts from artifacts only:\n"
            "  - representatives.csv (graph/select_representatives.py)\n"
            "  - dtw_edges.csv (+ optional dtw_paths_tiepoints.jsonl) (graph/build_sparse_dtw_edges.py)\n"
            "No legacy selection code, no on-the-fly DTW, no Delaunay."
        )
    )

    ap.add_argument("--reps-csv", type=str, required=True, help="Path to representatives.csv")
    ap.add_argument("--dtw-edges-csv", type=str, required=True, help="Path to dtw_edges.csv")
    ap.add_argument("--dtw-paths-jsonl", type=str, default="", help="Optional path to dtw_paths_tiepoints.jsonl (recommended for RGT)")

    ap.add_argument("--out-dir", type=str, required=True, help="Output directory")

    ap.add_argument("--curve", type=str, default="GR", help="Fallback curve mnemonic if reps have no picked_gr (default GR)")
    ap.add_argument("--n-samples", type=int, default=0, help="Override resample length (default: infer from dtw_edges.csv)")
    ap.add_argument("--p-lo", type=float, default=1.0, help="Low percentile for normalization (default 1)")
    ap.add_argument("--p-hi", type=float, default=99.0, help="High percentile for normalization (default 99)")

    # Framework pruning
    ap.add_argument("--framework", type=str, default="mst", help="mst|topk|threshold")
    ap.add_argument("--topk", type=int, default=3, help="topk mode: edges per node")
    ap.add_argument("--sim-threshold", type=float, default=0.60, help="threshold mode: similarity cutoff")
    ap.add_argument("--sim-scale", type=float, default=0.25, help="Similarity scale in exp(-cps/scale)")

    # RGT solve
    ap.add_argument("--rgt-damping", type=float, default=1e-2, help="RGT solve damping")
    ap.add_argument("--rgt-maxiter", type=int, default=500, help="RGT CG max iterations")
    ap.add_argument("--rgt-tol", type=float, default=1e-6, help="RGT CG tolerance")
    ap.add_argument("--rgt-precise", action="store_true", help="Use precise DTW indexing (slower; needs real dtw_path indices)")

    args = ap.parse_args(argv)

    _require_nx()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reps_csv = Path(args.reps_csv)
    dtw_edges_csv = Path(args.dtw_edges_csv)
    dtw_paths_jsonl = Path(args.dtw_paths_jsonl) if (args.dtw_paths_jsonl or "").strip() else None

    G, diag = build_graph_from_artifacts(
        reps_csv=reps_csv,
        dtw_edges_csv=dtw_edges_csv,
        dtw_paths_jsonl=dtw_paths_jsonl,
        fallback_curve=str(args.curve),
        n_samples=(int(args.n_samples) if int(args.n_samples) > 0 else None),
        p_lo=float(args.p_lo),
        p_hi=float(args.p_hi),
    )

    if G.number_of_nodes() < 3:
        raise SystemExit(f"Loaded only {G.number_of_nodes()} nodes; need at least 3.")

    # Similarity + prune
    fw_cfg = FrameworkConfig(
        mode=str(args.framework).strip().lower(),
        topk=int(args.topk),
        sim_threshold=float(args.sim_threshold),
        sim_scale=float(args.sim_scale),
    )
    add_similarity_scores(G, scale=float(fw_cfg.sim_scale))
    F = prune_to_framework(G, cfg=fw_cfg)

    # RGT (only if we have paths)
    have_any_paths = any(ed.get("dtw_path", None) is not None for _, _, ed in F.edges(data=True))
    shifts: Dict[str, np.ndarray] = {}

    if have_any_paths:
        rgt_cfg = RgtConfig(
            damping=float(args.rgt_damping),
            maxiter=int(args.rgt_maxiter),
            tol=float(args.rgt_tol),
            simplified_indexing=(not bool(args.rgt_precise)),
        )
        shifts = solve_rgt_shifts(F, cfg=rgt_cfg)
        np.savez_compressed(out_dir / "rgt_shifts_resampled.npz", **{k: v for k, v in shifts.items()})
        diag["rgt"] = {"ok": True, "config": rgt_cfg.__dict__}
    else:
        diag["rgt"] = {
            "ok": False,
            "reason": "No dtw_path on any framework edge. Re-run sparse DTW step with --emit-paths and pass --dtw-paths-jsonl.",
        }

    # Write outputs
    node_rows: List[Dict[str, Any]] = []
    for n, nd in F.nodes(data=True):
        node_rows.append(
            {
                "rep_id": n,
                "h3_cell": nd.get("h3_cell", ""),
                "h3_res": nd.get("h3_res", ""),
                "lat": f"{float(nd.get('lat', np.nan)):.8f}" if np.isfinite(float(nd.get("lat", np.nan))) else "",
                "lon": f"{float(nd.get('lon', np.nan)):.8f}" if np.isfinite(float(nd.get("lon", np.nan))) else "",
                "las_path": nd.get("las_path", ""),
                "curve_used": nd.get("curve_used", ""),
                "score": nd.get("score", ""),
            }
        )
    write_csv(
        out_dir / "framework_nodes.csv",
        ["rep_id", "h3_cell", "h3_res", "lat", "lon", "las_path", "curve_used", "score"],
        node_rows,
    )

    edge_rows: List[Dict[str, Any]] = []
    for u, v, ed in F.edges(data=True):
        edge_rows.append(
            {
                "src_rep_id": u,
                "dst_rep_id": v,
                "dist_km": f"{float(ed.get('dist_km', np.nan)):.3f}" if np.isfinite(float(ed.get("dist_km", np.nan))) else "",
                "dtw_cost": f"{float(ed.get('dtw_cost', np.nan)):.6g}" if np.isfinite(float(ed.get("dtw_cost", np.nan))) else "",
                "dtw_cost_per_step": f"{float(ed.get('dtw_cost_per_step', np.nan)):.6g}" if np.isfinite(float(ed.get("dtw_cost_per_step", np.nan))) else "",
                "sim": f"{float(ed.get('sim', np.nan)):.6g}" if np.isfinite(float(ed.get("sim", np.nan))) else "",
                "dtw_steps": int(ed.get("dtw_steps", 0)),
                "curve_src": ed.get("curve_src", ""),
                "curve_dst": ed.get("curve_dst", ""),
            }
        )
    write_csv(
        out_dir / "framework_edges.csv",
        ["src_rep_id", "dst_rep_id", "dist_km", "dtw_cost", "dtw_cost_per_step", "sim", "dtw_steps", "curve_src", "curve_dst"],
        edge_rows,
    )

    diag["framework"] = {
        "mode": fw_cfg.mode,
        "n_nodes_in": int(G.number_of_nodes()),
        "n_edges_in": int(G.number_of_edges()),
        "n_nodes_out": int(F.number_of_nodes()),
        "n_edges_out": int(F.number_of_edges()),
        "config": fw_cfg.__dict__,
    }

    (out_dir / "framework_diagnostics.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")

    print(f"Nodes loaded: {diag.get('nodes_loaded', 0)} (failed: {diag.get('nodes_failed', 0)})")
    print(f"Edges in: {G.number_of_edges()}  -> framework: {F.number_of_edges()} ({fw_cfg.mode})")
    if diag.get("rgt", {}).get("ok"):
        print(f"Wrote: {out_dir / 'rgt_shifts_resampled.npz'}")
    else:
        print("RGT not solved (missing dtw paths).")
    print(f"Wrote: {out_dir / 'framework_nodes.csv'}")
    print(f"Wrote: {out_dir / 'framework_edges.csv'}")
    print(f"Wrote: {out_dir / 'framework_diagnostics.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
