# src/strataframe/graph/build_sparse_dtw_edges.py
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .las_utils import (
    downsample_path,
    dtw_cost_and_path,
    extract_depth_and_curve,
    read_las_normal,
    resample_and_normalize_curve,
)

# Canonical header normalization / alias families (single source of truth)
from strataframe.curves.normalize_header import (
    aliases_for,
    norm_mnemonic as canon_mnemonic,
)

from strataframe.io.csv import read_csv_rows, write_csv, to_float
from strataframe.spatial.geodesy import haversine_km


# -----------------------------------------------------------------------------
# Config / cache
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SparseDTWConfig:
    # Canonical request only. Any alias/header resolution happens via aliases_for().
    curve_canonical: str = "GR"

    n_samples: int = 256
    alpha: float = 0.15

    max_km: float = 20.0
    knn_bridges: int = 1
    bridge_max_km: float = 60.0

    emit_paths: bool = False
    n_tiepoints: int = 64


def _safe_extract_depth_and_curve(
    las: Any,
    *,
    curve_mnemonic: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Be tolerant to extract_depth_and_curve returning:
      - (depth, curve)
      - (depth, curve, meta)
      - (depth, curve, unit, meta) or similar

    Returns: depth, curve, meta_dict
    """
    out = extract_depth_and_curve(las, curve_mnemonic=curve_mnemonic)

    # Common cases: tuple/list with >=2 items
    if isinstance(out, (tuple, list)):
        if len(out) < 2:
            raise RuntimeError(f"extract_depth_and_curve returned too few values: {len(out)}")
        depth = np.asarray(out[0])
        curve = np.asarray(out[1])

        meta: Dict[str, Any] = {}
        if len(out) >= 3:
            # third item might be unit (str) or a dict-like meta
            if isinstance(out[2], dict):
                meta.update(out[2])
            else:
                meta["extra_3"] = out[2]
        if len(out) >= 4:
            if isinstance(out[3], dict):
                meta.update(out[3])
            else:
                meta["extra_4"] = out[3]
        return depth, curve, meta

    # If a dict-like is returned (less common), try keys
    if isinstance(out, dict):
        if "depth" not in out or ("curve" not in out and "values" not in out):
            raise RuntimeError("extract_depth_and_curve returned dict without expected keys")
        depth = np.asarray(out["depth"])
        curve = np.asarray(out.get("curve", out.get("values")))
        meta = {k: v for k, v in out.items() if k not in {"depth", "curve", "values"}}
        return depth, curve, meta

    raise RuntimeError(f"Unsupported extract_depth_and_curve return type: {type(out)}")


def _iter_las_curve_mnemonics_raw(las: Any) -> List[str]:
    """
    Return raw header mnemonics in LAS curve order.
    We do not canonicalize here; this is just the header tokens.
    """
    out: List[str] = []
    for c in getattr(las, "curves", []) or []:
        try:
            out.append(str(getattr(c, "mnemonic", "") or "").strip())
        except Exception:
            continue
    return [m for m in out if m]


def resolve_curve_mnemonic_raw(
    las: Any,
    *,
    requested_canonical: str,
) -> Tuple[str, str]:
    """
    Resolve an *actual* LAS header mnemonic to use for extraction, given a canonical request.

    Returns:
      (raw_mnemonic_to_index, canonical_mnemonic)

    Resolution order:
      1) exact raw header token match against aliases_for(canon) (case/format tolerant via canon_mnemonic)
      2) first curve whose canonical maps to requested canonical (canon_mnemonic(raw)==canon)
    """
    canon = canon_mnemonic(requested_canonical)
    if not canon:
        raise ValueError(f"Empty/invalid requested_canonical: {requested_canonical!r}")

    raw_list = _iter_las_curve_mnemonics_raw(las)
    if not raw_list:
        raise RuntimeError("LAS has no curves to resolve")

    # Canonical forms of raw header mnemonics
    raw_can = [canon_mnemonic(m) for m in raw_list]

    # Prefer aliases_for(canon) ordering (but compare canon-to-canon)
    try:
        fam = aliases_for(canon)
    except Exception:
        fam = [canon]

    fam_can = [canon_mnemonic(a) for a in fam if a]
    fam_can = [x for x in fam_can if x]  # drop empties

    # (1) If any raw curve canonical matches one of the family canonical tokens,
    # pick the first family token that appears.
    for fc in fam_can:
        for i, rc in enumerate(raw_can):
            if rc == fc:
                return raw_list[i], canon

    # (2) Otherwise, pick first curve that canonicalizes to requested canonical
    for i, rc in enumerate(raw_can):
        if rc == canon:
            return raw_list[i], canon

    raise KeyError(f"No curve found for canonical={canon} (aliases={fam})")


class WellCurveCache:
    def __init__(self, cfg: SparseDTWConfig):
        self.cfg = cfg
        self._cache: Dict[int, Dict[str, Any]] = {}

    def get(self, rep: Dict[str, str]) -> Dict[str, Any]:
        rep_id = int(rep["rep_id"])
        if rep_id in self._cache:
            return self._cache[rep_id]

        las_path = Path(rep["las_path"])
        if not las_path.exists():
            raise FileNotFoundError(f"LAS not found: {las_path}")

        las = read_las_normal(las_path)

        # Canonical-only request:
        # - prefer a canonical field if present in reps.csv
        # - else canonicalize any legacy "picked_gr" value
        # - else default to config curve_canonical
        picked_canon = (rep.get("picked_curve_canon", "") or "").strip()
        if picked_canon:
            req_canon = canon_mnemonic(picked_canon)
        else:
            legacy = (rep.get("picked_gr", "") or "").strip()
            req_canon = canon_mnemonic(legacy) if legacy else canon_mnemonic(self.cfg.curve_canonical)

        if not req_canon:
            req_canon = canon_mnemonic(self.cfg.curve_canonical)

        # Resolve actual header mnemonic before extraction/indexing
        raw_mn, canon_mn = resolve_curve_mnemonic_raw(las, requested_canonical=req_canon)

        try:
            depth, y, meta = _safe_extract_depth_and_curve(las, curve_mnemonic=raw_mn)
        except Exception as e:
            raise RuntimeError(
                f"Failed to extract curve for rep_id={rep_id} "
                f"requested_canon={canon_mn} resolved_raw={raw_mn} las={las_path} :: {e}"
            ) from e

        x_norm, z_top, z_base = resample_and_normalize_curve(depth, y, n_samples=int(self.cfg.n_samples))

        obj = {
            "rep_id": rep_id,
            # Keep CSV schema stable: store canonical mnemonic for downstream compares
            "curve_mnemonic": canon_mn,
            # Optional debug trace (kept internal; does not change CSV schema)
            "curve_mnemonic_raw": raw_mn,
            "x": x_norm,
            "z_top": float(z_top),
            "z_base": float(z_base),
        }
        self._cache[rep_id] = obj
        return obj


# -----------------------------------------------------------------------------
# Edge candidates
# -----------------------------------------------------------------------------

def build_candidate_edges(
    reps: List[Dict[str, str]],
    *,
    max_km: float,
    knn_bridges: int,
    bridge_max_km: float,
) -> List[Tuple[int, int, float]]:
    coords: Dict[int, Tuple[float, float]] = {}
    for r in reps:
        rid = int(r["rep_id"])
        lat = to_float(r.get("lat", "") or "")
        lon = to_float(r.get("lon", "") or "")
        if lat is None or lon is None:
            continue
        coords[rid] = (lat, lon)

    ids = sorted(coords.keys())
    if len(ids) < 2:
        return []

    edges: Dict[Tuple[int, int], float] = {}

    # Local edges
    for i, a in enumerate(ids):
        la, lo = coords[a]
        for b in ids[i + 1 :]:
            lb, lob = coords[b]
            d = haversine_km(la, lo, lb, lob)
            if d <= float(max_km):
                edges[(a, b)] = float(d)

    # Sparse bridges
    k = int(max(0, knn_bridges))
    if k > 0:
        for a in ids:
            la, lo = coords[a]
            dists: List[Tuple[float, int]] = []
            for b in ids:
                if b == a:
                    continue
                lb, lob = coords[b]
                dists.append((haversine_km(la, lo, lb, lob), b))
            dists.sort(key=lambda t: t[0])

            for d, b in dists[:k]:
                if d > float(bridge_max_km):
                    continue
                x, y = (a, b) if a < b else (b, a)
                edges.setdefault((x, y), float(d))

    return [(a, b, edges[(a, b)]) for (a, b) in sorted(edges.keys())]


# -----------------------------------------------------------------------------
# DTW build
# -----------------------------------------------------------------------------

def _safe_dtw_cost_and_path(
    x1: np.ndarray,
    x2: np.ndarray,
    *,
    alpha: float,
    backtrack: bool,
) -> Tuple[float, float, Optional[np.ndarray]]:
    """
    Tolerate dtw_cost_and_path returning:
      - (total_cost, cost_per_step, path)
      - (total_cost, path)  [cost_per_step computed]
      - dict-like objects (rare)
    """
    out = dtw_cost_and_path(x1, x2, alpha=float(alpha), backtrack=bool(backtrack))

    if isinstance(out, (tuple, list)):
        if len(out) == 3:
            return float(out[0]), float(out[1]), out[2]
        if len(out) == 2:
            total = float(out[0])
            path = out[1]
            # conservative: estimate per-step using path length if available
            n_steps = int(len(path)) if path is not None else max(int(len(x1)), 1)
            return total, float(total) / float(max(n_steps, 1)), path
        raise RuntimeError(f"dtw_cost_and_path returned unexpected tuple size: {len(out)}")

    if isinstance(out, dict):
        total = float(out.get("cost", out.get("total_cost", np.nan)))
        cstep = out.get("cost_per_step", None)
        path = out.get("path", None)
        if cstep is None:
            n_steps = int(len(path)) if path is not None else max(int(len(x1)), 1)
            cstep = float(total) / float(max(n_steps, 1))
        return float(total), float(cstep), path

    raise RuntimeError(f"Unsupported dtw_cost_and_path return type: {type(out)}")


def build_sparse_dtw_edges(
    reps: List[Dict[str, str]],
    *,
    cfg: SparseDTWConfig,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Optional[List[Dict[str, Any]]]]:
    cache = WellCurveCache(cfg)

    candidates = build_candidate_edges(
        reps,
        max_km=float(cfg.max_km),
        knn_bridges=int(cfg.knn_bridges),
        bridge_max_km=float(cfg.bridge_max_km),
    )

    rep_by_id: Dict[int, Dict[str, str]] = {int(r["rep_id"]): r for r in reps}

    edges_out: List[Dict[str, Any]] = []
    paths_out: Optional[List[Dict[str, Any]]] = [] if cfg.emit_paths else None

    diag: Dict[str, Any] = {
        "n_reps": int(len(reps)),
        "n_candidates": int(len(candidates)),
        "n_edges_ok": 0,
        "n_edges_failed": 0,
        "failed": [],
        "config": {
            "n_samples": int(cfg.n_samples),
            "alpha": float(cfg.alpha),
            "max_km": float(cfg.max_km),
            "knn_bridges": int(cfg.knn_bridges),
            "bridge_max_km": float(cfg.bridge_max_km),
            "emit_paths": bool(cfg.emit_paths),
            "n_tiepoints": int(cfg.n_tiepoints),
        },
    }

    for a, b, dkm in candidates:
        ra = rep_by_id.get(a)
        rb = rep_by_id.get(b)
        if ra is None or rb is None:
            continue

        # Separate stages so failures are self-localizing
        try:
            wa = cache.get(ra)
            wb = cache.get(rb)
        except Exception as e:
            diag["n_edges_failed"] += 1
            diag["failed"].append({"src_rep_id": a, "dst_rep_id": b, "stage": "load_curve", "error": str(e)})
            continue

        try:
            cost_total, cost_step, path = _safe_dtw_cost_and_path(
                wa["x"],
                wb["x"],
                alpha=float(cfg.alpha),
                backtrack=bool(cfg.emit_paths),
            )
        except Exception as e:
            diag["n_edges_failed"] += 1
            diag["failed"].append({"src_rep_id": a, "dst_rep_id": b, "stage": "dtw", "error": str(e)})
            continue

        edges_out.append(
            {
                "src_rep_id": a,
                "dst_rep_id": b,
                "dist_km": f"{dkm:.3f}",
                "curve_src": wa["curve_mnemonic"],
                "curve_dst": wb["curve_mnemonic"],
                "n_samples": int(cfg.n_samples),
                "alpha": f"{float(cfg.alpha):.5f}",
                "dtw_cost": f"{float(cost_total):.6g}",
                "dtw_cost_per_step": f"{float(cost_step):.6g}",
                "src_z_top": f"{wa['z_top']:.3f}",
                "src_z_base": f"{wa['z_base']:.3f}",
                "dst_z_top": f"{wb['z_top']:.3f}",
                "dst_z_base": f"{wb['z_base']:.3f}",
                "src_h3_cell": ra.get("h3_cell", ""),
                "dst_h3_cell": rb.get("h3_cell", ""),
            }
        )

        if cfg.emit_paths and paths_out is not None:
            try:
                tp = downsample_path(path, n_tiepoints=int(cfg.n_tiepoints)).tolist() if path is not None else []
            except Exception:
                tp = []
            paths_out.append({"src_rep_id": a, "dst_rep_id": b, "tiepoints_ij": tp})

        diag["n_edges_ok"] += 1

    return edges_out, diag, paths_out


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build sparse DTW correlations over representative wells (ChronoLog-style).")
    ap.add_argument("--reps-csv", type=str, required=True, help="Path to representatives.csv")
    ap.add_argument("--out-dir", type=str, required=True, help="Output directory")
    ap.add_argument("--n-samples", type=int, default=256, help="Resample length (default 256)")
    ap.add_argument("--alpha", type=float, default=0.15, help="DTW exponent alpha (default 0.15)")
    ap.add_argument("--max-km", type=float, default=20.0, help="Local edge distance threshold (default 20 km)")
    ap.add_argument("--knn-bridges", type=int, default=1, help="kNN bridges per node to connect clusters (default 1)")
    ap.add_argument("--bridge-max-km", type=float, default=60.0, help="Max distance for bridge edges (default 60 km)")
    ap.add_argument("--emit-paths", action="store_true", help="Also write downsampled DTW paths (tiepoints) as JSONL")
    ap.add_argument("--n-tiepoints", type=int, default=64, help="Tiepoints per edge when --emit-paths (default 64)")
    args = ap.parse_args(argv)

    reps_csv = Path(args.reps_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reps = read_csv_rows(reps_csv)

    cfg = SparseDTWConfig(
        curve_canonical="GR",
        n_samples=int(args.n_samples),
        alpha=float(args.alpha),
        max_km=float(args.max_km),
        knn_bridges=int(args.knn_bridges),
        bridge_max_km=float(args.bridge_max_km),
        emit_paths=bool(args.emit_paths),
        n_tiepoints=int(args.n_tiepoints),
    )

    edges, diag, paths = build_sparse_dtw_edges(reps, cfg=cfg)

    write_csv(
        out_dir / "dtw_edges.csv",
        [
            "src_rep_id", "dst_rep_id", "dist_km",
            "curve_src", "curve_dst",
            "n_samples", "alpha",
            "dtw_cost", "dtw_cost_per_step",
            "src_z_top", "src_z_base",
            "dst_z_top", "dst_z_base",
            "src_h3_cell", "dst_h3_cell",
        ],
        edges,
    )

    (out_dir / "dtw_diagnostics.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")

    if cfg.emit_paths and paths is not None:
        p = out_dir / "dtw_paths_tiepoints.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for row in paths:
                f.write(json.dumps(row) + "\n")

    print(f"Representatives: {diag['n_reps']}")
    print(f"Candidate edges: {diag['n_candidates']}")
    print(f"DTW OK: {diag['n_edges_ok']}")
    print(f"DTW failed: {diag['n_edges_failed']}")
    print(f"Wrote: {out_dir / 'dtw_edges.csv'}")
    print(f"Wrote: {out_dir / 'dtw_diagnostics.json'}")
    if cfg.emit_paths:
        print(f"Wrote: {out_dir / 'dtw_paths_tiepoints.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
