# src/strataframe/steps/step2_build_gr_vectors_cache.py
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from strataframe.graph.las_utils import read_las_curve_resampled_ascii, read_las_header_only


@dataclass(frozen=True)
class Step2CacheConfig:
    curve_mnemonic: str = "GR"
    n_samples: int = 1024
    p_lo: float = 1.0
    p_hi: float = 99.0
    min_finite: int = 10
    max_las_mb: float = 256.0
    max_curves: int = 0
    max_rows: int = 0
    progress_every: int = 500
    gc_every: int = 200


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def run_step2_build_gr_vectors_cache(
    *,
    nodes_csv: Path,
    out_dir: Path,
    cfg: Step2CacheConfig,
    overwrite: bool = False,
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / "gr_vectors.npz"
    out_qc = out_dir / "gr_vectors_qc.csv"
    out_diag = out_dir / "diagnostics.json"
    out_manifest = out_dir / "manifest.json"

    if not overwrite and (out_npz.exists() or out_qc.exists()):
        raise FileExistsError(f"Cache outputs already exist under {out_dir}. Use overwrite=true.")

    nodes = pd.read_csv(nodes_csv)
    if "node_id" not in nodes.columns or "las_path" not in nodes.columns:
        raise ValueError("graph_nodes.csv must include node_id and las_path")

    nodes["node_id"] = pd.to_numeric(nodes["node_id"], errors="coerce").astype("Int64")
    nodes = nodes[nodes["node_id"].notna()].copy()

    keep_ids: List[int] = []
    z_tops: List[float] = []
    z_bases: List[float] = []
    x_list: List[np.ndarray] = []

    qc_rows: List[Dict[str, Any]] = []
    counts = {
        "n_nodes": int(len(nodes)),
        "n_ok": 0,
        "n_read_fail": 0,
        "n_skip_size": 0,
        "n_skip_curves": 0,
        "n_resample_fail": 0,
    }

    for i, row in nodes.iterrows():
        node_id = int(row["node_id"])
        las_path = Path(str(row["las_path"]))
        if not las_path.exists():
            counts["n_read_fail"] += 1
            qc_rows.append({"node_id": node_id, "status": "missing_file", "las_path": str(las_path)})
            continue

        if cfg.max_las_mb and cfg.max_las_mb > 0:
            try:
                size_mb = float(las_path.stat().st_size) / (1024.0 * 1024.0)
                if size_mb > float(cfg.max_las_mb):
                    counts["n_skip_size"] += 1
                    qc_rows.append({"node_id": node_id, "status": "skip_size", "las_path": str(las_path)})
                    continue
            except Exception:
                pass

        if cfg.max_curves and cfg.max_curves > 0:
            try:
                hdr = read_las_header_only(las_path)
                n_curves = int(len(hdr.get("curves", []) or []))
                if n_curves > int(cfg.max_curves):
                    counts["n_skip_curves"] += 1
                    qc_rows.append({"node_id": node_id, "status": "skip_curves", "las_path": str(las_path)})
                    continue
            except Exception:
                # if header read fails, try reading anyway
                pass

        try:
            x_norm, z_top, z_base, n_finite_raw, _ = read_las_curve_resampled_ascii(
                las_path,
                n_samples=int(cfg.n_samples),
                curve_candidates=(str(cfg.curve_mnemonic),),
                p_lo=float(cfg.p_lo),
                p_hi=float(cfg.p_hi),
                min_finite=int(cfg.min_finite),
                max_rows=int(cfg.max_rows),
            )
        except Exception:
            counts["n_resample_fail"] += 1
            qc_rows.append({"node_id": node_id, "status": "resample_fail", "las_path": str(las_path)})
            continue

        keep_ids.append(int(node_id))
        z_tops.append(float(z_top))
        z_bases.append(float(z_base))
        x_list.append(x_norm.astype("float32", copy=False))
        counts["n_ok"] += 1

        qc_rows.append(
            {
                "node_id": node_id,
                "status": "ok",
                "las_path": str(las_path),
                "z_top": float(z_top),
                "z_base": float(z_base),
                "n_finite_raw": int(n_finite_raw),
            }
        )

        if cfg.progress_every and (len(keep_ids) % int(cfg.progress_every) == 0):
            print(f"[step2-cache] {len(keep_ids)} ok / {int(len(nodes))} total")
        if cfg.gc_every and (len(keep_ids) % int(cfg.gc_every) == 0):
            import gc
            gc.collect()

    if not keep_ids:
        raise RuntimeError("No curves cached; all reads failed.")

    x_mat = np.stack(x_list, axis=0).astype("float32", copy=False)
    np.savez_compressed(
        out_npz,
        node_id=np.asarray(keep_ids, dtype="int64"),
        z_top=np.asarray(z_tops, dtype="float32"),
        z_base=np.asarray(z_bases, dtype="float32"),
        x_norm=x_mat,
        meta_json=json.dumps(asdict(cfg)),
    )

    pd.DataFrame(qc_rows).to_csv(out_qc, index=False)
    out_diag.write_text(json.dumps({"counts": counts, "config": asdict(cfg)}, indent=2), encoding="utf-8")
    out_manifest.write_text(
        json.dumps(
            {
                "step": "step2_build_gr_vectors_cache",
                "inputs": {"nodes_csv": str(nodes_csv)},
                "outputs": {"gr_vectors_npz": str(out_npz), "gr_vectors_qc_csv": str(out_qc)},
                "config": asdict(cfg),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"counts": counts, "out_npz": str(out_npz)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build gr_vectors cache for step2 DTW.")
    ap.add_argument("--nodes-csv", required=True, help="graph_nodes.csv")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--overwrite", action="store_true")

    ap.add_argument("--curve-mnemonic", default="GR")
    ap.add_argument("--n-samples", type=int, default=1024)
    ap.add_argument("--p-lo", type=float, default=1.0)
    ap.add_argument("--p-hi", type=float, default=99.0)
    ap.add_argument("--min-finite", type=int, default=10)
    ap.add_argument("--max-las-mb", type=float, default=256.0)
    ap.add_argument("--max-curves", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--progress-every", type=int, default=500)
    ap.add_argument("--gc-every", type=int, default=200)

    args = ap.parse_args()
    cfg = Step2CacheConfig(
        curve_mnemonic=str(args.curve_mnemonic),
        n_samples=int(args.n_samples),
        p_lo=float(args.p_lo),
        p_hi=float(args.p_hi),
        min_finite=int(args.min_finite),
        max_las_mb=float(args.max_las_mb),
        max_curves=int(args.max_curves),
        max_rows=int(args.max_rows),
        progress_every=int(args.progress_every),
        gc_every=int(args.gc_every),
    )

    run_step2_build_gr_vectors_cache(
        nodes_csv=Path(args.nodes_csv),
        out_dir=Path(args.out_dir),
        cfg=cfg,
        overwrite=bool(args.overwrite),
    )


if __name__ == "__main__":
    main()
