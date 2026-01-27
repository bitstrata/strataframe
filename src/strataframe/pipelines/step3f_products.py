# src/strataframe/pipelines/step3f_products.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

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

from strataframe.rgt.chronostrat import ChronostratConfig, build_chronostrat_diagram
from strataframe.rgt.shifts_npz import load_shifts_npz
from strataframe.rgt.wavelet_tops import CwtConfig, derive_tops_multiscale
from strataframe.rgt.top_depth import TopsExportConfig, tops_to_depth_table, tops_depth_fieldnames
from strataframe.pipelines.step3_graph_attach_arrays import attach_rep_arrays
from strataframe.pipelines.step3b_rep_arrays import load_rep_arrays_npz
from strataframe.pipelines.step3_io_framework import load_framework_graph


def _require() -> None:
    if pd is None:
        raise RuntimeError("pandas is required. Install with: pip install pandas")
    if nx is None:
        raise RuntimeError("networkx is required. Install with: pip install networkx")


@dataclass(frozen=True)
class Step3ProductsConfig:
    # Chronostrat diagram
    chronostrat: ChronostratConfig = ChronostratConfig()

    # CWT tops
    cwt: CwtConfig = CwtConfig()

    # Which hierarchy levels to export to depth tables (indices into tops_by_level)
    export_levels: Sequence[int] = (0, 2, 4)

    # Column naming for tops export
    tops_prefix: str = "TOP"


def _save_chronostrat_npz(out_path: Path, diag: Dict[str, object]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    node_order = np.asarray(diag["node_order"], dtype=np.str_)  # type: ignore[index]
    rgt_grid = np.asarray(diag["rgt_grid"], dtype="float64")    # type: ignore[index]
    diag_arr = np.asarray(diag["diag"], dtype="float64")        # type: ignore[index]
    type_log = np.asarray(diag["type_log"], dtype="float64")    # type: ignore[index]
    depth_grid = np.asarray(diag["depth_grid"], dtype="float64")  # type: ignore[index]
    reversal_frac = np.asarray(diag["reversal_frac"], dtype="float64")  # type: ignore[index]

    np.savez_compressed(
        out_path,
        node_order=node_order,
        rgt_grid=rgt_grid,
        diag=diag_arr,
        type_log=type_log,
        depth_grid=depth_grid,
        reversal_frac=reversal_frac,
    )


def run_step3_products(
    *,
    out_dir: str | Path,
    framework_nodes_csv: str | Path,
    framework_edges_csv: str | Path,
    rep_arrays_npz: str | Path,
    rgt_shifts_npz: str | Path,
    cfg: Step3ProductsConfig = Step3ProductsConfig(),
) -> Dict[str, Any]:
    """
    Produces:
      - chronostrat_diag.npz
      - type_log.csv
      - cwt_tops.npz (tops + wavelet response)
      - tops_level{L}.csv for requested export levels
      - products_manifest.json
    """
    _require()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Rebuild framework graph
    H = load_framework_graph(nodes_csv=framework_nodes_csv, edges_csv=framework_edges_csv)

    # Attach rep arrays
    rep_arrays = load_rep_arrays_npz(rep_arrays_npz)
    attach_rep_arrays(H, rep_arrays, log_key="log_rs", depth_key="depth_rs", node_log_key="log_rs", node_depth_key="depth_rs")

    # Load shifts
    shifts = load_shifts_npz(rgt_shifts_npz)

    # Chronostrat + type log
    diag = build_chronostrat_diagram(H, shifts, cfg=cfg.chronostrat, z_key="depth_rs", log_key="log_rs")

    chron_npz = out_dir / "chronostrat_diag.npz"
    _save_chronostrat_npz(chron_npz, diag)

    # Save type log as CSV
    rgt_grid = np.asarray(diag["rgt_grid"], dtype="float64")
    type_log = np.asarray(diag["type_log"], dtype="float64")
    type_csv = out_dir / "type_log.csv"
    pd.DataFrame({"rgt": rgt_grid, "type_log": type_log}).to_csv(type_csv, index=False)

    # Multiscale tops
    tops = derive_tops_multiscale(type_log, cfg=cfg.cwt)

    cwt_npz = out_dir / "cwt_tops.npz"
    np.savez_compressed(
        cwt_npz,
        widths=np.asarray(tops["widths"], dtype=np.int32),  # type: ignore[index]
        W=np.asarray(tops["W"], dtype="float64"),           # type: ignore[index]
        # ragged lists stored as object array (small); keep bounded in later consumers
        tops_by_level=np.asarray(tops["tops_by_level"], dtype=object),  # type: ignore[index]
    )

    # Export per-level depth tables
    node_order = diag["node_order"]          # type: ignore[index]
    depth_grid = np.asarray(diag["depth_grid"], dtype="float64")
    tops_by_level = tops["tops_by_level"]    # type: ignore[index]

    exported: List[str] = []
    for L in cfg.export_levels:
        te_cfg = TopsExportConfig(level=int(L), prefix=str(cfg.tops_prefix))
        rows = tops_to_depth_table(
            node_order=node_order, depth_grid=depth_grid, rgt_grid=rgt_grid, tops_by_level=tops_by_level, cfg=te_cfg
        )
        cols = tops_depth_fieldnames(rgt_grid=rgt_grid, tops_by_level=tops_by_level, cfg=te_cfg)
        out_csv = out_dir / f"tops_level{int(L)}.csv"
        pd.DataFrame(rows, columns=cols).to_csv(out_csv, index=False)
        exported.append(str(out_csv))

    manifest = {
        "chronostrat_npz": str(chron_npz),
        "type_log_csv": str(type_csv),
        "cwt_tops_npz": str(cwt_npz),
        "exported_tops_csv": exported,
        "diag_meta": diag.get("diag_meta", {}),
        "cwt_cfg": {
            "widths": list(getattr(cfg.cwt, "widths", [])),
            "snap_window": int(getattr(cfg.cwt, "snap_window", 0)),
            "include_endpoints": bool(getattr(cfg.cwt, "include_endpoints", True)),
        },
        "chronostrat_cfg": {
            "n_rgt": int(getattr(cfg.chronostrat, "n_rgt", 0)),
            "rgt_pad_frac": float(getattr(cfg.chronostrat, "rgt_pad_frac", 0.0)),
            "monotonic_mode": str(getattr(getattr(cfg.chronostrat, "monotonic", None), "mode", "")),
        },
    }
    (out_dir / "products_manifest.json").write_text(json.dumps(manifest, indent=2))

    return manifest
