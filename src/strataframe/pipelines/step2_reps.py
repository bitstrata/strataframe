# src/strataframe/pipelines/step2_reps.py
from __future__ import annotations

import csv
import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from strataframe.contracts.step2_outputs import CELL_TYPEWELL_HEADER, REP_CSV_HEADER, REP_PLACEMENT_HEADER
from strataframe.graph.candidate_filters import prefilter_rows_for_las
from strataframe.graph.select_representatives import RepSelectConfig, select_representatives
from strataframe.io.csv import write_csv
from strataframe.io.stream_csv import CsvSink
from strataframe.io.wells_gr_index import load_las_basenames_from_wells_gr
from strataframe.spatial.make_well_to_cell import (
    grid_cell_id_from_ij,
    load_or_build_well_to_cell_rows,
    parse_grid_cell_id,
)
from strataframe.typewell.local_typewell import TypeWellConfig, build_cell_typewell, place_rep_against_template
from strataframe.utils.hash_utils import stable_hash32
from strataframe.utils.state_json import write_state_json


@dataclass(frozen=True)
class Step2RepsConfig:
    n_rep: int = 1000
    quota_mode: str = "equal"  # "equal" | "proportional"
    q_min: int = 5
    candidate_method: str = "farthest"  # "farthest" | "random"
    seed: int = 42

    max_candidates_per_cell: int = 25
    require_gr: bool = True

    # Grid binning
    grid_km: float = 10.0

    # Typewell / placement
    build_typewells: bool = True
    typewell: TypeWellConfig = TypeWellConfig()

    # Safety valves
    max_las_mb: int = 512

    # Typewell workload controls
    typewell_max_cells: int = 0
    typewell_max_kernel_wells: int = 200
    typewell_gc_every: int = 10
    resume: bool = True


def run_step2_reps(
    *,
    well_to_cell_csv: Path,
    las_root: Path,
    out_dir: Path,
    cfg: Step2RepsConfig,
    dry_run: bool = False,
    manifest_json: Optional[Path] = None,
    force_rebuild_well_to_cell: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load/build well_to_cell (dry_run-safe)
    rows, w2c_diag = load_or_build_well_to_cell_rows(
        well_to_cell_csv=Path(well_to_cell_csv),
        manifest_json=Path(manifest_json) if manifest_json else None,
        grid_km=float(cfg.grid_km),
        force_rebuild=bool(force_rebuild_well_to_cell),
        dry_run=bool(dry_run),
    )
    if not rows:
        raise FileNotFoundError(
            "well_to_cell rows are unavailable. "
            f"path={well_to_cell_csv} diagnostics={json.dumps(w2c_diag.get('diagnostics', {}), indent=2)}"
        )

    # 2) Prefilter candidates (Step0 whitelist + LAS existence/size)
    step0_guess = out_dir.parent / "00_wells_gr" / "wells_gr.parquet"
    allowed_names = load_las_basenames_from_wells_gr(step0_guess)
    rows, pre_diag = prefilter_rows_for_las(
        rows,
        las_root=Path(las_root),
        max_las_mb=int(cfg.max_las_mb),
        allowed_names=allowed_names,
    )
    w2c_diag = dict(w2c_diag)
    if allowed_names is not None:
        w2c_diag["step0_wells_gr_path"] = str(step0_guess)
    w2c_diag.update(pre_diag)

    if not rows:
        raise RuntimeError(
            "All candidates were filtered out before representative selection. "
            f"well_to_cell={json.dumps(w2c_diag, indent=2)}"
        )

    # 3) Select representatives
    rep_cfg = RepSelectConfig(
        reps_per_cell=2,
        max_candidates_per_cell=int(cfg.max_candidates_per_cell),
        require_gr=bool(cfg.require_gr),
    )

    reps, diag = select_representatives(
        rows,
        las_root=Path(las_root),
        cfg=rep_cfg,
        n_rep_target=int(cfg.n_rep),
        quota_mode=str(cfg.quota_mode),
        q_min=int(cfg.q_min),
        candidate_method=str(cfg.candidate_method),
        seed=int(cfg.seed),
        out_dir=None if dry_run else out_dir,
    )

    diag = dict(diag or {})
    diag["well_to_cell"] = w2c_diag

    if not dry_run:
        write_csv(out_dir / "representatives.csv", REP_CSV_HEADER, reps)
        (out_dir / "rep_selection_diagnostics.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")

    # 4) Typewells + placement (local typewell)
    if bool(cfg.build_typewells) and (not dry_run) and reps:
        templates_dir = out_dir / "cell_templates"
        templates_dir.mkdir(parents=True, exist_ok=True)

        by_cell: Dict[str, List[Dict[str, str]]] = {}
        for r in rows:
            c = (r.get("cell_id") or r.get("grid_cell") or "").strip()
            if c:
                by_cell.setdefault(c, []).append(r)

        rep_cells = sorted({str(r.get("cell_id", "")).strip() for r in reps if str(r.get("cell_id", "")).strip()})

        hard_cap = int(cfg.typewell_max_cells or 0)
        if hard_cap <= 0 and len(rep_cells) > 300:
            hard_cap = 300
        if hard_cap > 0 and len(rep_cells) > hard_cap:
            rng = np.random.default_rng(int(cfg.seed))
            rep_cells = list(rng.choice(rep_cells, size=hard_cap, replace=False))
            rep_cells.sort()

        cell_typewells_csv = out_dir / "cell_typewells.csv"
        rep_placement_csv = out_dir / "rep_placement.csv"

        existing_cells: set[str] = set()
        if bool(cfg.resume) and cell_typewells_csv.exists():
            try:
                with cell_typewells_csv.open("r", encoding="utf-8", newline="") as f:
                    for rr in csv.DictReader(f):
                        cid = (rr.get("cell_id") or "").strip()
                        if cid:
                            existing_cells.add(cid)
            except Exception:
                existing_cells = set()

        sink_tw = CsvSink(cell_typewells_csv, CELL_TYPEWELL_HEADER, append=bool(cfg.resume) and cell_typewells_csv.exists())
        sink_pl = CsvSink(rep_placement_csv, REP_PLACEMENT_HEADER, append=bool(cfg.resume) and rep_placement_csv.exists())

        reps_by_cell: Dict[str, List[Dict[str, Any]]] = {}
        for r in reps:
            cid = str(r.get("cell_id", "")).strip()
            if cid:
                reps_by_cell.setdefault(cid, []).append(r)

        tw_cfg = cfg.typewell
        max_kernel = int(cfg.typewell_max_kernel_wells)

        for idx, cell_id in enumerate(rep_cells, start=1):
            if cell_id in existing_cells:
                continue

            ij = parse_grid_cell_id(cell_id)
            if ij is None:
                meta = {k: "" for k in CELL_TYPEWELL_HEADER}
                meta.update({"cell_id": cell_id, "n_kernel": "0", "error": f"bad_cell_id: {cell_id!r}"})
                sink_tw.write(meta)
                for rep in reps_by_cell.get(cell_id, []):
                    sink_pl.write(
                        {
                            "rep_id": rep.get("rep_id", ""),
                            "cell_id": cell_id,
                            "url": rep.get("url", ""),
                            "las_path": rep.get("las_path", ""),
                            "status": "bad_cell_id",
                            "error": f"bad_cell_id: {cell_id!r}",
                        }
                    )
                continue

            ix, iy = ij
            write_state_json(out_dir, {"phase": "typewell", "cell_id": cell_id, "i": idx, "n": len(rep_cells)})

            kernel_rows: List[Dict[str, str]] = []
            r0 = int(getattr(tw_cfg, "kernel_radius", 1))
            rmax = int(getattr(tw_cfg, "kernel_radius_max", max(2, r0)))
            min_k = int(getattr(tw_cfg, "min_kernel_wells", 1))

            r_use = r0
            while True:
                kernel_rows = []
                for dx in range(-r_use, r_use + 1):
                    for dy in range(-r_use, r_use + 1):
                        cid = grid_cell_id_from_ij(ix + dx, iy + dy)
                        kernel_rows.extend(by_cell.get(cid, []))
                if len(kernel_rows) >= min_k or r_use >= rmax:
                    break
                r_use += 1

            if len(kernel_rows) > max_kernel:
                cell_seed = (int(cfg.seed) + stable_hash32(cell_id)) & 0xFFFFFFFF
                rng = np.random.default_rng(cell_seed)
                sel = rng.choice(len(kernel_rows), size=max_kernel, replace=False)
                kernel_rows = [kernel_rows[int(i)] for i in sel]

            try:
                meta = build_cell_typewell(
                    cell_id=cell_id,
                    cell_ix=int(ix),
                    cell_iy=int(iy),
                    kernel_rows=kernel_rows,
                    las_root=Path(las_root),
                    cfg=tw_cfg,
                    templates_dir=templates_dir,
                    seed=int(cfg.seed),
                )
            except Exception as e:
                meta = {k: "" for k in CELL_TYPEWELL_HEADER}
                meta.update(
                    {
                        "cell_id": cell_id,
                        "grid_ix": str(int(ix)),
                        "grid_iy": str(int(iy)),
                        "n_kernel": str(int(len(kernel_rows))),
                        "error": f"{type(e).__name__}: {e}",
                    }
                )

            meta.setdefault("cell_id", cell_id)
            meta.setdefault("grid_ix", str(int(ix)))
            meta.setdefault("grid_iy", str(int(iy)))
            meta.setdefault("n_kernel", str(int(len(kernel_rows))))
            meta.setdefault("template_path", "")
            meta.setdefault("error", "")
            sink_tw.write(meta)

            tpl = str(meta.get("template_path", "") or "").strip()
            reps_here = reps_by_cell.get(cell_id, [])

            for rep in reps_here:
                if not tpl or not Path(tpl).is_file():
                    sink_pl.write(
                        {
                            "rep_id": rep.get("rep_id", ""),
                            "cell_id": cell_id,
                            "url": rep.get("url", ""),
                            "las_path": rep.get("las_path", ""),
                            "status": "no_template",
                            "error": (meta.get("error") or "no_template"),
                        }
                    )
                    continue

                try:
                    p = place_rep_against_template(
                        rep_row=rep,
                        cell_meta=meta,
                        las_root=Path(las_root),
                        cfg=tw_cfg,
                    )
                except Exception as e:
                    sink_pl.write(
                        {
                            "rep_id": rep.get("rep_id", ""),
                            "cell_id": cell_id,
                            "url": rep.get("url", ""),
                            "las_path": rep.get("las_path", ""),
                            "status": "place_fail",
                            "error": f"{type(e).__name__}: {e}",
                        }
                    )
                    continue

                p.setdefault("error", "")
                p.setdefault("match_span", "")
                p.setdefault("warp_ratio", "")
                sink_pl.write(p)

            if int(cfg.typewell_gc_every) > 0 and (idx % int(cfg.typewell_gc_every) == 0):
                gc.collect()

        sink_tw.close()
        sink_pl.close()
        write_state_json(out_dir, {"phase": "done", "n_cells": len(rep_cells)})

    return reps, diag
