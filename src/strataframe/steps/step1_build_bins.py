# src/strataframe/steps/step1_build_bins.py
from __future__ import annotations

import json
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

from strataframe.io.csv import read_csv_rows, write_csv
from strataframe.graph.bin_wells_grid import GridBinningConfig, bin_wells_grid
from strataframe.graph.ks_manifest import WellRecord, extract_well_locations, read_ks_manifest


@dataclass(frozen=True)
class Step1Config:
    """
    Step 1: Assign wells to spatial bins using a rectangular grid (ONLY).

    Outputs:
      - bins.csv          (well_id, bin_id)
      - bins_meta.csv     (bin_id, n_wells, centroid_lat, centroid_lon, radius_km)
      - well_to_cell.csv  (one row per manifest LAS row, with cell assignment info)
      - manifest.json
    """
    target_bins: int = 100
    min_bin_size: int = 10

    # Grid params
    grid_cell_km: Optional[float] = None  # if None, auto-chosen to approach target_bins
    grid_pad_frac: float = 0.01

    # Optional: filter to a subset of wells (e.g., Step0 keep list)
    # CSV containing at least one of: well_id, api_num_nodash, api, kgs_id, url
    filter_wells_csv: Optional[str] = None


@dataclass(frozen=True)
class Step1Paths:
    out_dir: Path

    @property
    def bins_csv(self) -> Path:
        return self.out_dir / "bins.csv"

    @property
    def bins_meta_csv(self) -> Path:
        return self.out_dir / "bins_meta.csv"

    @property
    def well_to_cell_csv(self) -> Path:
        return self.out_dir / "well_to_cell.csv"

    @property
    def manifest_json(self) -> Path:
        return self.out_dir / "manifest.json"


# -----------------------------------------------------------------------------
# Filtering helpers
# -----------------------------------------------------------------------------

def _load_filter_ids(path: Path) -> set[str]:
    rows = read_csv_rows(path)
    keep: set[str] = set()

    keys = ["well_id", "api_num_nodash", "api", "kgs_id", "url"]
    for r in rows:
        for k in keys:
            v = (r.get(k, "") or "").strip()
            if not v:
                continue
            if k == "url":
                v = v.split("/")[-1]
            keep.add(v)
    return keep


def _well_id_candidates(w: WellRecord) -> List[str]:
    u = (w.url or "").strip()
    return [
        (w.api_num_nodash or "").strip(),
        (w.api or "").strip(),
        (w.kgs_id or "").strip(),
        u,
        u.split("/")[-1] if u else "",
    ]


def _select_well_id(w: WellRecord) -> str:
    """
    Preference order:
      1) api_num_nodash
      2) api
      3) kgs_id
      4) URL basename
    """
    for v in (w.api_num_nodash, w.api, w.kgs_id):
        s = (v or "").strip()
        if s:
            return s
    u = (w.url or "").strip()
    if u:
        return u.split("/")[-1]
    return f"well_{abs(hash((w.lat, w.lon))) % 10_000_000}"


# -----------------------------------------------------------------------------
# well_to_cell construction (grid-only)
# -----------------------------------------------------------------------------

def _parse_assignment_value(a: Any) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    """
    Accept common assignment encodings:
      - tuple/list: (row, col, cell_id, ...)
      - dict: {"row":..., "col":..., "cell_id":...} (or ix/iy variants)

    Returns (row, col, cell_id) or (None,None,None).
    """
    if a is None:
        return None, None, None

    if isinstance(a, (tuple, list)) and len(a) >= 3:
        try:
            r = int(a[0])
            c = int(a[1])
            cid = str(a[2])
            return r, c, cid
        except Exception:
            return None, None, None

    if isinstance(a, dict):
        rr = a.get("row", a.get("iy", a.get("r", None)))
        cc = a.get("col", a.get("ix", a.get("c", None)))
        cid = a.get("cell_id", a.get("cell", a.get("id", None)))
        try:
            r = int(rr) if rr is not None and str(rr).strip() != "" else None
            c = int(cc) if cc is not None and str(cc).strip() != "" else None
            s = str(cid) if cid is not None and str(cid).strip() != "" else None
            return r, c, s
        except Exception:
            return None, None, None

    return None, None, None


def _build_well_to_cell_rows(
    wells: List[WellRecord],
    *,
    wid_to_bin: Dict[str, str],
    assignment: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Create a per-manifest-row table (grid-only).

    Columns (stable):
      url,kgs_id,api,api_num_nodash,operator,lease,lat,lon,well_id,
      cell_id,cell_row,cell_col,bin_id
    """
    out: List[Dict[str, Any]] = []

    for w in wells:
        wid = _select_well_id(w)

        row: Dict[str, Any] = {
            "url": (w.url or "").strip(),
            "kgs_id": (w.kgs_id or "").strip(),
            "api": (w.api or "").strip(),
            "api_num_nodash": (w.api_num_nodash or "").strip(),
            "operator": (w.operator or "").strip(),
            "lease": (w.lease or "").strip(),
            "lat": f"{float(w.lat):.8f}",
            "lon": f"{float(w.lon):.8f}",
            "well_id": str(wid),
            "cell_id": "",
            "cell_row": "",
            "cell_col": "",
            "bin_id": wid_to_bin.get(str(wid), ""),
        }

        if assignment is not None:
            r, c, cid = _parse_assignment_value(assignment.get(str(wid)))
            if cid is not None:
                row["cell_id"] = str(cid)
                row["cell_row"] = str(int(r)) if r is not None else ""
                row["cell_col"] = str(int(c)) if c is not None else ""

        out.append(row)

    return out


# -----------------------------------------------------------------------------
# Grid config construction (supports both naming conventions)
# -----------------------------------------------------------------------------

def _make_grid_cfg(cfg: Step1Config) -> GridBinningConfig:
    """
    Construct GridBinningConfig robustly, supporting either:
      - GridBinningConfig(cell_km=..., pad_frac=...)
      - GridBinningConfig(grid_cell_km=..., grid_pad_frac=...)
    without breaking if the dataclass field names differ.
    """
    kwargs: Dict[str, Any] = {
        "target_bins": int(cfg.target_bins),
        "min_bin_size": int(cfg.min_bin_size),
    }

    cell_km_val = float(cfg.grid_cell_km) if cfg.grid_cell_km is not None else None
    pad_val = float(cfg.grid_pad_frac)

    if is_dataclass(GridBinningConfig):
        fn = {f.name for f in fields(GridBinningConfig)}
        if "cell_km" in fn:
            kwargs["cell_km"] = cell_km_val
        elif "grid_cell_km" in fn:
            kwargs["grid_cell_km"] = cell_km_val

        if "pad_frac" in fn:
            kwargs["pad_frac"] = pad_val
        elif "grid_pad_frac" in fn:
            kwargs["grid_pad_frac"] = pad_val
    else:
        kwargs["cell_km"] = cell_km_val
        kwargs["pad_frac"] = pad_val

    return GridBinningConfig(**kwargs)  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# Main step (grid-only)
# -----------------------------------------------------------------------------

def run_step1_build_bins(
    *,
    ks_manifest_path: Path,
    out_dir: Path,
    cfg: Step1Config,
    overwrite: bool = False,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = Step1Paths(out_dir=out_dir)

    if not overwrite:
        if (
            paths.bins_csv.exists()
            or paths.bins_meta_csv.exists()
            or paths.well_to_cell_csv.exists()
            or paths.manifest_json.exists()
        ):
            raise FileExistsError(
                f"Step1 outputs already exist under {out_dir}. "
                f"Use overwrite=true or pick a new run dir."
            )

    rows = read_ks_manifest(ks_manifest_path)
    wells = extract_well_locations(rows)

    # Optional filter
    if cfg.filter_wells_csv:
        filter_ids = _load_filter_ids(Path(cfg.filter_wells_csv))

        def _match(w: WellRecord) -> bool:
            return any(c and c in filter_ids for c in _well_id_candidates(w))

        wells = [w for w in wells if _match(w)]

    gcfg = _make_grid_cfg(cfg)

    # bin_wells_grid is expected to return:
    #   (bins_rows, bins_meta, diag, assignment)  OR  (bins_rows, bins_meta, diag)
    res = bin_wells_grid(wells, cfg=gcfg)
    if isinstance(res, tuple) and len(res) == 4:
        bins_rows, bins_meta, diag, assignment = res
    elif isinstance(res, tuple) and len(res) == 3:
        bins_rows, bins_meta, diag = res
        assignment = None
    else:
        raise RuntimeError("bin_wells_grid returned an unexpected result shape.")

    write_csv(paths.bins_csv, ["well_id", "bin_id"], bins_rows)
    write_csv(paths.bins_meta_csv, ["bin_id", "n_wells", "centroid_lat", "centroid_lon", "radius_km"], bins_meta)

    # bins.csv join map (bin_id may reflect merged bins)
    wid_to_bin: Dict[str, str] = {}
    for r in bins_rows:
        wid_to_bin[str(r.get("well_id", "")).strip()] = str(r.get("bin_id", "")).strip()

    # well_to_cell.csv includes raw cell indices (if assignment provided) plus final bin_id
    w2c_rows = _build_well_to_cell_rows(
        wells,
        wid_to_bin=wid_to_bin,
        assignment=assignment if isinstance(assignment, dict) else None,
    )
    write_csv(
        paths.well_to_cell_csv,
        [
            "url",
            "kgs_id",
            "api",
            "api_num_nodash",
            "operator",
            "lease",
            "lat",
            "lon",
            "well_id",
            "cell_id",
            "cell_row",
            "cell_col",
            "bin_id",
        ],
        w2c_rows,
    )

    # Convenience for downstream: provide grid_km in a stable key if available.
    grid_km = None
    try:
        if isinstance(diag, dict):
            grid_km = diag.get("chosen_grid_cell_km", None)
    except Exception:
        grid_km = None

    diag_out = dict(diag) if isinstance(diag, dict) else {}
    if grid_km is not None:
        diag_out.setdefault("grid_km", grid_km)

    manifest = {
        "step": "step1_bin_wells",
        "inputs": {
            "ks_manifest": str(ks_manifest_path),
            "filter_wells_csv": str(cfg.filter_wells_csv) if cfg.filter_wells_csv else "",
        },
        "params": {
            "target_bins": int(cfg.target_bins),
            "min_bin_size": int(cfg.min_bin_size),
            "grid_cell_km": (float(cfg.grid_cell_km) if cfg.grid_cell_km is not None else None),
            "grid_pad_frac": float(cfg.grid_pad_frac),
        },
        "outputs": {
            "bins_csv": str(paths.bins_csv),
            "bins_meta_csv": str(paths.bins_meta_csv),
            "well_to_cell_csv": str(paths.well_to_cell_csv),
        },
        "diagnostics": diag_out,
    }
    paths.manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
