# src/strataframe/graph/bin_wells_grid.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from strataframe.spatial.geodesy import haversine_km
from strataframe.graph.ks_manifest import WellRecord

try:
    from pyproj import Transformer  # type: ignore
except Exception:  # pragma: no cover
    Transformer = None  # type: ignore


# -----------------------------------------------------------------------------
# Projection helpers (lon/lat -> projected meters)
# -----------------------------------------------------------------------------

DEFAULT_GRID_EPSG = 5070  # NAD83 / Conus Albers (good single-CRS choice for Kansas)


def require_pyproj() -> None:
    if Transformer is None:
        raise RuntimeError("pyproj is required for grid binning. Install with: pip install pyproj")


def _make_transformers(epsg: int) -> Tuple[Any, Any]:
    """
    Always enforce (lon,lat) axis order.
    Returns: (fwd: 4326->epsg, inv: epsg->4326)
    """
    require_pyproj()
    fwd = Transformer.from_crs("EPSG:4326", f"EPSG:{int(epsg)}", always_xy=True)
    inv = Transformer.from_crs(f"EPSG:{int(epsg)}", "EPSG:4326", always_xy=True)
    return fwd, inv


def lonlat_to_xy_m(lon: float, lat: float, *, epsg: int) -> Tuple[float, float]:
    fwd, _ = _make_transformers(epsg)
    x, y = fwd.transform(float(lon), float(lat))
    return float(x), float(y)


def xy_m_to_lonlat(x: float, y: float, *, epsg: int) -> Tuple[float, float]:
    _, inv = _make_transformers(epsg)
    lon, lat = inv.transform(float(x), float(y))
    return float(lon), float(lat)


# -----------------------------------------------------------------------------
# Identity helpers
# -----------------------------------------------------------------------------

def _well_id(w: WellRecord) -> str:
    # Stable join preference order for downstream artifacts.
    for v in (w.api_num_nodash, w.api, w.kgs_id):
        s = (v or "").strip()
        if s:
            return s
    u = (w.url or "").strip()
    if u:
        return u.split("/")[-1]
    return f"well_{abs(hash((w.lat, w.lon))) % 10_000_000}"


def _centroid_latlon(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    if not points:
        return (float("nan"), float("nan"))
    lat = sum(p[0] for p in points) / float(len(points))
    lon = sum(p[1] for p in points) / float(len(points))
    return (float(lat), float(lon))


def _radius_km(points: List[Tuple[float, float]], cen: Tuple[float, float]) -> float:
    if not points:
        return 0.0
    clat, clon = cen
    r = 0.0
    for lat, lon in points:
        d = haversine_km(clat, clon, lat, lon)
        if d > r:
            r = float(d)
    return float(r)


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class GridBinningConfig:
    target_bins: int = 100
    min_bin_size: int = 10

    # IMPORTANT: keep these names to match Step1 caller (cell_km, pad_frac)
    cell_km: Optional[float] = None  # if None -> auto-chosen
    pad_frac: float = 0.01           # pad extent by this fraction before indexing

    # Projection CRS for binning (projected meters)
    crs_epsg: int = DEFAULT_GRID_EPSG

    # Auto-sizing search
    scale_step: float = 1.20
    n_scales: int = 24


# -----------------------------------------------------------------------------
# Grid indexing (in projected meters)
# -----------------------------------------------------------------------------

def _bounds_with_padding(
    xs: List[float],
    ys: List[float],
    *,
    cell_m: float,
    pad_frac: float,
) -> Tuple[float, float, float, float, float, float]:
    """
    Returns: x0, y0, x_min_p, x_max_p, y_min_p, y_max_p
    where x0/y0 are snapped down to multiples of cell_m.
    """
    if not xs or not ys:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    x_min = float(min(xs))
    x_max = float(max(xs))
    y_min = float(min(ys))
    y_max = float(max(ys))

    xr = max(1e-9, x_max - x_min)
    yr = max(1e-9, y_max - y_min)
    pad = float(max(xr, yr) * float(pad_frac))

    x_min_p = x_min - pad
    x_max_p = x_max + pad
    y_min_p = y_min - pad
    y_max_p = y_max + pad

    x0 = math.floor(x_min_p / float(cell_m)) * float(cell_m)
    y0 = math.floor(y_min_p / float(cell_m)) * float(cell_m)

    return (float(x0), float(y0), float(x_min_p), float(x_max_p), float(y_min_p), float(y_max_p))


def grid_index(
    x_m: float,
    y_m: float,
    *,
    x0_m: float,
    y0_m: float,
    cell_m: float,
) -> Tuple[int, int]:
    ix = int(math.floor((float(x_m) - float(x0_m)) / float(cell_m)))
    iy = int(math.floor((float(y_m) - float(y0_m)) / float(cell_m)))
    return ix, iy


def grid_cell_id(ix: int, iy: int) -> str:
    # Parseable, stable identifier for grid cells.
    return f"g{int(ix)}_{int(iy)}"


def _index_extents(
    *,
    x0_m: float,
    y0_m: float,
    cell_m: float,
    x_min_p: float,
    x_max_p: float,
    y_min_p: float,
    y_max_p: float,
) -> Tuple[int, int, int, int]:
    """
    Returns inclusive index extents that cover the padded bounds.
    """
    ix_min = int(math.floor((x_min_p - x0_m) / cell_m))
    ix_max = int(math.ceil((x_max_p - x0_m) / cell_m) - 1)
    iy_min = int(math.floor((y_min_p - y0_m) / cell_m))
    iy_max = int(math.ceil((y_max_p - y0_m) / cell_m) - 1)
    return ix_min, ix_max, iy_min, iy_max


# -----------------------------------------------------------------------------
# Scale selection (try to achieve <= target_bins)
# -----------------------------------------------------------------------------

def _occupancy_for_cell_m(
    wells_xy: List[Tuple[str, float, float, float, float]],  # (wid, lat, lon, x_m, y_m)
    *,
    cell_m: float,
    pad_frac: float,
) -> Tuple[int, float, float]:
    xs = [x for _, _, _, x, _ in wells_xy]
    ys = [y for _, _, _, _, y in wells_xy]

    x0, y0, x_min_p, x_max_p, y_min_p, y_max_p = _bounds_with_padding(xs, ys, cell_m=float(cell_m), pad_frac=float(pad_frac))

    cells = set()
    for wid, lat, lon, x, y in wells_xy:
        ix, iy = grid_index(x, y, x0_m=float(x0), y0_m=float(y0), cell_m=float(cell_m))
        cells.add(grid_cell_id(ix, iy))

    return int(len(cells)), float(x0), float(y0)


def choose_cell_m_leq_target(
    wells_xy: List[Tuple[str, float, float, float, float]],
    *,
    target_bins: int,
    cell_km_hint: Optional[float],
    pad_frac: float,
    scale_step: float,
    n_scales: int,
) -> Tuple[float, Dict[str, int], Dict[str, Any]]:
    if not wells_xy:
        raise ValueError("No wells to bin (empty after filtering).")

    xs = [x for _, _, _, x, _ in wells_xy]
    ys = [y for _, _, _, _, y in wells_xy]
    x_min = float(min(xs)); x_max = float(max(xs))
    y_min = float(min(ys)); y_max = float(max(ys))
    area_m2 = max(1e-6, (x_max - x_min) * (y_max - y_min))

    # initial guess: sqrt(area / target)
    if cell_km_hint is not None and math.isfinite(float(cell_km_hint)) and float(cell_km_hint) > 0:
        base_km = float(cell_km_hint)
    else:
        base_m = float(math.sqrt(area_m2 / max(1, int(target_bins))))
        base_km = max(1e-6, base_m / 1000.0)

    step = max(1.05, float(scale_step))
    ks = list(range(-6, int(n_scales)))
    candidates_km = [base_km * (step ** k) for k in ks]
    candidates_km = sorted({float(f"{c:.6f}") for c in candidates_km if c > 0})

    occ_map: Dict[str, int] = {}
    best: Optional[Tuple[int, float]] = None  # (delta, chosen_km)
    best_occ = -1

    for ck_km in candidates_km:
        ck_m = float(ck_km) * 1000.0
        occ, _, _ = _occupancy_for_cell_m(wells_xy, cell_m=float(ck_m), pad_frac=float(pad_frac))
        occ_map[f"{ck_km:.6f}"] = int(occ)

        if occ <= int(target_bins) and occ > 0:
            delta = abs(int(target_bins) - int(occ))
            # tie-break: prefer coarser (larger cells) for stability
            if best is None or (delta < best[0]) or (delta == best[0] and ck_km > best[1]):
                best = (delta, float(ck_km))
                best_occ = int(occ)

    if best is not None:
        chosen_km = float(best[1])
        chosen_m = chosen_km * 1000.0
        diag = {
            "area_m2_est": float(area_m2),
            "base_cell_km": float(base_km),
            "chosen_occ": int(best_occ),
        }
        return float(chosen_m), occ_map, diag

    # fallback: coarsest candidate
    chosen_km = float(max(candidates_km))
    chosen_m = chosen_km * 1000.0
    occ, _, _ = _occupancy_for_cell_m(wells_xy, cell_m=float(chosen_m), pad_frac=float(pad_frac))
    diag = {
        "area_m2_est": float(area_m2),
        "base_cell_km": float(base_km),
        "chosen_occ": int(occ),
        "fallback": "coarsest",
    }
    return float(chosen_m), occ_map, diag


# -----------------------------------------------------------------------------
# Binning + merge tiny bins (centroid-nearest merge in lat/lon)
# -----------------------------------------------------------------------------

@dataclass
class _Bin:
    bin_id: str
    source_cells: set[str]
    wells: List[Tuple[str, float, float]]  # (well_id, lat, lon)

    def centroid(self) -> Tuple[float, float]:
        pts = [(lat, lon) for _, lat, lon in self.wells]
        return _centroid_latlon(pts)

    def n(self) -> int:
        return int(len(self.wells))


def _merge_small_bins(
    bins: Dict[str, _Bin],
    *,
    min_bin_size: int,
) -> Tuple[Dict[str, _Bin], List[Dict[str, Any]]]:
    merges: List[Dict[str, Any]] = []
    if int(min_bin_size) <= 1 or len(bins) <= 1:
        return bins, merges

    while True:
        small = [b for b in bins.values() if b.n() < int(min_bin_size)]
        if not small or len(bins) <= 1:
            break

        small.sort(key=lambda b: (b.n(), b.bin_id))
        donor = small[0]
        donor_c = donor.centroid()

        best_dst: Optional[_Bin] = None
        best_d = float("inf")
        for cand in bins.values():
            if cand.bin_id == donor.bin_id:
                continue
            cc = cand.centroid()
            d = haversine_km(donor_c[0], donor_c[1], cc[0], cc[1])
            if d < best_d:
                best_d = float(d)
                best_dst = cand

        if best_dst is None:
            break

        dst = best_dst
        before_dst_n = dst.n()
        before_donor_n = donor.n()

        dst.wells.extend(donor.wells)
        dst.source_cells |= donor.source_cells
        bins.pop(donor.bin_id, None)

        merges.append(
            {
                "src_bin_id": donor.bin_id,
                "dst_bin_id": dst.bin_id,
                "src_n_wells": int(before_donor_n),
                "dst_n_wells_before": int(before_dst_n),
                "dst_n_wells_after": int(dst.n()),
                "centroid_dist_km": float(best_d),
            }
        )

    return bins, merges


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def bin_wells_grid(
    wells: List[WellRecord],
    *,
    cfg: GridBinningConfig,
) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]], Dict[str, Any], Dict[str, Tuple[int, int, str]]]:
    """
    Returns:
      bins_rows:   [{well_id, bin_id}, ...]
      bins_meta:   [{bin_id, n_wells, centroid_lat, centroid_lon, radius_km}, ...]
      diagnostics: dict for manifest.json
      assignment:  {well_id: (cell_row, cell_col, cell_id)}  # for well_to_cell.csv
    """
    if not wells:
        raise ValueError("No wells with valid coordinates were found in the manifest.")

    epsg = int(cfg.crs_epsg) if int(cfg.crs_epsg) > 0 else DEFAULT_GRID_EPSG

    # project wells to meters
    wells_xy: List[Tuple[str, float, float, float, float]] = []
    for w in wells:
        wid = _well_id(w)
        x_m, y_m = lonlat_to_xy_m(float(w.lon), float(w.lat), epsg=epsg)
        wells_xy.append((wid, float(w.lat), float(w.lon), float(x_m), float(y_m)))

    chosen_cell_m, occ_map, sel_diag = choose_cell_m_leq_target(
        wells_xy,
        target_bins=int(cfg.target_bins),
        cell_km_hint=cfg.cell_km,
        pad_frac=float(cfg.pad_frac),
        scale_step=float(cfg.scale_step),
        n_scales=int(cfg.n_scales),
    )
    chosen_cell_km = float(chosen_cell_m) / 1000.0

    # final bounds + origin
    xs = [x for _, _, _, x, _ in wells_xy]
    ys = [y for _, _, _, _, y in wells_xy]
    x0_m, y0_m, x_min_p, x_max_p, y_min_p, y_max_p = _bounds_with_padding(
        xs, ys, cell_m=float(chosen_cell_m), pad_frac=float(cfg.pad_frac)
    )
    ix_min, ix_max, iy_min, iy_max = _index_extents(
        x0_m=float(x0_m),
        y0_m=float(y0_m),
        cell_m=float(chosen_cell_m),
        x_min_p=float(x_min_p),
        x_max_p=float(x_max_p),
        y_min_p=float(y_min_p),
        y_max_p=float(y_max_p),
    )

    # per-well assignment (pre-merge; used for well_to_cell row/col)
    assignment: Dict[str, Tuple[int, int, str]] = {}

    # initial bins by occupied cell
    bins: Dict[str, _Bin] = {}
    for wid, lat, lon, x_m, y_m in wells_xy:
        ix, iy = grid_index(x_m, y_m, x0_m=float(x0_m), y0_m=float(y0_m), cell_m=float(chosen_cell_m))
        cid = grid_cell_id(ix, iy)

        # NOTE: Step1 expects (cell_row, cell_col, cell_id)
        assignment[str(wid)] = (int(iy), int(ix), str(cid))

        b = bins.get(cid)
        if b is None:
            bins[cid] = _Bin(bin_id=cid, source_cells={cid}, wells=[(wid, float(lat), float(lon))])
        else:
            b.wells.append((wid, float(lat), float(lon)))

    n_bins_initial = int(len(bins))

    # Optional: merge tiny bins (set min_bin_size=1 to disable)
    bins, merge_log = _merge_small_bins(bins, min_bin_size=int(cfg.min_bin_size))
    n_bins_final = int(len(bins))

    # bins.csv
    bins_rows: List[Dict[str, str]] = []
    for bin_id in sorted(bins.keys()):
        b = bins[bin_id]
        for wid, _, _ in sorted(b.wells, key=lambda t: t[0]):
            bins_rows.append({"well_id": wid, "bin_id": bin_id})

    # bins_meta.csv
    bins_meta: List[Dict[str, Any]] = []
    radii: List[float] = []
    sizes: List[int] = []

    for bin_id in sorted(bins.keys()):
        b = bins[bin_id]
        pts = [(lat, lon) for _, lat, lon in b.wells]
        cen = _centroid_latlon(pts)
        rad = _radius_km(pts, cen)

        sizes.append(int(len(pts)))
        radii.append(float(rad))

        bins_meta.append(
            {
                "bin_id": bin_id,
                "n_wells": int(len(pts)),
                "centroid_lat": f"{cen[0]:.8f}",
                "centroid_lon": f"{cen[1]:.8f}",
                "radius_km": f"{rad:.3f}",
            }
        )

    def _pct(xs2: List[float], p: float) -> float:
        if not xs2:
            return 0.0
        ys2 = sorted(xs2)
        k = int(round((p / 100.0) * (len(ys2) - 1)))
        return float(ys2[max(0, min(k, len(ys2) - 1))])

    origin_lon, origin_lat = xy_m_to_lonlat(float(x0_m), float(y0_m), epsg=epsg)

    grid_info = {
        "crs_epsg": int(epsg),
        "cell_m": float(chosen_cell_m),
        "cell_km": float(chosen_cell_km),
        "origin_x_m": float(x0_m),
        "origin_y_m": float(y0_m),
        "origin_lon": float(origin_lon),
        "origin_lat": float(origin_lat),
        "x_min_m": float(x_min_p),
        "x_max_m": float(x_max_p),
        "y_min_m": float(y_min_p),
        "y_max_m": float(y_max_p),
        "ix_min": int(ix_min),
        "ix_max": int(ix_max),
        "iy_min": int(iy_min),
        "iy_max": int(iy_max),
    }

    diagnostics: Dict[str, Any] = {
        "method": "grid",
        "grid_cell_km_proposed": float(cfg.cell_km) if cfg.cell_km is not None else None,
        "chosen_grid_cell_km": float(chosen_cell_km),
        "chosen_grid_cell_m": float(chosen_cell_m),
        "occupancy_by_cell_km": occ_map,
        "scale_selection": sel_diag,
        "n_wells_in": int(len(wells)),
        "n_bins_initial": int(n_bins_initial),
        "n_bins_final": int(n_bins_final),
        "min_bin_size": int(cfg.min_bin_size),
        "bin_size_stats": {
            "min": int(min(sizes)) if sizes else 0,
            "p50": int(_pct([float(s) for s in sizes], 50.0)) if sizes else 0,
            "max": int(max(sizes)) if sizes else 0,
        },
        "radius_km_stats": {
            "min": float(min(radii)) if radii else 0.0,
            "p50": float(_pct(radii, 50.0)) if radii else 0.0,
            "max": float(max(radii)) if radii else 0.0,
        },
        "merge_log": merge_log,
        "grid": grid_info,
        "bin_sources": {
            bid: {"source_cells": sorted(list(bins[bid].source_cells)), "n_wells": bins[bid].n()}
            for bid in sorted(bins.keys())
        },
    }

    return bins_rows, bins_meta, diagnostics, assignment
