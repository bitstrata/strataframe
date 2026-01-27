# src/strataframe/graph/select_representatives.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .las_utils import (
    list_curve_mnemonics,
    normalize_mnemonic,  # canonicalizes (via shared curves.normalize_header)
    read_las_header_only,
)

from strataframe.io.csv import read_csv_rows, write_csv


# -----------------------------------------------------------------------------
# Scoring policy (simple, explicit, ChronoLog-friendly)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class RepSelectConfig:
    reps_per_cell: int = 2
    max_candidates_per_cell: int = 25

    # Require GR present (hard gate)
    require_gr: bool = True

    # Canonical-only curve families (aliases are handled by normalize_mnemonic()).
    gr_canon: Tuple[str, ...] = ("GR",)

    # Density family (keep RHOC distinct in data model, but treat as density for scoring)
    density_canon: Tuple[str, ...] = ("RHOB", "RHOC")

    # Neutron family
    neutron_canon: Tuple[str, ...] = ("NPHI", "NPOR", "CNPOR")

    # Density/total porosity family
    porosity_canon: Tuple[str, ...] = ("DPHI", "DPOR", "SPOR")

    # Photoelectric
    pe_canon: Tuple[str, ...] = ("PE",)

    # Sonic
    sonic_canon: Tuple[str, ...] = ("DT", "DTC", "DTS")

    # Weights
    w_gr: float = 10.0
    w_por: float = 4.0
    w_den: float = 2.0
    w_neu: float = 2.0
    w_pe: float = 3.0
    w_dt: float = 1.0

    # Synergy: density+neutron together is particularly useful
    w_den_neu_synergy: float = 2.0


def _pick_first_present(canon_set: set[str], prefs: Sequence[str]) -> str:
    """
    Return the first CANONICAL mnemonic in prefs that exists in canon_set.
    """
    for p in prefs:
        c = normalize_mnemonic(p)
        if c and c in canon_set:
            return c
    return ""


def score_log_suite(mnemonics_raw: Sequence[str], cfg: RepSelectConfig) -> Tuple[float, Dict[str, Any]]:
    """
    Score a well's header curve set using CANONICAL mnemonics only.
    """
    canon = [normalize_mnemonic(m) for m in mnemonics_raw if (m or "").strip()]
    canon_set = {c for c in canon if c}

    picked_gr = _pick_first_present(canon_set, cfg.gr_canon)
    picked_den = _pick_first_present(canon_set, cfg.density_canon)
    picked_neu = _pick_first_present(canon_set, cfg.neutron_canon)
    picked_por = _pick_first_present(canon_set, cfg.porosity_canon)
    picked_pe = _pick_first_present(canon_set, cfg.pe_canon)
    picked_dt = _pick_first_present(canon_set, cfg.sonic_canon)

    info: Dict[str, Any] = {
        "has_gr": bool(picked_gr),
        "has_por": bool(picked_por),
        "has_den": bool(picked_den),
        "has_neu": bool(picked_neu),
        "has_pe": bool(picked_pe),
        "has_dt": bool(picked_dt),
        # IMPORTANT: these are CANONICAL picks (not raw LAS header names)
        "picked": {
            "GR": picked_gr,
            "POR": picked_por,
            "DEN": picked_den,
            "NEU": picked_neu,
            "PE": picked_pe,
            "DT": picked_dt,
        },
        "n_unique_canon": int(len(canon_set)),
    }

    if cfg.require_gr and not picked_gr:
        return -1e9, info

    score = 0.0
    if picked_gr:
        score += cfg.w_gr
    if picked_por:
        score += cfg.w_por
    if picked_den:
        score += cfg.w_den
    if picked_neu:
        score += cfg.w_neu
    if picked_pe:
        score += cfg.w_pe
    if picked_dt:
        score += cfg.w_dt
    if picked_den and picked_neu:
        score += cfg.w_den_neu_synergy

    score += 0.01 * float(len(canon_set))
    return float(score), info


# -----------------------------------------------------------------------------
# Path resolution
# -----------------------------------------------------------------------------

def resolve_las_path_from_url(url: str, las_root: Path) -> Optional[Path]:
    u = (url or "").strip()
    if not u:
        return None
    basename = u.split("/")[-1]
    p = las_root / basename
    return p if p.exists() else None


# -----------------------------------------------------------------------------
# Main selection
# -----------------------------------------------------------------------------

def select_representatives(
    well_to_cell_rows: List[Dict[str, str]],
    *,
    las_root: Path,
    cfg: RepSelectConfig,
    # Step-2 controls
    n_rep_target: int = 1000,
    quota_mode: str = "equal",          # "equal" | "proportional"
    q_min: int = 5,                     # only used for proportional
    candidate_method: str = "farthest", # "farthest" | "random"
    seed: int = 42,
    # If provided, we will write representatives_meta.json here
    out_dir: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Step 2 selector (grid workflow):
      - cells = cell_id
      - quota allocation to hit ~n_rep_target (equal or proportional)
      - candidate preselection by coordinates ONLY (farthest/random) to limit LAS reads
      - score candidates by LAS header suite; require GR if cfg.require_gr
      - preserves representatives.csv schema

    Returns:
      reps_out (list of row dicts for representatives.csv),
      diag (diagnostics dict; includes 'representatives_meta' payload)
    """
    import hashlib
    from datetime import datetime, timezone

    import numpy as np

    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _stable_hash32(s: str) -> int:
        h = hashlib.md5((s or "").encode("utf-8")).hexdigest()[:8]
        return int(h, 16) & 0xFFFFFFFF

    def _as_float(v: Any) -> float:
        try:
            return float(v)
        except Exception:
            return float("nan")

    def _haversine_km_vec(lat1r, lon1r, lat2r, lon2r) -> np.ndarray:
        dlat = lat2r - lat1r
        dlon = lon2r - lon1r
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
        a = np.clip(a, 0.0, 1.0)
        return 2.0 * 6371.0088 * np.arcsin(np.sqrt(a))

    def _farthest_point_sample_indices(lat: np.ndarray, lon: np.ndarray, k: int, s: int) -> np.ndarray:
        n = int(lat.shape[0])
        if k <= 0:
            return np.array([], dtype=int)
        if k >= n:
            return np.arange(n, dtype=int)

        rng = np.random.default_rng(int(s) & 0xFFFFFFFF)

        latr = np.radians(lat.astype("float64", copy=False))
        lonr = np.radians(lon.astype("float64", copy=False))

        latc = float(np.mean(latr))
        lonc = float(np.mean(lonr))
        d0 = _haversine_km_vec(latr, lonr, latc, lonc)
        maxd = float(np.max(d0))
        cand = np.where(np.isclose(d0, maxd))[0]
        start = int(rng.choice(cand))

        selected = [start]
        min_d = _haversine_km_vec(latr, lonr, latr[start], lonr[start])

        for _ in range(1, int(k)):
            nxt = int(np.argmax(min_d))
            selected.append(nxt)
            d_new = _haversine_km_vec(latr, lonr, latr[nxt], lonr[nxt])
            min_d = np.minimum(min_d, d_new)

        return np.asarray(selected, dtype=int)

    def _rebalance_quotas(quotas: Dict[str, int], caps: Dict[str, int], target: int) -> Dict[str, int]:
        q = {c: max(0, min(int(quotas.get(c, 0)), int(caps.get(c, 0)))) for c in caps}
        total = int(sum(q.values()))
        target = int(max(0, target))

        if total == target:
            return q

        def _add_order():
            rem = {c: int(caps[c] - q[c]) for c in q}
            return sorted(rem.keys(), key=lambda c: (-rem[c], c))

        def _sub_order():
            return sorted(q.keys(), key=lambda c: (-q[c], c))

        if total < target:
            need = target - total
            while need > 0:
                order = _add_order()
                if not order:
                    break
                progressed = False
                for c in order:
                    if need <= 0:
                        break
                    if q[c] < caps[c]:
                        q[c] += 1
                        need -= 1
                        progressed = True
                if not progressed:
                    break
            return q

        over = total - target
        while over > 0:
            order = _sub_order()
            if not order:
                break
            progressed = False
            for c in order:
                if over <= 0:
                    break
                if q[c] > 0:
                    q[c] -= 1
                    over -= 1
                    progressed = True
            if not progressed:
                break
        return q

    def _compute_quotas(cell_caps: Dict[str, int], *, n_rep: int, mode: str, qmin: int) -> Dict[str, int]:
        nonempty = [c for c, n in cell_caps.items() if int(n) > 0]
        if not nonempty:
            return {c: 0 for c in cell_caps}

        n_total = int(sum(int(cell_caps[c]) for c in nonempty))
        n_rep_eff = int(min(max(0, int(n_rep)), n_total)) if int(n_rep) > 0 else 0

        if n_rep_eff <= 0:
            q0 = int(max(0, int(cfg.reps_per_cell)))
            base = {c: min(q0, int(cell_caps[c])) for c in nonempty}
            out = {c: 0 for c in cell_caps}
            out.update(base)
            return out

        if mode not in {"equal", "proportional"}:
            raise ValueError(f"quota_mode must be 'equal' or 'proportional' (got {mode!r})")

        if mode == "equal":
            q_each = int(np.ceil(float(n_rep_eff) / float(len(nonempty))))
            base = {c: min(q_each, int(cell_caps[c])) for c in nonempty}
        else:
            qmin_i = int(max(0, int(qmin)))
            base = {}
            for c in nonempty:
                cap = int(cell_caps[c])
                qb = int(round(float(n_rep_eff) * float(cap) / float(n_total))) if n_total > 0 else 0
                qb = max(qmin_i, qb)
                qb = min(qb, cap)
                base[c] = qb

        base = _rebalance_quotas(base, {c: int(cell_caps[c]) for c in nonempty}, target=n_rep_eff)

        out = {c: 0 for c in cell_caps}
        out.update(base)
        return out

    # Group rows by cell_id
    by_cell: Dict[str, List[Dict[str, str]]] = {}
    for r in well_to_cell_rows:
        cid = (r.get("cell_id", "") or "").strip()
        if not cid:
            continue
        by_cell.setdefault(cid, []).append(r)

    cell_caps: Dict[str, int] = {c: int(len(rows)) for c, rows in by_cell.items()}
    n_cells = int(len(by_cell))
    n_total_wells = int(sum(cell_caps.values()))

    quotas = _compute_quotas(
        cell_caps,
        n_rep=int(n_rep_target),
        mode=str(quota_mode),
        qmin=int(q_min),
    )
    n_rep_effective = int(sum(quotas.values()))

    reps_out: List[Dict[str, Any]] = []
    diag: Dict[str, Any] = {
        "n_cells": n_cells,
        "n_total_wells": n_total_wells,
        "n_rep_target": int(n_rep_target),
        "n_rep_effective": n_rep_effective,
        "quota_mode": str(quota_mode),
        "q_min": int(q_min),
        "candidate_method": str(candidate_method),
        "seed": int(seed),
        "cells": {},
        "skipped_missing_las": 0,
        "skipped_read_fail": 0,
        "skipped_no_gr": 0,
        "created_utc": _utc_now_iso(),
    }

    rep_id = 0

    for cell_id, rows in sorted(by_cell.items(), key=lambda t: t[0]):
        cap = int(len(rows))
        q_cell = int(quotas.get(cell_id, 0))
        if q_cell <= 0 or cap <= 0:
            diag["cells"][cell_id] = {"n_wells_in_cell": cap, "quota": q_cell, "n_candidates": 0, "n_scored": 0, "n_selected": 0}
            continue

        lats = np.array([_as_float(r.get("lat", "")) for r in rows], dtype="float64")
        lons = np.array([_as_float(r.get("lon", "")) for r in rows], dtype="float64")

        ok = np.isfinite(lats) & np.isfinite(lons)
        if int(np.count_nonzero(ok)) < cap:
            rows_idx = [i for i in range(cap) if bool(ok[i])]
            rows2 = [rows[i] for i in rows_idx]
            lats = lats[ok]
            lons = lons[ok]
            rows = rows2
            cap = int(len(rows))
            q_cell = int(min(q_cell, cap))

        if cap <= 0 or q_cell <= 0:
            diag["cells"][cell_id] = {"n_wells_in_cell": int(len(by_cell.get(cell_id, []))), "quota": int(quotas.get(cell_id, 0)), "n_candidates": 0, "n_scored": 0, "n_selected": 0}
            continue

        max_cand = int(max(int(cfg.max_candidates_per_cell), q_cell))
        max_cand = int(min(max_cand, cap))

        cell_seed = (int(seed) + _stable_hash32(cell_id)) & 0xFFFFFFFF
        rng = np.random.default_rng(cell_seed)

        method = str(candidate_method).strip().lower()
        if method not in {"farthest", "random"}:
            raise ValueError(f"candidate_method must be 'farthest' or 'random' (got {candidate_method!r})")

        if method == "random":
            cand_idx = rng.choice(np.arange(cap, dtype=int), size=max_cand, replace=False)
            cand_idx = np.asarray(cand_idx, dtype=int)
        else:
            cand_idx = _farthest_point_sample_indices(lats, lons, k=max_cand, s=cell_seed)

        cand_set = set(int(i) for i in cand_idx.tolist())
        remaining_idx = [i for i in range(cap) if i not in cand_set]
        backfill_order = remaining_idx
        rng.shuffle(backfill_order)

        attempt_order = cand_idx.tolist() + backfill_order

        scored: List[Tuple[float, Dict[str, Any]]] = []
        n_attempted = 0

        for i in attempt_order:
            if n_attempted >= (2 * max_cand):
                break

            r = rows[int(i)]
            url = (r.get("url", "") or "").strip()
            las_path = resolve_las_path_from_url(url, las_root)
            if las_path is None:
                diag["skipped_missing_las"] += 1
                n_attempted += 1
                continue

            try:
                las = read_las_header_only(las_path)
                mn_raw = list_curve_mnemonics(las)
            except Exception:
                diag["skipped_read_fail"] += 1
                n_attempted += 1
                continue

            score, info = score_log_suite(mn_raw, cfg)
            if score <= -1e8:
                diag["skipped_no_gr"] += 1
                n_attempted += 1
                continue

            scored.append((float(score), {"row": r, "las_path": str(las_path), "info": info}))
            n_attempted += 1

        scored.sort(key=lambda t: t[0], reverse=True)
        chosen = scored[:q_cell]

        diag["cells"][cell_id] = {
            "n_wells_in_cell": int(cap),
            "quota": int(q_cell),
            "n_candidates": int(max_cand),
            "n_attempted": int(n_attempted),
            "n_scored": int(len(scored)),
            "n_selected": int(len(chosen)),
        }

        # Stable tag (grid) if present in row; else blank
        cell_tag = (rows[0].get("cell_tag", "") or "").strip()
        if not cell_tag:
            gk = (rows[0].get("grid_km", "") or "").strip()
            try:
                cell_tag = f"grid:{float(gk):g}km" if gk else ""
            except Exception:
                cell_tag = ""

        for s, payload in chosen:
            r = payload["row"]
            info = payload["info"]
            rep_id += 1

            reps_out.append(
                {
                    "rep_id": int(rep_id),
                    "cell_id": cell_id,
                    "cell_tag": cell_tag,
                    "score": f"{float(s):.3f}",
                    "url": r.get("url", ""),
                    "kgs_id": r.get("kgs_id", ""),
                    "api": r.get("api", ""),
                    "api_num_nodash": r.get("api_num_nodash", ""),
                    "operator": r.get("operator", ""),
                    "lease": r.get("lease", ""),
                    "lat": r.get("lat", ""),
                    "lon": r.get("lon", ""),
                    "las_path": payload["las_path"],
                    "picked_gr": info["picked"].get("GR", ""),
                    "picked_por": info["picked"].get("POR", ""),
                    "picked_den": info["picked"].get("DEN", ""),
                    "picked_neu": info["picked"].get("NEU", ""),
                    "picked_pe": info["picked"].get("PE", ""),
                    "picked_dt": info["picked"].get("DT", ""),
                }
            )

    counts_per_cell: Dict[str, int] = {}
    for r in reps_out:
        c = str(r.get("cell_id", "") or "")
        counts_per_cell[c] = counts_per_cell.get(c, 0) + 1

    representatives_meta: Dict[str, Any] = {
        "created_utc": diag["created_utc"],
        "n_rep_target": int(n_rep_target),
        "n_rep_effective": int(n_rep_effective),
        "n_rep_selected": int(len(reps_out)),
        "n_cells": int(n_cells),
        "quota_mode": str(quota_mode),
        "q_min": int(q_min),
        "candidate_method": str(candidate_method),
        "seed": int(seed),
        "max_candidates_per_cell": int(cfg.max_candidates_per_cell),
        "counts_per_cell": {str(k): int(v) for k, v in sorted(counts_per_cell.items())},
        "cells": {
            str(cell): {
                "n_wells": int(cell_caps.get(cell, 0)),
                "quota": int(quotas.get(cell, 0)),
                "n_selected": int(diag["cells"].get(cell, {}).get("n_selected", 0)),
            }
            for cell in sorted(by_cell.keys())
        },
    }

    diag["representatives_meta"] = representatives_meta

    if out_dir is not None:
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "representatives_meta.json").write_text(
                json.dumps(representatives_meta, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            diag["representatives_meta_path"] = str(out_dir / "representatives_meta.json")
        except Exception as e:
            diag["representatives_meta_write_error"] = str(e)

    return reps_out, diag


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Select representative wells per grid cell using log-suite scoring.")
    ap.add_argument("--well-to-cell-csv", type=str, required=True, help="Path to out/.../well_to_cell.csv")
    ap.add_argument("--las-root", type=str, default="data/las", help="Directory containing LAS files")
    ap.add_argument("--out-dir", type=str, required=True, help="Output directory")
    ap.add_argument("--reps-per-cell", type=int, default=2, help="Representatives per cell (default 2)")
    ap.add_argument("--max-candidates-per-cell", type=int, default=25, help="Max candidates to score per cell (default 25)")
    args = ap.parse_args(argv)

    well_to_cell_csv = Path(args.well_to_cell_csv)
    las_root = Path(args.las_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv_rows(well_to_cell_csv)

    cfg = RepSelectConfig(
        reps_per_cell=int(args.reps_per_cell),
        max_candidates_per_cell=int(args.max_candidates_per_cell),
    )

    reps, diag = select_representatives(rows, las_root=las_root, cfg=cfg, out_dir=out_dir)

    write_csv(
        out_dir / "representatives.csv",
        [
            "rep_id", "cell_id", "cell_tag", "score",
            "url", "kgs_id", "api", "api_num_nodash", "operator", "lease", "lat", "lon",
            "las_path",
            "picked_gr", "picked_por", "picked_den", "picked_neu", "picked_pe", "picked_dt",
        ],
        reps,
    )

    (out_dir / "rep_selection_diagnostics.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")

    print(f"Cells processed: {diag['n_cells']}")
    print(f"Representatives selected: {len(reps)} (reps_per_cell={cfg.reps_per_cell})")
    print(f"Wrote: {out_dir / 'representatives.csv'}")
    print(f"Wrote: {out_dir / 'rep_selection_diagnostics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
