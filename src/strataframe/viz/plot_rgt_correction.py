# src/strataframe/viz/plot_rgt_correction.py
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None  # type: ignore

from strataframe.io.csv import read_csv_rows

# Prefer correlation.well if your codebase has it (matches plot_dtw_pair),
# otherwise fall back to preprocess.well.
try:  # pragma: no cover
    from strataframe.correlation.well import Well, WellLoadConfig  # type: ignore
except Exception:  # pragma: no cover
    from strataframe.preprocess.well import Well, WellLoadConfig  # type: ignore

from strataframe.rgt.monotonic import MonotonicConfig, enforce_monotonic_rgt


def _require() -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required. Install with: pip install matplotlib")


def _get_first_present(row: Dict[str, str], keys: List[str]) -> str:
    for k in keys:
        v = (row.get(k) or "").strip()
        if v:
            return v
    return ""


def _float_or_nan(x: str) -> float:
    try:
        return float((x or "").strip())
    except Exception:
        return float("nan")


def _rep_lookup(nodes_rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for r in nodes_rows:
        rid = _get_first_present(r, ["rep_id", "well_id", "node_id", "id"])
        if rid:
            out[rid] = r
    return out


def _load_shifts_for_rep(
    shifts_npz: Path,
    rep_id: str,
) -> np.ndarray:
    """
    Supports two common NPZ layouts:

    (A) Per-rep arrays stored under key == rep_id:
        z[rep_id] -> (nS,)

    (B) Matrix layout:
        z["rep_ids"] -> (W,) strings
        z["shifts"] or z["shift"] -> (W, nS)
    """
    with np.load(shifts_npz, allow_pickle=False) as z:
        # Layout A
        if rep_id in z.files:
            return np.asarray(z[rep_id], dtype="float64").reshape(-1)

        # Layout B
        if "rep_ids" in z.files and ("shifts" in z.files or "shift" in z.files):
            rep_ids = np.asarray(z["rep_ids"], dtype=np.str_)
            mat_key = "shifts" if "shifts" in z.files else "shift"
            mat = np.asarray(z[mat_key], dtype="float64")
            if mat.ndim != 2 or mat.shape[0] != rep_ids.size:
                raise ValueError(
                    f"Unexpected matrix shape in {shifts_npz}: {mat_key}={mat.shape}, rep_ids={rep_ids.shape}"
                )
            m = np.where(rep_ids.astype(str) == str(rep_id))[0]
            if m.size == 0:
                raise KeyError(
                    f"rep_id '{rep_id}' not found in {mat_key} matrix. Example keys: {z.files[:20]} ..."
                )
            return mat[int(m[0]), :].reshape(-1)

        raise KeyError(
            f"rep_id '{rep_id}' not found in shifts npz and no (rep_ids, shifts/shift) matrix present. "
            f"Available keys: {z.files[:30]} ..."
        )


def _list_shift_ids(shifts_npz: Path) -> List[str]:
    with np.load(shifts_npz, allow_pickle=False) as z:
        # Layout B
        if "rep_ids" in z.files:
            rep_ids = np.asarray(z["rep_ids"], dtype=np.str_)
            return [str(x) for x in rep_ids.tolist()]
        # Layout A (all keys that look like ids)
        return [str(k) for k in z.files]


def _pick_rep_id(
    *,
    rep_id: str,
    random_pick: bool,
    seed: int,
    nodes_reps: Dict[str, Dict[str, str]],
    shifts_npz: Path,
) -> str:
    rep_id = str(rep_id).strip()

    if rep_id and not random_pick:
        return rep_id

    # random selection from intersection of nodes CSV and shifts NPZ
    ids_npz = set(_list_shift_ids(shifts_npz))
    ids_nodes = set(nodes_reps.keys())
    common = sorted(ids_npz.intersection(ids_nodes))

    if not common:
        # fallback: anything in NPZ
        fallback = sorted(ids_npz)
        if not fallback:
            raise ValueError(f"No rep ids found in shifts npz: {shifts_npz}")
        rnd = random.Random(int(seed))
        return rnd.choice(fallback)

    rnd = random.Random(int(seed))
    return rnd.choice(common)


def main(argv: Optional[List[str]] = None) -> int:
    _require()

    ap = argparse.ArgumentParser(description="Visualize per-well RGT shift correction from rgt_shifts_resampled.npz.")
    ap.add_argument("--framework-dir", type=str, default="", help="Directory containing framework_nodes.csv and rgt_shifts_resampled.npz")
    ap.add_argument("--nodes-csv", type=str, default="", help="Path to framework_nodes.csv (overrides --framework-dir)")
    ap.add_argument("--shifts-npz", type=str, default="", help="Path to rgt_shifts_resampled.npz (overrides --framework-dir)")

    # NEW: rep selection modes
    ap.add_argument("--rep-id", type=str, default="", help="rep_id key inside the shifts npz (optional if --random)")
    ap.add_argument("--random", action="store_true", help="Pick a random rep_id (prefers intersection of nodes CSV and shifts NPZ)")
    ap.add_argument("--seed", type=int, default=13, help="Seed used when selecting a random rep_id")
    ap.add_argument("--list-ids", action="store_true", help="Print available rep ids in shifts NPZ and exit")

    ap.add_argument("--curve-fallback", type=str, default="GR", help="Fallback curve if nodes CSV has none")
    ap.add_argument("--monotonic", type=str, default="nondecreasing", help="nondecreasing|nonincreasing (for optional monotonic fix display)")
    ap.add_argument("--show-monotonic", action="store_true", help="Also plot monotonic-enforced RGT and reversal mask fraction")
    ap.add_argument("--las-root", type=str, default="", help="Optional LAS root for relative las_path entries in nodes CSV")
    ap.add_argument("--out", type=str, default="", help="Optional output image path (png). If omitted, shows interactive window.")
    ap.add_argument("--dpi", type=int, default=160)

    args = ap.parse_args(argv)

    if args.framework_dir:
        fdir = Path(args.framework_dir)
        nodes_csv = Path(args.nodes_csv) if args.nodes_csv else (fdir / "framework_nodes.csv")
        shifts_npz = Path(args.shifts_npz) if args.shifts_npz else (fdir / "rgt_shifts_resampled.npz")
    else:
        if not args.nodes_csv or not args.shifts_npz:
            raise SystemExit("Provide either --framework-dir or both --nodes-csv and --shifts-npz.")
        nodes_csv = Path(args.nodes_csv)
        shifts_npz = Path(args.shifts_npz)

    if bool(args.list_ids):
        ids = _list_shift_ids(shifts_npz)
        print(f"{shifts_npz} rep ids (count={len(ids)}):")
        for x in ids[:200]:
            print(" ", x)
        if len(ids) > 200:
            print(f"... ({len(ids)-200} more)")
        return 0

    if not nodes_csv.exists():
        raise FileNotFoundError(f"nodes CSV not found: {nodes_csv}")
    if not shifts_npz.exists():
        raise FileNotFoundError(f"shifts NPZ not found: {shifts_npz}")

    nodes_rows = read_csv_rows(nodes_csv)
    reps = _rep_lookup(nodes_rows)

    rep_id = _pick_rep_id(
        rep_id=str(args.rep_id),
        random_pick=bool(args.random),
        seed=int(args.seed),
        nodes_reps=reps,
        shifts_npz=shifts_npz,
    )

    rr = reps.get(rep_id)
    if rr is None:
        raise ValueError(f"rep_id not found in nodes CSV: {rep_id}")

    las_path_s = (rr.get("las_path") or rr.get("las") or "").strip()
    if not las_path_s:
        raise ValueError(f"Missing las_path for rep_id={rep_id} in nodes CSV")

    las_root = Path(args.las_root) if str(args.las_root).strip() else None

    las_path = Path(las_path_s)
    if not las_path.is_absolute():
        # Prefer explicit las_root; else resolve relative to nodes CSV folder
        if las_root is not None:
            las_path = las_root / las_path
        else:
            las_path = nodes_csv.parent / las_path

    if not las_path.exists():
        raise FileNotFoundError(f"LAS not found: {las_path}")

    # Load shift vector
    shift = _load_shifts_for_rep(shifts_npz, rep_id=rep_id)

    nS = int(shift.size)
    if nS <= 2:
        raise ValueError(f"Shift array too short: n={nS}")

    curve_used = (_get_first_present(rr, ["curve_used", "curve", "mnemonic"]) or str(args.curve_fallback)).strip()

    w = Well.from_las(
        well_id=rep_id,
        lat=_float_or_nan(rr.get("lat", "") or "nan"),
        lon=_float_or_nan(rr.get("lon", "") or "nan"),
        las_path=las_path,
        cfg=WellLoadConfig(curve=curve_used, resample_n=int(nS), pct_lo=1.0, pct_hi=99.0, fill_nans=True),
    )

    if int(w.depth_rs.size) != int(shift.size):
        raise RuntimeError(f"Size mismatch after load: depth_rs={w.depth_rs.size} vs shift={shift.size}")

    depth = np.asarray(w.depth_rs, dtype="float64")
    rgt = depth + shift

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"RGT correction: {rep_id} | n={nS} | curve={getattr(w, 'curve_used', curve_used)}")

    ax0, ax1 = axes

    # Shift vs sample index
    ax0.plot(shift)
    ax0.set_title("Shift s[i]")
    ax0.set_xlabel("sample index")
    ax0.set_ylabel("shift (depth units)")

    # Depth and corrected coordinate vs sample index
    ax1.plot(depth, label="depth_rs")
    ax1.plot(rgt, label="rgt = depth_rs + shift")
    ax1.set_title("Depth vs RGT-corrected coordinate")
    ax1.set_xlabel("sample index")
    ax1.set_ylabel("depth units")
    ax1.legend(loc="best", fontsize=8)

    if bool(args.show_monotonic):
        mode = str(args.monotonic).strip().lower()
        mc = MonotonicConfig(mode=mode)  # type: ignore[arg-type]
        rgt_fix, rev = enforce_monotonic_rgt(rgt, cfg=mc)
        rev = np.asarray(rev, dtype=bool)

        fin = np.isfinite(rgt)
        frac = float(np.mean(rev[fin]) if int(fin.sum()) > 0 else 0.0)

        ax1.plot(rgt_fix, linestyle="--", label=f"monotonic ({mc.mode}), rev_frac={frac:.3f}")
        ax1.legend(loc="best", fontsize=8)

    fig.tight_layout()

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=int(args.dpi), bbox_inches="tight")
        print(f"Wrote: {out}")
        print(f"rep_id: {rep_id} | nS={nS} | curve={getattr(w, 'curve_used', curve_used)}")
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
