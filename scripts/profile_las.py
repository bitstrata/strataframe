# scripts/profile_las.py
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _safe_str(x: Any, maxlen: int = 200) -> str:
    s = "" if x is None else str(x)
    s = s.replace("\n", " ").replace("\r", " ").strip()
    return s[:maxlen]


def _well_value(las, mnemonic: str) -> str:
    try:
        it = las.well[mnemonic]
        return _safe_str(it.value)
    except Exception:
        return ""


def profile_one(path: Path) -> Dict[str, Any]:
    import lasio  # local import so script fails clearly if missing

    las = lasio.read(path, ignore_data=False)

    # Curves metadata
    curve_mnems = [c.mnemonic for c in las.curves]
    curve_units = [c.unit for c in las.curves]
    curve_desc = [c.descr for c in las.curves]

    # Dataframe
    df = las.df()  # depth index
    dtypes = df.dtypes.astype(str).to_dict()
    null_frac = df.isna().mean().to_dict()

    # Try to infer depth step stats
    idx = df.index.to_numpy()
    step_med = float(np.nanmedian(np.diff(idx))) if len(idx) > 2 else np.nan

    out: Dict[str, Any] = {
        "file": str(path),
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "index_name": _safe_str(df.index.name),
        "index_dtype": _safe_str(df.index.dtype),
        "depth_min": float(np.nanmin(idx)) if len(idx) else np.nan,
        "depth_max": float(np.nanmax(idx)) if len(idx) else np.nan,
        "depth_step_median": step_med,
        # Common well header fields (best-effort; varies by vendor)
        "UWI": _well_value(las, "UWI"),
        "API": _well_value(las, "API"),
        "WELL": _well_value(las, "WELL"),
        "WELLNAME": _well_value(las, "WELLNAME"),
        "LEASE": _well_value(las, "LEASE"),
        "FLD": _well_value(las, "FLD"),
        "LOC": _well_value(las, "LOC"),
        "LAT": _well_value(las, "LAT"),
        "LON": _well_value(las, "LON"),
        "COMP": _well_value(las, "COMP"),
        # Curves
        "curves": "|".join(curve_mnems),
        "curve_units": "|".join(_safe_str(u) for u in curve_units),
        "curve_desc": "|".join(_safe_str(d) for d in curve_desc),
        # dtype + null summaries (as compact JSON-ish strings)
        "dtypes": _safe_str(dtypes, 5000),
        "null_frac_top": _safe_str(
            dict(sorted(null_frac.items(), key=lambda kv: kv[1], reverse=True)[:10]),
            2000,
        ),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/las"))
    ap.add_argument("--n", type=int, default=10, help="Number of files to sample")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", type=Path, default=Path("out/las_profile.csv"))
    args = ap.parse_args()

    files = sorted(args.root.rglob("*.las"))
    if not files:
        raise SystemExit(f"No .las files found under {args.root}")

    rng = np.random.default_rng(args.seed)
    if args.n < len(files):
        pick_idx = rng.choice(len(files), size=args.n, replace=False)
        picks = [files[i] for i in sorted(pick_idx)]
    else:
        picks = files

    rejects = []
    rows = []

    for p in picks:
        try:
            rows.append(profile_one(p))
            print(f"[OK] {p}")
        except Exception as e:
            rejects.append({"file": str(p), "error": str(e)})
            print(f"[REJECT] {p} -> {e}")

    # write outputs
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    if rejects:
        rej = pd.DataFrame(rejects)
        rej_path = args.out.with_name("las_rejects.csv")
        rej.to_csv(rej_path, index=False)
        print(f"Wrote rejects: {rej_path} ({len(rejects)})")


if __name__ == "__main__":
    main()
