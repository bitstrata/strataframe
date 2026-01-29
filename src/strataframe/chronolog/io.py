# src/strataframe/chronolog/io.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def read_parquet_any(path: Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    if p.is_dir():
        try:
            import pyarrow.dataset as ds  # type: ignore
            return ds.dataset(str(p), format="parquet").to_table().to_pandas()
        except Exception:
            parts = sorted([x for x in p.glob("*.parquet") if x.is_file()])
            if not parts:
                raise ValueError(f"No parquet part files found in directory: {p}")
            return pd.concat([pd.read_parquet(x) for x in parts], ignore_index=True)

    return pd.read_parquet(p)


def load_wells_gr(path: Path, *, require_latlon: bool = True) -> pd.DataFrame:
    df = read_parquet_any(path) if (path.is_dir() or path.suffix.lower() == ".parquet") else pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    lat_col = cols.get("lat")
    lon_col = cols.get("lon")
    if require_latlon and (lat_col is None or lon_col is None):
        raise ValueError(f"wells_gr missing lat/lon columns. Found: {list(df.columns)}")

    if lat_col is not None:
        df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    if lon_col is not None:
        df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")

    if require_latlon:
        df = df[df[lat_col].notna() & df[lon_col].notna()].copy()

    return df
