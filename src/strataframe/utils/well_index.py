# src/strataframe/utils/well_index.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from strataframe.io.csv import read_csv_rows, to_float


def read_well_index_rows(path: Path) -> List[Dict[str, Any]]:
    """
    Supports:
      - CSV/TSV/etc via read_csv_rows
      - Parquet via pandas (if available)
    """
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        if pd is None:
            raise RuntimeError("pandas is required to read parquet well index. Install pandas + pyarrow.")
        return pd.read_parquet(path).to_dict(orient="records")

    rows = read_csv_rows(path)
    return [{k: (v if v is not None else "") for k, v in r.items()} for r in rows]


def coords_from_row(r: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    Return a dict with lat/lon and/or x/y, plus coord_mode.

    coord_mode:
      - "latlon" if lat+lon present
      - "xy" if x+y present
      - "" if none
    """
    lat = to_float(str(r.get("lat", "") or r.get("latitude", "") or ""))
    lon = to_float(str(r.get("lon", "") or r.get("longitude", "") or ""))

    x = to_float(str(r.get("x", "") or ""))
    y = to_float(str(r.get("y", "") or ""))

    mode = ""
    if lat is not None and lon is not None:
        mode = "latlon"
    elif x is not None and y is not None:
        mode = "xy"

    return {
        "coord_mode": mode,
        "lat": None if lat is None else float(lat),
        "lon": None if lon is None else float(lon),
        "x": None if x is None else float(x),
        "y": None if y is None else float(y),
    }
