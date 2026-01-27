# src/strataframe/io/wells_gr_index.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Set


def load_las_basenames_from_wells_gr(wells_gr_parquet: Path) -> Optional[Set[str]]:
    """
    Return set of LAS basenames from Step0 wells_gr.parquet (column: las_path).

    Supports both:
      - single parquet file
      - parquet dataset directory (wells_gr.parquet/part-*.parquet)
    """
    p = Path(wells_gr_parquet)
    if not p.exists():
        return None

    # Prefer pyarrow.dataset for column-only scanning
    try:
        import pyarrow.dataset as ds  # type: ignore

        dataset = ds.dataset(str(p), format="parquet")
        table = dataset.to_table(columns=["las_path"])
        arr = table.column("las_path").to_pylist()
        return {Path(x).name for x in arr if isinstance(x, str) and x.strip()}
    except Exception:
        pass

    # Fallback to pandas
    try:
        import pandas as pd  # type: ignore

        df = pd.read_parquet(p)
        cols = {c.lower(): c for c in df.columns}
        c = cols.get("las_path")
        if c is None:
            return None
        return {Path(x).name for x in df[c].astype(str).tolist() if str(x).strip()}
    except Exception:
        return None
