# src/strataframe/io/ks_manifest.py
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional


def read_ks_manifest(path: Path) -> List[Dict[str, str]]:
    """
    Reads data/ks_las_files.txt (quoted CSV) and returns rows as dicts.
    Expected columns include (common):
      URL, KGS_ID, API, API_NUM_NODASH, Operator, Lease, Latitude, Longitude, ...
    """
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        if rdr.fieldnames is None:
            raise ValueError(f"No header row found in {path}")
        out: List[Dict[str, str]] = []
        for r in rdr:
            out.append({k: (v if v is not None else "") for k, v in r.items()})
        return out


def api_nodash(api: str, api_num_nodash: str) -> str:
    if (api_num_nodash or "").strip():
        return (api_num_nodash or "").strip()
    return "".join(ch for ch in (api or "") if ch.isdigit())


def url_to_local_path(url: str, las_root: Path) -> Path:
    """
    Deterministic mapping from KGS URL to local file path.
    Default: use trailing filename from URL.
    """
    name = (url or "").strip().split("/")[-1]
    return Path(las_root) / name


def get_str(row: Dict[str, str], *keys: str) -> str:
    for k in keys:
        v = (row.get(k, "") or "").strip()
        if v:
            return v
    return ""


def resolve_manifest_row_to_las_path(row: Dict[str, str], las_root: Path) -> Optional[Path]:
    """
    Resolve LAS path from a manifest row.
    Primary source is URL -> basename under las_root.
    """
    url = get_str(row, "URL", "url")
    if not url:
        return None
    p = url_to_local_path(url, Path(las_root))
    return p
