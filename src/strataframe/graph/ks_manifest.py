# src/strataframe/graph/ks_manifest.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from strataframe.io.csv import read_csv_rows, to_float


@dataclass(frozen=True)
class WellRecord:
    """
    Minimal well record extracted from data/ks_las_files.txt.

    Design intent:
      - Keep original row context (`row`) so downstream steps can join/aggregate.
      - Store lat/lon as floats (rows without usable coordinates are dropped).
    """
    url: str
    kgs_id: str
    api: str
    api_num_nodash: str
    operator: str
    lease: str
    lat: float
    lon: float
    row: Dict[str, str]


def _get(row: Dict[str, str], key: str, default: str = "") -> str:
    """
    Case-insensitive, whitespace-tolerant column getter for CSV dict rows.
    Prefers exact key match, then a normalized (upper/strip) match.
    """
    if key in row:
        v = row.get(key, default)
        return v if v is not None else default

    k_up = key.strip().upper()
    for k, v in row.items():
        if (k or "").strip().upper() == k_up:
            return v if v is not None else default
    return default


def api_nodash(api: str, api_num_nodash: str) -> str:
    """
    Prefer explicit API_NUM_NODASH; otherwise derive from API by extracting digits.
    """
    s = (api_num_nodash or "").strip()
    if s:
        return s
    a = (api or "").strip()
    if not a:
        return ""
    return "".join(ch for ch in a if ch.isdigit())


def select_well_id(w: WellRecord) -> str:
    """
    Stable well_id selection used across the pipeline.

    Preference order:
      1) api_num_nodash
      2) api
      3) kgs_id
      4) URL basename
      5) deterministic fallback from coordinates
    """
    for v in (w.api_num_nodash, w.api, w.kgs_id):
        s = (v or "").strip()
        if s:
            return s
    u = (w.url or "").strip()
    if u:
        return u.split("/")[-1]
    return f"well_{abs(hash((w.lat, w.lon))) % 10_000_000}"


def well_id_candidates(w: WellRecord) -> List[str]:
    """
    Candidate identifiers useful for filtering (e.g., Step1 filter_wells_csv).
    """
    u = (w.url or "").strip()
    return [
        (w.api_num_nodash or "").strip(),
        (w.api or "").strip(),
        (w.kgs_id or "").strip(),
        u,
        (u.split("/")[-1] if u else ""),
    ]


def read_ks_manifest(path: Path) -> List[Dict[str, str]]:
    """
    Read data/ks_las_files.txt (quoted CSV) and return rows as dicts (empty-string fill).
    """
    return read_csv_rows(path)


def extract_well_locations(rows: List[Dict[str, str]]) -> List[WellRecord]:
    """
    Extract lon/lat + identifiers from manifest rows.
    Skips rows without usable coordinates.
    """
    out: List[WellRecord] = []

    for r in rows:
        url = (_get(r, "URL") or "").strip()
        kgs_id = (_get(r, "KGS_ID") or "").strip()
        api = (_get(r, "API") or "").strip()
        api_nd = api_nodash(api, (_get(r, "API_NUM_NODASH") or "").strip())
        operator = (_get(r, "Operator") or "").strip()
        lease = (_get(r, "Lease") or "").strip()

        lat = to_float((_get(r, "Latitude") or "").strip())
        lon = to_float((_get(r, "Longitude") or "").strip())
        if lat is None or lon is None:
            continue

        out.append(
            WellRecord(
                url=url,
                kgs_id=kgs_id,
                api=api,
                api_num_nodash=api_nd,
                operator=operator,
                lease=lease,
                lat=float(lat),
                lon=float(lon),
                row=r,
            )
        )

    return out
