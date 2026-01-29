# src/strataframe/graph/candidate_filters.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from strataframe.graph.las_utils import _parse_las_header_minimal


def url_to_local_path(url: str, las_root: Path) -> Optional[Path]:
    u = (url or "").strip()
    if not u:
        return None
    name = u.split("/")[-1].strip()
    if not name:
        return None
    name = name.split("?", 1)[0].split("#", 1)[0].strip()
    if not name:
        return None
    return Path(las_root) / name


def las_size_mb(p: Path) -> float:
    try:
        return float(p.stat().st_size) / (1024.0 * 1024.0)
    except Exception:
        return float("nan")


def prefilter_rows_for_las(
    rows: List[Dict[str, str]],
    *,
    las_root: Path,
    max_las_mb: int,
    max_curves: int = 0,
    allowed_names: Optional[set[str]],
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """
    Apply:
      (a) optional Step0 whitelist (LAS basename)
      (b) missing/oversize LAS prefilter
    Adds las_path field.
    """
    diag: Dict[str, Any] = {}

    if allowed_names is not None:
        kept: List[Dict[str, str]] = []
        dropped = 0
        for r in rows:
            url = (r.get("url") or "").strip()
            name = url.split("/")[-1].split("?", 1)[0].split("#", 1)[0].strip() if url else ""
            if name and name in allowed_names:
                kept.append(r)
            else:
                dropped += 1
        diag["step0_wells_gr_filter"] = {"dropped": int(dropped), "kept": int(len(kept))}
        rows = kept

    kept2: List[Dict[str, str]] = []
    skip_missing = 0
    skip_big = 0
    skip_many_curves = 0
    skip_bad_header = 0

    for r in rows:
        lp = url_to_local_path((r.get("url") or ""), las_root)
        if lp is None or (not lp.exists()):
            skip_missing += 1
            continue
        mb = las_size_mb(lp)
        if np.isfinite(mb) and float(mb) > float(max_las_mb):
            skip_big += 1
            continue
        if int(max_curves) > 0:
            try:
                hdr = _parse_las_header_minimal(lp)
                n_curves = int(len(hdr.get("curves", []) or []))
            except Exception:
                n_curves = 0
            if n_curves <= 0:
                skip_bad_header += 1
                continue
            if n_curves > int(max_curves):
                skip_many_curves += 1
                continue
        rr = dict(r)
        rr["las_path"] = str(lp)
        kept2.append(rr)

    diag["las_prefilter"] = {
        "max_las_mb": int(max_las_mb),
        "max_curves": int(max_curves),
        "skip_missing": int(skip_missing),
        "skip_big": int(skip_big),
        "skip_many_curves": int(skip_many_curves),
        "skip_bad_header": int(skip_bad_header),
        "kept": int(len(kept2)),
    }
    return kept2, diag
