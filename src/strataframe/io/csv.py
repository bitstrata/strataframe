# src/strataframe/io/csv.py
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


def _read_sample_text(path: Path, *, max_bytes: int = 32_000) -> str:
    """
    Read a small prefix of the file for delimiter sniffing, without loading the whole file.
    """
    with path.open("rb") as f:
        blob = f.read(max_bytes)
    # Handle UTF-8 BOM if present; decode defensively
    return blob.decode("utf-8-sig", errors="replace")


def _first_nonempty_line(sample: str) -> str:
    for line in sample.splitlines():
        s = line.strip()
        if s:
            return s
    return ""


def _detect_delimiter(sample: str) -> str:
    """
    Robust delimiter detection for common CSV/TSV-ish files.

    Strategy:
      1) Fast-path based on first non-empty line when unambiguous.
      2) Heuristic scoring across multiple lines (prefers consistent column counts).
      3) Fallback to csv.Sniffer with constrained delimiters.
    """
    first = _first_nonempty_line(sample)

    cands = [",", "\t", ";", "|"]

    # Fast paths (only when unambiguous)
    has = {d: (d in first) for d in cands}
    if has["\t"] and not (has[","] or has[";"] or has["|"]):
        return "\t"
    if has[","] and not (has["\t"] or has[";"] or has["|"]):
        return ","

    # Heuristic scoring across lines
    lines = [ln for ln in sample.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    lines = lines[:50]  # bounded work

    def _score(delim: str) -> float:
        # Score = consistency + magnitude of splits
        counts: List[int] = []
        for ln in lines:
            counts.append(ln.count(delim))
        if not counts:
            return 0.0
        mx = max(counts)
        if mx == 0:
            return 0.0
        mean = sum(counts) / len(counts)
        var = sum((c - mean) ** 2 for c in counts) / max(1, len(counts) - 1)
        # Higher is better: magnitude helps, variance hurts
        return float(mean) / (1.0 + float(var))

    scores = {d: _score(d) for d in cands}
    best = max(cands, key=lambda d: scores[d])
    if scores.get(best, 0.0) > 0.0:
        return best

    # Sniffer fallback (constrained)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=cands)
        return dialect.delimiter
    except Exception:
        return ","


def read_csv_rows(path: Path, *, delimiter: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Read a delimited file into list[dict[str,str]] with empty-string fill for missing values.
    If delimiter is None, detect it using a small file prefix.
    """
    if delimiter is None:
        sample = _read_sample_text(path, max_bytes=32_000)
        delimiter = _detect_delimiter(sample)

    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        rdr = csv.DictReader(f, delimiter=delimiter)
        if rdr.fieldnames is None:
            raise ValueError(f"No header row found in {path}")

        out: List[Dict[str, str]] = []
        for r in rdr:
            if r is None:
                continue
            # Normalize None -> "" so downstream code can do .get(...,"") safely.
            out.append({k: (v if v is not None else "") for k, v in r.items()})
        return out


def write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Dict[str, Any]],
    *,
    delimiter: str = ",",
) -> None:
    """
    Write rows (dict-like) to a delimited file using explicit fieldnames.
    Missing keys are written as empty strings.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def to_float(x: str) -> Optional[float]:
    """
    Best-effort float parsing:
      - strips whitespace
      - accepts common null spellings
      - tolerates thousands separators
    """
    s = (x or "").strip()
    if not s:
        return None
    s_up = s.upper()
    if s_up in {"NA", "N/A", "NULL", "NONE", "NAN"}:
        return None
    try:
        # tolerate "1,234.5"
        v = float(s.replace(",", ""))
        if v != v:  # NaN
            return None
        return v
    except Exception:
        return None
