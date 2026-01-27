# src/strataframe/preprocess/infer_curve_identity.py
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Canonical curve-header policy (single source of truth)
from strataframe.curves.normalize_header import (
    aliases_for,
    norm_mnemonic as canon_mnemonic,
    norm_unit as canon_unit,
)

# =============================================================================
# Models
# =============================================================================

@dataclass(frozen=True)
class CurveStats:
    n: int
    vmin: float
    p01: float
    p05: float
    p50: float
    p95: float
    p99: float
    vmax: float
    frac_zero: float
    frac_neg: float
    frac_nan: float


@dataclass(frozen=True)
class Signature:
    name: str
    canonical_mnemonic: str
    canonical_unit: str
    p50_range: Optional[Tuple[float, float]] = None
    p95_range: Optional[Tuple[float, float]] = None
    frac_neg_max: Optional[float] = None
    frac_zero_max: Optional[float] = None
    shape: Optional[str] = None


SIGNATURES: Dict[str, Signature] = {
    "GR": Signature(
        name="Gamma Ray",
        canonical_mnemonic="GR",
        canonical_unit="GAPI",
        p50_range=(0.0, 200.0),
        p95_range=(0.0, 400.0),
        frac_neg_max=0.02,
        frac_zero_max=0.35,
    ),
    "CALI": Signature(
        name="Caliper",
        canonical_mnemonic="CALI",
        canonical_unit="IN",
        p50_range=(4.0, 25.0),
        p95_range=(4.0, 40.0),
        frac_neg_max=0.01,
    ),
    "DCAL": Signature(
        name="Delta Caliper",
        canonical_mnemonic="DCAL",
        canonical_unit="IN",
        p50_range=(-5.0, 10.0),
        p95_range=(-10.0, 20.0),
    ),
    "MCAL": Signature(
        name="Micro Caliper",
        canonical_mnemonic="MCAL",
        canonical_unit="IN",
        p50_range=(4.0, 25.0),
        p95_range=(4.0, 40.0),
    ),
    "RHOB": Signature(
        name="Bulk Density",
        canonical_mnemonic="RHOB",
        canonical_unit="G/CC",
        p50_range=(1.6, 3.2),
        p95_range=(1.3, 3.5),
        frac_neg_max=0.001,
    ),
    "DT": Signature(
        name="Sonic (compressional)",
        canonical_mnemonic="DT",
        canonical_unit="US/FT",
        p50_range=(35.0, 180.0),
        p95_range=(20.0, 260.0),
        frac_neg_max=0.01,
    ),
    "RES": Signature(
        name="Resistivity-like",
        canonical_mnemonic="RES",
        canonical_unit="OHM-M",
        p50_range=(0.05, 5000.0),
        p95_range=(0.05, 50000.0),
        frac_neg_max=0.001,
        shape="log_skewed",
    ),
}

# Canonical-only unit families (normalize once; compare canon-to-canon everywhere)
_RESISTIVITY_UNITS_RAW = {"OHM-M", "OHM", "OHMM", "OHM.M", "OHM/M", "OHMS"}
_CONDUCTIVITY_UNITS_RAW = {"MMHO/M", "MMHO", "MMHOS/M", "MMHO-M"}

RESISTIVITY_UNITS_CANON = {u for u in (canon_unit(x) for x in _RESISTIVITY_UNITS_RAW) if u}
CONDUCTIVITY_UNITS_CANON = {u for u in (canon_unit(x) for x in _CONDUCTIVITY_UNITS_RAW) if u}


# =============================================================================
# CSV IO
# =============================================================================

def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        if rdr.fieldnames is None:
            raise ValueError(f"No header row found in {path}")
        out: List[Dict[str, str]] = []
        for r in rdr:
            out.append({k: (v if v is not None else "") for k, v in r.items()})
        return out


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


# =============================================================================
# LAS streaming reader (handles WRAP=YES; avoids lasio memory blowups)
# =============================================================================

_SECTION_RE = re.compile(r"^\s*~\s*([A-Z]+)", re.IGNORECASE)
_COMMENT_RE = re.compile(r"^\s*#")
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class LasMeta:
    wrap: bool
    null_value: Optional[float]
    curves: List[str]  # mnemonics in ~CURVE order


def _iter_text_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        for line in f:
            yield line.rstrip("\n")


def _clean_mnemonic(m: str) -> str:
    m = (m or "").strip().upper()
    if not m:
        return ""
    return _WS_RE.sub("", m)


def _parse_param_value(line: str) -> Tuple[str, str]:
    pre = line.split(":", 1)[0].strip()
    if not pre:
        return "", ""
    if "." not in pre:
        parts = pre.split()
        if not parts:
            return "", ""
        mn = _clean_mnemonic(parts[0])
        val = parts[1].strip() if len(parts) > 1 else ""
        return mn, val

    left, rest = pre.split(".", 1)
    mn = _clean_mnemonic(left)
    parts = rest.strip().split()
    val = parts[0].strip() if parts else ""
    return mn, val


def parse_las_meta(las_path: Path) -> LasMeta:
    wrap = False
    null_value: Optional[float] = None
    curves: List[str] = []

    in_version = False
    in_well = False
    in_curve = False

    for raw in _iter_text_lines(las_path):
        line = raw.strip()
        if not line or _COMMENT_RE.match(line):
            continue

        msec = _SECTION_RE.match(line)
        if msec:
            sec = (msec.group(1) or "").upper()
            in_version = sec.startswith("V")
            in_well = sec.startswith("W")
            in_curve = sec.startswith("C")
            if sec.startswith("A"):
                break
            continue

        if in_version:
            mn, val = _parse_param_value(line)
            if mn == "WRAP":
                wrap = val.strip().upper().startswith("Y")
            continue

        if in_well:
            mn, val = _parse_param_value(line)
            if mn == "NULL":
                try:
                    null_value = float(val)
                except Exception:
                    null_value = None
            continue

        if in_curve:
            # curve mnemonic is left of "."
            pre = line.split(":", 1)[0].strip()
            if not pre:
                continue
            if "." in pre:
                mn = _clean_mnemonic(pre.split(".", 1)[0])
            else:
                parts = pre.split()
                mn = _clean_mnemonic(parts[0]) if parts else ""
            if mn:
                curves.append(mn)
            continue

    return LasMeta(wrap=wrap, null_value=null_value, curves=curves)


def _to_float(tok: str) -> float:
    try:
        return float(tok)
    except Exception:
        return float("nan")


def find_curve_column(meta: LasMeta, curve_mnemonic: str) -> Optional[int]:
    """
    Resolve a curve column index robustly.

    Strategy (in order):
      1) exact raw token match (after local whitespace cleanup)
      2) canonical match (canon_mnemonic on both sides)
      3) canonical family alias match (aliases_for(canonical))
      4) prefix fallback (raw token startswith)
    """
    target_raw = _clean_mnemonic(curve_mnemonic)
    if not target_raw or not meta.curves:
        return None

    # Raw-cleaned curve tokens from ~CURVE
    raw_curves = [_clean_mnemonic(c) for c in meta.curves]

    # (1) exact raw token match
    for i, c in enumerate(raw_curves):
        if c == target_raw:
            return i

    # Canonical target
    target_can = canon_mnemonic(target_raw)
    if not target_can:
        # (4) raw prefix fallback only
        for i, c in enumerate(raw_curves):
            if c.startswith(target_raw):
                return i
        return None

    # Canonical curves
    can_curves = [canon_mnemonic(c) for c in raw_curves]

    # (2) exact canonical match
    idxs = [i for i, cc in enumerate(can_curves) if cc == target_can]
    if len(idxs) == 1:
        return idxs[0]

    # (3) canonical family alias match
    # aliases_for("GR") -> ["GR","GAMMA","GAMMA_RAY",...]
    try:
        fam = aliases_for(target_can)
        fam_can = {canon_mnemonic(a) for a in fam if a}
    except Exception:
        fam_can = {target_can}

    fam_hits = [i for i, cc in enumerate(can_curves) if cc in fam_can]
    if len(fam_hits) == 1:
        return fam_hits[0]
    if fam_hits:
        # If multiple, choose one that is "closest" to the raw target by token prefix
        for i in fam_hits:
            if raw_curves[i].startswith(target_raw):
                return i
        return fam_hits[0]

    # (4) raw prefix fallback (last resort)
    for i, c in enumerate(raw_curves):
        if c.startswith(target_raw):
            return i

    # If canonical matches existed but were ambiguous, pick first canonical match
    if idxs:
        return idxs[0]

    return None


def iter_las_column_values(
    las_path: Path,
    *,
    col_index: int,
    n_cols: int,
    wrap: bool,
    null_value: Optional[float],
) -> Iterable[float]:
    in_ascii = False
    buf: List[str] = []

    for raw in _iter_text_lines(las_path):
        line = raw.strip()
        if not line or _COMMENT_RE.match(line):
            continue

        msec = _SECTION_RE.match(line)
        if msec:
            sec = (msec.group(1) or "").upper()
            in_ascii = sec.startswith("A")
            continue

        if not in_ascii:
            continue

        toks = line.split()
        if not toks:
            continue

        if not wrap:
            if len(toks) < n_cols:
                continue
            v = _to_float(toks[col_index])
            if null_value is not None and np.isfinite(v) and v == float(null_value):
                v = float("nan")
            yield v
            continue

        buf.extend(toks)
        while len(buf) >= n_cols:
            row = buf[:n_cols]
            buf = buf[n_cols:]
            v = _to_float(row[col_index])
            if null_value is not None and np.isfinite(v) and v == float(null_value):
                v = float("nan")
            yield v


def compute_curve_stats_stream(
    values: Iterable[float],
    *,
    max_sample: int = 200_000,
    min_finite: int = 20,
) -> Optional[CurveStats]:
    n_total = 0
    n_finite = 0
    n_zero = 0
    n_neg = 0
    vmin = float("inf")
    vmax = float("-inf")

    mean = 0.0
    m2 = 0.0

    sample: List[float] = []
    rng = np.random.default_rng(12345)

    for v in values:
        n_total += 1
        if not np.isfinite(v):
            continue

        n_finite += 1
        if v == 0.0:
            n_zero += 1
        if v < 0.0:
            n_neg += 1

        if v < vmin:
            vmin = float(v)
        if v > vmax:
            vmax = float(v)

        delta = v - mean
        mean += delta / n_finite
        m2 += delta * (v - mean)

        if len(sample) < max_sample:
            sample.append(float(v))
        else:
            j = int(rng.integers(0, n_finite))
            if j < max_sample:
                sample[j] = float(v)

    if n_total == 0 or n_finite < min_finite:
        return None

    var = m2 / max(1, (n_finite - 1))
    if not np.isfinite(var) or var <= 0.0:
        return None

    arr = np.asarray(sample, dtype="float64")
    if arr.size < min_finite:
        return None

    p01, p05, p50, p95, p99 = np.percentile(arr, [1, 5, 50, 95, 99])

    frac_nan = float(np.clip(1.0 - (n_finite / n_total), 0.0, 1.0))
    frac_zero = float(np.clip(n_zero / n_finite, 0.0, 1.0))
    frac_neg = float(np.clip(n_neg / n_finite, 0.0, 1.0))

    return CurveStats(
        n=int(n_finite),
        vmin=float(vmin),
        p01=float(p01),
        p05=float(p05),
        p50=float(p50),
        p95=float(p95),
        p99=float(p99),
        vmax=float(vmax),
        frac_zero=frac_zero,
        frac_neg=frac_neg,
        frac_nan=frac_nan,
    )


def load_las_curve_stats(las_path: Path, curve_mnemonic: str) -> Optional[CurveStats]:
    if not las_path.exists():
        return None
    try:
        meta = parse_las_meta(las_path)
    except Exception:
        return None
    if not meta.curves:
        return None
    idx = find_curve_column(meta, curve_mnemonic)
    if idx is None:
        return None
    n_cols = len(meta.curves)
    try:
        it = iter_las_column_values(
            las_path,
            col_index=idx,
            n_cols=n_cols,
            wrap=meta.wrap,
            null_value=meta.null_value,
        )
        return compute_curve_stats_stream(it)
    except Exception:
        return None


# =============================================================================
# Scoring + inference
# =============================================================================

def _range_score(x: float, lo: float, hi: float) -> float:
    if lo > hi:
        lo, hi = hi, lo
    if lo <= x <= hi:
        return 1.0
    width = max(hi - lo, 1e-9)
    margin = max(0.2 * width, 1.0)
    d = (lo - x) / margin if x < lo else (x - hi) / margin
    return float(math.exp(-d))


def _cap_penalty(frac: float, max_allowed: float) -> float:
    if max_allowed <= 0:
        return 1.0
    if frac <= max_allowed:
        return 1.0
    r = (frac - max_allowed) / max_allowed
    return float(math.exp(-2.0 * r))


def score_against_signature(stats: CurveStats, sig: Signature) -> float:
    for x in (stats.p50, stats.p95, stats.p99, stats.frac_neg, stats.frac_zero):
        if not np.isfinite(float(x)):
            return 0.0

    score = 0.0
    if sig.p50_range is not None:
        score += 0.45 * _range_score(stats.p50, sig.p50_range[0], sig.p50_range[1])
    else:
        score += 0.45 * 0.5

    if sig.p95_range is not None:
        score += 0.35 * _range_score(stats.p95, sig.p95_range[0], sig.p95_range[1])
    else:
        score += 0.35 * 0.5

    mult = 1.0
    if sig.frac_neg_max is not None:
        mult *= _cap_penalty(stats.frac_neg, sig.frac_neg_max)
    if sig.frac_zero_max is not None:
        mult *= _cap_penalty(stats.frac_zero, sig.frac_zero_max)

    if sig.shape == "log_skewed":
        p50 = float(stats.p50)
        p99 = float(stats.p99)
        if p50 > 0.0 and p99 > 0.0:
            ratio = (p99 + 1e-9) / (p50 + 1e-9)
            if ratio > 0.0:
                bonus = min(0.10, max(0.0, (math.log10(ratio) - 0.5) / 2.0 * 0.10))
                score += bonus

    score *= mult
    return float(max(0.0, min(1.0, score)))


def score_unit_match(unit_n: str, sig: Signature) -> float:
    u = canon_unit(unit_n)
    if not u:
        return 0.6  # unknown unit: allow, but not perfect
    return 1.0 if u == sig.canonical_unit else 0.2


def infer_from_unit(unit_n: str) -> List[Signature]:
    u = canon_unit(unit_n)
    cands: List[Signature] = []

    if u in {"GAPI", "CPS", "CPM"}:
        cands.append(SIGNATURES["GR"])

    if u in {"IN", "MM"}:
        cands.extend([SIGNATURES["CALI"], SIGNATURES["DCAL"], SIGNATURES["MCAL"]])

    if u in {"G/CC", "KG/M3"}:
        cands.append(SIGNATURES["RHOB"])

    if u in {"US/FT", "US"}:
        cands.append(SIGNATURES["DT"])

    # Resistivity / conductivity families (canonical)
    if u in RESISTIVITY_UNITS_CANON:
        cands.append(SIGNATURES["RES"])
    if u in CONDUCTIVITY_UNITS_CANON:
        cands.append(SIGNATURES["RES"])

    return cands


def infer_from_values(stats: CurveStats) -> List[Tuple[Signature, float]]:
    scored: List[Tuple[Signature, float]] = []
    for sig in SIGNATURES.values():
        scored.append((sig, score_against_signature(stats, sig)))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored


_AUTO_THRESHOLD = 0.85
_SUGGEST_THRESHOLD = 0.60


def decide(
    *,
    mnemonic_n: str,
    unit_n: str,
    stats: Optional[CurveStats],
) -> Tuple[str, str, float, str, str]:
    if stats is None:
        return mnemonic_n, unit_n, 0.0, "no_stats", "NO_ACTION"

    score_mn = 0.0
    mn_sig: Optional[Signature] = None
    if mnemonic_n in {"GR", "CALI", "DCAL", "MCAL", "RHOB", "DT"}:
        mn_sig = SIGNATURES.get(mnemonic_n)

    if mn_sig is not None:
        score_mn = 0.75 * score_against_signature(stats, mn_sig) + 0.25 * score_unit_match(unit_n, mn_sig)

    unit_cands = infer_from_unit(unit_n)
    score_unit = 0.0
    best_unit_sig: Optional[Signature] = None
    if unit_cands:
        best_sig: Optional[Signature] = None
        best_s = -1.0
        for sig in unit_cands:
            s = 0.80 * score_against_signature(stats, sig) + 0.20 * score_unit_match(unit_n, sig)
            if s > best_s:
                best_sig, best_s = sig, s
        best_unit_sig, score_unit = best_sig, float(best_s)

    ranked = infer_from_values(stats)
    best_val_sig, score_val = ranked[0][0], float(ranked[0][1])

    best_basis = "values_only"
    best_sig = best_val_sig
    best_score = score_val

    if best_unit_sig is not None and score_unit > best_score + 0.05:
        best_basis = "unit_values"
        best_sig = best_unit_sig
        best_score = score_unit

    if mn_sig is not None and score_mn >= best_score - 0.02:
        best_basis = "mnemonic_values"
        best_sig = mn_sig
        best_score = max(best_score, score_mn)

    inferred_mn = best_sig.canonical_mnemonic
    inferred_un = best_sig.canonical_unit
    conf = float(max(0.0, min(1.0, best_score)))

    if conf >= _AUTO_THRESHOLD:
        decision = "AUTO_FIX"
    elif conf >= _SUGGEST_THRESHOLD:
        decision = "SUGGEST"
    else:
        decision = "NO_ACTION"

    return inferred_mn, inferred_un, conf, best_basis, decision


# =============================================================================
# Mapping outputs (tiny per-file)
# =============================================================================

def _resolve_las_path(url: str, file_: str, las_root: Path) -> Optional[Path]:
    if file_:
        p = Path(file_)
        if p.exists():
            return p
        p2 = Path.cwd() / p
        if p2.exists():
            return p2
    if url:
        basename = url.strip().split("/")[-1]
        p = las_root / basename
        if p.exists():
            return p
    return None


def _file_key(las_path: Path) -> str:
    # Stable-ish key: basename + short hash of absolute path
    ab = str(las_path.resolve())
    h = hashlib.sha1(ab.encode("utf-8")).hexdigest()[:10]
    base = las_path.name
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", base)
    return f"{safe}__{h}"


@dataclass
class MapEntry:
    mnemonic: str
    unit: str
    inferred_mnemonic: str
    inferred_unit: str
    confidence: float
    basis: str
    decision: str  # AUTO_FIX | SUGGEST


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Infer likely true curve mnemonic/unit for quarantined LAS headers "
            "by scoring value distributions against soft signatures. "
            "Writes tiny per-file mapping CSVs instead of a huge row-wise suggestions file."
        )
    )
    ap.add_argument("--quarantine-csv", type=str, required=True, help="Path to las_quarantine_report.csv")
    ap.add_argument("--las-root", type=str, default="data/las", help="Fallback LAS root if 'file' column absent/unresolvable")
    ap.add_argument("--out-dir", type=str, required=True, help="Output directory")
    ap.add_argument("--only-severity", type=str, default="ERROR,WARN", help="Comma-separated severities to process")
    ap.add_argument("--max-rows", type=int, default=0, help="Debug: process at most N rows (0 = no limit)")
    ap.add_argument("--auto-threshold", type=float, default=0.85, help="AUTO_FIX threshold (default 0.85)")
    ap.add_argument("--suggest-threshold", type=float, default=0.60, help="SUGGEST threshold (default 0.60)")
    ap.add_argument("--include-suggest", action="store_true", help="Include SUGGEST entries in per-file maps (default: AUTO_FIX only)")
    ap.add_argument("--write-autofix-global", action="store_true", help="Also write a small aggregated curve_identity_autofix.csv")
    args = ap.parse_args(argv)

    global _AUTO_THRESHOLD, _SUGGEST_THRESHOLD
    _AUTO_THRESHOLD = float(args.auto_threshold)
    _SUGGEST_THRESHOLD = float(args.suggest_threshold)

    qrows = read_csv_rows(Path(args.quarantine_csv))
    las_root = Path(args.las_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    maps_dir = out_dir / "curve_identity_maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    only = {s.strip().upper() for s in (args.only_severity or "").split(",") if s.strip()}
    max_rows = None if args.max_rows <= 0 else int(args.max_rows)

    # Cache stats per (las_path, mnemonic_raw_upper) to avoid repeat reads
    stats_cache: Dict[Tuple[str, str], Optional[CurveStats]] = {}

    # Per-file maps, deduped by (mnemonic_raw, unit_raw) keeping best confidence
    per_file: Dict[str, Dict[Tuple[str, str], MapEntry]] = {}
    per_file_meta: Dict[str, Dict[str, str]] = {}  # some context for index (url/file)

    n_processed = 0
    n_auto = 0
    n_sug = 0
    n_no = 0

    autofix_global_rows: List[Dict[str, Any]] = []

    for i, r in enumerate(qrows):
        if max_rows is not None and i >= max_rows:
            break

        sev = (r.get("severity", "") or "").upper()
        if only and sev not in only:
            continue

        url = r.get("url", "") or ""
        file_ = r.get("file", "") or ""

        mnemonic_raw = r.get("mnemonic", "") or ""
        unit_raw = r.get("unit", "") or ""
        mnemonic_n = r.get("mnemonic_n", "") or ""
        unit_n = r.get("unit_n", "") or ""

        las_path = _resolve_las_path(url, file_, las_root)
        stats: Optional[CurveStats] = None

        if las_path is not None:
            ck = (str(las_path.resolve()), (mnemonic_raw or "").strip().upper())
            if ck in stats_cache:
                stats = stats_cache[ck]
            else:
                stats = load_las_curve_stats(las_path, mnemonic_raw)
                stats_cache[ck] = stats

        inferred_mn, inferred_un, conf, basis, decision = decide(
            mnemonic_n=mnemonic_n,
            unit_n=unit_n,
            stats=stats,
        )

        n_processed += 1
        if decision == "AUTO_FIX":
            n_auto += 1
        elif decision == "SUGGEST":
            n_sug += 1
        else:
            n_no += 1

        # Keep outputs tiny: write only AUTO_FIX, optionally SUGGEST
        if decision not in {"AUTO_FIX", "SUGGEST"}:
            continue
        if decision == "SUGGEST" and not bool(args.include_suggest):
            continue
        if las_path is None:
            continue

        fk = _file_key(las_path)
        per_file.setdefault(fk, {})
        per_file_meta.setdefault(fk, {"las_path": str(las_path), "url": url, "file_col": file_})

        k = ((mnemonic_raw or "").strip(), (unit_raw or "").strip())
        entry = MapEntry(
            mnemonic=k[0],
            unit=k[1],
            inferred_mnemonic=inferred_mn,
            inferred_unit=inferred_un,
            confidence=float(conf),
            basis=basis,
            decision=decision,
        )

        # Dedup: keep best confidence for this (mnemonic,unit)
        cur = per_file[fk].get(k)
        if cur is None or entry.confidence > cur.confidence:
            per_file[fk][k] = entry

        if args.write_autofix_global and decision == "AUTO_FIX":
            autofix_global_rows.append(
                {
                    "las_path": str(las_path),
                    "mnemonic": k[0],
                    "unit": k[1],
                    "inferred_mnemonic": inferred_mn,
                    "inferred_unit": inferred_un,
                    "confidence": f"{conf:.3f}",
                    "basis": basis,
                }
            )

    # Write per-file maps + index
    index_rows: List[Dict[str, Any]] = []
    for fk, entries in sorted(per_file.items()):
        las_path = per_file_meta.get(fk, {}).get("las_path", "")
        url = per_file_meta.get(fk, {}).get("url", "")
        map_path = maps_dir / f"{fk}.csv"

        rows = []
        n_auto_f = 0
        n_sug_f = 0
        for (_, _), e in sorted(entries.items(), key=lambda t: (-t[1].confidence, t[1].mnemonic, t[1].unit)):
            if e.decision == "AUTO_FIX":
                n_auto_f += 1
            elif e.decision == "SUGGEST":
                n_sug_f += 1
            rows.append(
                {
                    "mnemonic": e.mnemonic,
                    "unit": e.unit,
                    "inferred_mnemonic": e.inferred_mnemonic,
                    "inferred_unit": e.inferred_unit,
                    "confidence": f"{e.confidence:.3f}",
                    "basis": e.basis,
                    "decision": e.decision,
                }
            )

        write_csv(
            map_path,
            ["mnemonic", "unit", "inferred_mnemonic", "inferred_unit", "confidence", "basis", "decision"],
            rows,
        )

        index_rows.append(
            {
                "file_key": fk,
                "las_path": las_path,
                "url": url,
                "map_csv": str(map_path),
                "n_entries": len(rows),
                "n_auto_fix": n_auto_f,
                "n_suggest": n_sug_f,
            }
        )

    write_csv(
        out_dir / "curve_identity_map_index.csv",
        ["file_key", "las_path", "url", "map_csv", "n_entries", "n_auto_fix", "n_suggest"],
        index_rows,
    )

    if args.write_autofix_global:
        write_csv(
            out_dir / "curve_identity_autofix.csv",
            ["las_path", "mnemonic", "unit", "inferred_mnemonic", "inferred_unit", "confidence", "basis"],
            autofix_global_rows,
        )

    print(f"Rows processed: {n_processed}")
    print(f"AUTO_FIX: {n_auto}")
    print(f"SUGGEST:  {n_sug}")
    print(f"NO_ACTION:{n_no}")
    print(f"Wrote per-file maps to: {maps_dir}")
    print(f"Wrote index: {out_dir / 'curve_identity_map_index.csv'}")
    if args.write_autofix_global:
        print(f"Wrote global autofix: {out_dir / 'curve_identity_autofix.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
