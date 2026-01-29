# src/strataframe/graph/las_utils.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Optional lasio (preferred for LAS). If absent, we fail with a clear message.
try:
    import lasio  # type: ignore
except Exception:  # pragma: no cover
    lasio = None  # type: ignore

# Optional librosa for DTW
try:
    import librosa  # type: ignore
except Exception:  # pragma: no cover
    librosa = None  # type: ignore


# =============================================================================
# Canonical header normalization (shared owner in strataframe.curves.*)
# =============================================================================
try:
    from strataframe.curves.normalize_header import norm_mnemonic as _norm_mnemonic  # type: ignore
except Exception:  # pragma: no cover

    def _norm_mnemonic(m: Optional[str]) -> str:  # type: ignore
        return (m or "").strip().upper()


def normalize_mnemonic(m: str) -> str:
    """
    Backward-compatible wrapper returning the CANONICAL mnemonic.
    """
    return _norm_mnemonic(m)


# =============================================================================
# Minimal header parsing (text only; safe on wrapped files)
# =============================================================================

def _parse_las_header_minimal(path: Path, *, max_header_bytes: int = 2_000_000) -> dict:
    """
    Text-only LAS header parse:
      - WRAP (YES/NO)
      - NULL value (float or None)
      - curve mnemonics in order from ~C
    Stops once ~A is reached. Constant memory. Works on wrapped/unwrapped because it
    never parses ~A.
    """
    wrap = False
    null_value: Optional[float] = None
    curves: List[str] = []

    section = ""
    n_read = 0

    def _is_section(line: str) -> bool:
        s = line.lstrip()
        return s.startswith("~") and len(s) >= 2

    def _section_name(line: str) -> str:
        s = line.lstrip()
        t = s[1:].strip()  # "~C" / "~Curve" etc
        return t[:1].upper() if t else ""

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            n_read += len(line.encode("utf-8", errors="ignore"))
            if n_read > max_header_bytes:
                break

            s = line.strip()
            if not s or s.startswith("#"):
                continue

            if _is_section(line):
                sec = _section_name(line)
                section = sec
                if sec == "A":
                    break
                continue

            if section == "W":
                left = s.split(":", 1)[0].strip()
                if "." in left:
                    key, rest = left.split(".", 1)
                    key_n = key.strip().upper()
                    val = rest.strip().split()[0] if rest.strip() else ""
                    if key_n == "WRAP":
                        wrap = val.upper().startswith("Y")
                    elif key_n == "NULL":
                        try:
                            null_value = float(val)
                        except Exception:
                            null_value = None

            elif section == "C":
                left = s.split(":", 1)[0].strip()
                if "." in left:
                    key = left.split(".", 1)[0].strip()
                else:
                    key = left.split()[0].strip()
                if key and not key.startswith("~"):
                    curves.append(key)

    return {"wrap": wrap, "null": null_value, "curves": curves}


def read_las_header_only(path: Path) -> dict:
    """
    Replacement for any lasio-based header reader. Returns a small dict.
    """
    return _parse_las_header_minimal(Path(path))


# =============================================================================
# Exceptions + requirements
# =============================================================================

class LasReadError(RuntimeError):
    """Raised when a LAS file cannot be read in the requested mode."""


def require_lasio() -> None:
    if lasio is None:
        raise RuntimeError(
            "lasio is required. Install it with:\n"
            "  pip install lasio\n"
        )


def require_librosa() -> None:
    if librosa is None:
        raise RuntimeError(
            "librosa is required for DTW. Install it with:\n"
            "  pip install librosa\n"
        )


# =============================================================================
# LAS read helpers (deduplicated; wrapped-file tolerant)
# =============================================================================

_LASIO_SUPPORTS_ENGINE: Optional[bool] = None


def _lasio_supports_engine_kw() -> bool:
    """
    Detect whether lasio.read supports engine= kwarg (varies by lasio version).
    Cached to avoid repeated inspect overhead.
    """
    global _LASIO_SUPPORTS_ENGINE
    if _LASIO_SUPPORTS_ENGINE is not None:
        return bool(_LASIO_SUPPORTS_ENGINE)

    require_lasio()
    try:
        import inspect

        sig = inspect.signature(lasio.read)  # type: ignore[attr-defined]
        _LASIO_SUPPORTS_ENGINE = bool("engine" in sig.parameters)
    except Exception:
        _LASIO_SUPPORTS_ENGINE = False
    return bool(_LASIO_SUPPORTS_ENGINE)


def _read_lasio(
    las_path: Path,
    *,
    ignore_data: bool,
) -> object:
    """
    Read LAS using lasio with best-effort wrapped-file handling.

    Strategy:
      - Prefer engine='normal' when supported (robust for wrapped LAS).
      - Fall back to no engine kw when not supported.
      - If wrapped LAS fails on older lasio, raise a clear message.
    """
    require_lasio()
    p = Path(las_path)

    if not p.exists():
        raise FileNotFoundError(p)

    # Prefer engine='normal' if available
    if _lasio_supports_engine_kw():
        try:
            return lasio.read(str(p), ignore_data=bool(ignore_data), engine="normal")  # type: ignore[attr-defined]
        except Exception as e:
            # If engine path fails, try without engine once (some edge-case builds)
            try:
                return lasio.read(str(p), ignore_data=bool(ignore_data))  # type: ignore[attr-defined]
            except Exception as e2:
                msg = (str(e2) or str(e)).lower()
                if "wrapped" in msg:
                    raise LasReadError(
                        "LAS appears wrapped but this lasio build cannot read it reliably. "
                        "Upgrade lasio (recommended) or pre-unwind wrapped LAS files."
                    ) from e2
                raise LasReadError(f"Failed to read LAS: {p}") from e2

    # Older lasio: no engine kw
    try:
        return lasio.read(str(p), ignore_data=bool(ignore_data))  # type: ignore[attr-defined]
    except Exception as e:
        msg = str(e).lower()
        if "wrapped" in msg:
            raise LasReadError(
                "LAS appears wrapped but this lasio version cannot force engine='normal'. "
                "Upgrade lasio (recommended) or pre-unwind wrapped LAS files."
            ) from e
        raise LasReadError(f"Failed to read LAS: {p}") from e


def read_las_normal(las_path: Path):
    """
    Read full LAS (data + headers), wrapped-file tolerant.
    """
    return _read_lasio(Path(las_path), ignore_data=False)


# =============================================================================
# Curve discovery + resolution
# =============================================================================

def list_curve_mnemonics(las: Any) -> List[str]:
    """
    Returns RAW curve mnemonics as they appear in the LAS (trimmed).
    These raw names are what you can safely use to index las[raw].

    Supports:
      - dict objects returned by read_las_header_only()
      - lasio LASFile objects
    """
    if isinstance(las, dict):
        c = las.get("curves", [])
        return [str(x).strip() for x in (c or []) if str(x).strip()]

    out: List[str] = []
    for c in getattr(las, "curves", []) or []:
        try:
            raw = (getattr(c, "mnemonic", "") or "").strip()
            if raw:
                out.append(raw)
        except Exception:
            continue
    return out


def _find_curve_index(
    curves: Sequence[str],
    preferred: Sequence[str],
    *,
    prefer_exact_raw: bool = True,
    prefer_curve_order: bool = False,
) -> Optional[int]:
    """
    Return index into curves for the first preferred mnemonic match.

    Matching strategy:
      1) Exact (case-insensitive) raw mnemonic match
      2) Canonical match via normalize_mnemonic()
    """
    if not curves:
        return None
    if not preferred:
        return None

    curves_u = [str(c or "").strip().upper() for c in curves]
    # Exact raw match first (optional)
    if bool(prefer_exact_raw):
        for p in preferred:
            pu = str(p or "").strip().upper()
            if not pu:
                continue
            for i, cu in enumerate(curves_u):
                if cu == pu:
                    return int(i)

    # Canonical match fallback
    pref_norm = [normalize_mnemonic(p) for p in preferred if str(p or "").strip()]
    if bool(prefer_curve_order):
        pref_set = {p for p in pref_norm if p}
        for i, raw in enumerate(curves):
            if normalize_mnemonic(raw) in pref_set:
                return int(i)
    else:
        for p in pref_norm:
            for i, raw in enumerate(curves):
                if normalize_mnemonic(raw) == p:
                    return int(i)
    return None


def _iter_las_ascii_rows(
    path: Path,
    *,
    n_curves: int,
    wrapped: bool,
) -> "Iterable[List[str]]":
    """
    Yield tokenized rows from the ~A section of a LAS file.

    - For wrapped files, tokens are buffered and emitted in groups of n_curves.
    - For unwrapped files, each non-empty, non-comment line is yielded as-is.
    """
    in_data = False
    token_buf: List[str] = []

    with Path(path).open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not in_data:
                if line.lstrip().upper().startswith("~A"):
                    in_data = True
                continue

            s = line.strip()
            if not s or s.startswith("#"):
                continue

            # Make splitting robust: treat commas as whitespace
            s = s.replace(",", " ")
            parts = s.split()
            if not parts:
                continue

            if wrapped:
                token_buf.extend(parts)
                while len(token_buf) >= int(n_curves):
                    row = token_buf[:n_curves]
                    del token_buf[:n_curves]
                    yield row
            else:
                yield parts


def _choose_curve_index(
    curves: Sequence[str],
    *,
    primary: Sequence[str],
    fallback: Sequence[str] = (),
    prefer_curve_order: bool = True,
) -> Optional[int]:
    """
    Choose a curve index with a simple priority:
      1) First match from `primary` (canonical match, header order optional)
      2) If none, first match from `fallback`
    """
    idx = _find_curve_index(
        curves,
        primary,
        prefer_exact_raw=False,
        prefer_curve_order=bool(prefer_curve_order),
    )
    if idx is not None:
        return idx
    if fallback:
        return _find_curve_index(
            curves,
            fallback,
            prefer_exact_raw=False,
            prefer_curve_order=bool(prefer_curve_order),
        )
    return None


def read_las_depth_and_curve_ascii(
    path: Path,
    *,
    curve_candidates: Sequence[str] = ("GR",),
    depth_preferred: Sequence[str] = ("DEPT", "DEPTH", "MD", "TVD", "Z"),
    max_rows: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Streaming ASCII reader for depth + one curve from ~A (wrapped or unwrapped).

    Uses header-only parse to locate curve indices, then scans ~A without lasio.
    Returns (depth, curve) as float64 arrays.
    """
    p = Path(path)
    hdr = _parse_las_header_minimal(p)
    curves = [str(c or "").strip() for c in (hdr.get("curves", []) or []) if str(c or "").strip()]
    if not curves:
        raise RuntimeError("LAS header has no curve mnemonics (~C not found).")

    depth_idx = _choose_curve_index(curves, primary=depth_preferred)
    if depth_idx is None:
        raise RuntimeError(f"Depth curve not found. Preferred={list(depth_preferred)}")

    curve_idx = _choose_curve_index(curves, primary=("GR",), fallback=curve_candidates)
    if curve_idx is None:
        raise RuntimeError(f"Target curve not found. Candidates={list(curve_candidates)}")

    n_curves = int(len(curves))
    wrapped = bool(hdr.get("wrap", False))
    null_value = hdr.get("null", None)

    depth: List[float] = []
    curve: List[float] = []

    max_idx = max(int(depth_idx), int(curve_idx))
    def _to_float(tok: str) -> float:
        try:
            return float(tok)
        except Exception:
            return float("nan")

    def _process_row(tokens: Sequence[str]) -> None:
        if len(tokens) <= max_idx:
            return
        d = _to_float(tokens[depth_idx])
        v = _to_float(tokens[curve_idx])

        if null_value is not None:
            try:
                nv = float(null_value)
            except Exception:
                nv = None
            if nv is not None:
                if np.isfinite(d) and d == nv:
                    d = float("nan")
                if np.isfinite(v) and v == nv:
                    v = float("nan")

        depth.append(float(d))
        curve.append(float(v))

    n_rows = 0
    for row in _iter_las_ascii_rows(p, n_curves=n_curves, wrapped=wrapped):
        n_rows += 1
        if int(max_rows) > 0 and n_rows > int(max_rows):
            raise RuntimeError(f"LAS exceeds max_rows={int(max_rows)}")
        _process_row(row)

    if not depth:
        raise RuntimeError("No data rows parsed from ~A section.")

    return np.asarray(depth, dtype="float64"), np.asarray(curve, dtype="float64")


def read_las_curve_resampled_ascii(
    path: Path,
    *,
    n_samples: int,
    curve_candidates: Sequence[str] = ("GR",),
    depth_preferred: Sequence[str] = ("DEPT", "DEPTH", "MD", "TVD", "Z"),
    p_lo: float = 1.0,
    p_hi: float = 99.0,
    min_finite: int = 10,
    max_rows: int = 0,
    step_sample_size: int = 10000,
    window_min: Optional[float] = None,
    window_max: Optional[float] = None,
) -> Tuple[np.ndarray, float, float, int, float]:
    """
    Streaming ASCII resampler for depth + one curve from ~A (wrapped or unwrapped).

    Returns:
      x_norm (n_samples,), z_top, z_base, n_finite_raw, sample_step
    """
    n_samples_i = int(n_samples)
    if n_samples_i < 8:
        raise ValueError("n_samples too small; use >= 8")
    if not (0.0 <= float(p_lo) < float(p_hi) <= 100.0):
        raise ValueError("Percentiles must satisfy 0 <= p_lo < p_hi <= 100")

    p = Path(path)
    hdr = _parse_las_header_minimal(p)
    curves = [str(c or "").strip() for c in (hdr.get("curves", []) or []) if str(c or "").strip()]
    if not curves:
        raise RuntimeError("LAS header has no curve mnemonics (~C not found).")

    depth_idx = _choose_curve_index(curves, primary=depth_preferred)
    if depth_idx is None:
        raise RuntimeError(f"Depth curve not found. Preferred={list(depth_preferred)}")

    curve_idx = _choose_curve_index(curves, primary=("GR",), fallback=curve_candidates)
    if curve_idx is None:
        raise RuntimeError(f"Target curve not found. Candidates={list(curve_candidates)}")

    n_curves = int(len(curves))
    wrapped = bool(hdr.get("wrap", False))
    null_value = hdr.get("null", None)
    null_f: Optional[float] = None
    if null_value is not None:
        try:
            null_f = float(null_value)
        except Exception:
            null_f = None

    def _to_float(tok: str) -> float:
        try:
            return float(tok)
        except Exception:
            return float("nan")

    if window_min is not None and window_max is not None:
        if float(window_max) <= float(window_min):
            raise ValueError("window_max must be greater than window_min")

    # Pass 1: depth range + step sampling
    z_top = float("inf")
    z_base = float("-inf")
    diffs: List[float] = []
    prev_d: Optional[float] = None
    n_rows = 0

    for row in _iter_las_ascii_rows(p, n_curves=n_curves, wrapped=wrapped):
        n_rows += 1
        if int(max_rows) > 0 and n_rows > int(max_rows):
            raise RuntimeError(f"LAS exceeds max_rows={int(max_rows)}")
        if len(row) <= max(depth_idx, curve_idx):
            continue
        d = _to_float(row[depth_idx])
        if null_f is not None and np.isfinite(d) and d == null_f:
            d = float("nan")
        if not np.isfinite(d):
            continue
        if window_min is not None and d < float(window_min):
            continue
        if window_max is not None and d > float(window_max):
            continue
        if d < z_top:
            z_top = d
        if d > z_base:
            z_base = d
        if prev_d is not None:
            dd = float(d - prev_d)
            if np.isfinite(dd) and dd != 0.0:
                if len(diffs) < int(max(1, step_sample_size)):
                    diffs.append(float(dd))
        prev_d = d

    if not np.isfinite(z_top) or not np.isfinite(z_base) or z_base <= z_top:
        raise RuntimeError("Invalid depth range (no data in window).")

    sample_step = float(np.nanmedian(diffs)) if diffs else float("nan")

    # Pass 2: streaming interpolation onto uniform grid
    z = np.linspace(z_top, z_base, n_samples_i, dtype="float64")
    x = np.full((n_samples_i,), np.nan, dtype="float64")

    i = 0
    prev_depth: Optional[float] = None
    prev_val: Optional[float] = None
    cur_depth: Optional[float] = None
    cur_sum = 0.0
    cur_count = 0
    n_finite_raw = 0

    def _emit_point(d: float, v: float) -> None:
        nonlocal i, prev_depth, prev_val
        if prev_depth is None:
            prev_depth = float(d)
            prev_val = float(v)
            # fill any grid points before first depth
            while i < n_samples_i and z[i] < prev_depth:
                x[i] = float(prev_val)
                i += 1
            return

        d0 = float(prev_depth)
        v0 = float(prev_val) if prev_val is not None else float(v)
        d1 = float(d)
        v1 = float(v)
        if d1 <= d0:
            # non-increasing depth; skip segment
            prev_depth = d1
            prev_val = v1
            return

        while i < n_samples_i and z[i] <= d1:
            t = (float(z[i]) - d0) / (d1 - d0)
            x[i] = v0 + t * (v1 - v0)
            i += 1

        prev_depth = d1
        prev_val = v1

    n_rows = 0
    for row in _iter_las_ascii_rows(p, n_curves=n_curves, wrapped=wrapped):
        n_rows += 1
        if int(max_rows) > 0 and n_rows > int(max_rows):
            raise RuntimeError(f"LAS exceeds max_rows={int(max_rows)}")
        if len(row) <= max(depth_idx, curve_idx):
            continue
        d = _to_float(row[depth_idx])
        v = _to_float(row[curve_idx])
        if null_f is not None:
            if np.isfinite(d) and d == null_f:
                d = float("nan")
            if np.isfinite(v) and v == null_f:
                v = float("nan")
        if not np.isfinite(d) or not np.isfinite(v):
            continue
        if window_min is not None and d < float(window_min):
            continue
        if window_max is not None and d > float(window_max):
            continue
        n_finite_raw += 1
        if cur_depth is None:
            cur_depth = float(d)
            cur_sum = float(v)
            cur_count = 1
            continue
        if d == cur_depth:
            cur_sum += float(v)
            cur_count += 1
            continue

        # finalize previous depth point and emit
        _emit_point(float(cur_depth), float(cur_sum) / float(max(1, cur_count)))
        cur_depth = float(d)
        cur_sum = float(v)
        cur_count = 1

    if cur_depth is not None and cur_count > 0:
        _emit_point(float(cur_depth), float(cur_sum) / float(max(1, cur_count)))

    # Fill any remaining grid points with last value
    if prev_val is not None:
        while i < n_samples_i:
            x[i] = float(prev_val)
            i += 1

    if int(n_finite_raw) < int(min_finite):
        raise RuntimeError("Too few finite curve samples to resample.")

    # If still NaNs, fallback to zeros to keep downstream stable
    if not np.any(np.isfinite(x)):
        return np.zeros((n_samples_i,), dtype="float64"), float(z_top), float(z_base), int(n_finite_raw), float(sample_step)

    # Normalize to [0,1] using robust percentiles
    fin = np.isfinite(x)
    plo = float(np.percentile(x[fin], float(p_lo)))
    phi = float(np.percentile(x[fin], float(p_hi)))
    if (not np.isfinite(plo)) or (not np.isfinite(phi)) or (phi <= plo):
        plo = float(np.nanmin(x[fin]))
        phi = float(np.nanmax(x[fin]))
        if (not np.isfinite(plo)) or (not np.isfinite(phi)) or (phi <= plo):
            return np.zeros((n_samples_i,), dtype="float64"), float(z_top), float(z_base), int(n_finite_raw), float(sample_step)

    x_norm = (x - plo) / (phi - plo)
    x_norm = np.clip(x_norm, 0.0, 1.0)
    return x_norm.astype("float64", copy=False), float(z_top), float(z_base), int(n_finite_raw), float(sample_step)


def available_canonical_curves(las) -> Dict[str, List[str]]:
    """
    Map canonical_mnemonic -> list of raw mnemonics present in this LAS.
    """
    m: Dict[str, List[str]] = {}
    for raw in list_curve_mnemonics(las):
        canon = _norm_mnemonic(raw)
        if canon:
            m.setdefault(canon, []).append(raw)
    return m


def resolve_curve_mnemonic(las, canonical_mnemonic: str) -> Optional[str]:
    """
    Resolve a CANONICAL mnemonic (e.g., 'GR') to a RAW mnemonic present
    in the LAS suitable for las[raw] indexing.

    Never assume canonical == raw.
    """
    canon = _norm_mnemonic(canonical_mnemonic)
    if not canon:
        return None

    raw_list = available_canonical_curves(las).get(canon, [])
    if not raw_list:
        return None

    def _pref_key(raw: str) -> Tuple[int, int, int]:
        r = (raw or "").strip()
        r_up = r.upper()
        exact = 0 if r_up == canon else 1
        no_suffix = 0 if ":" not in r_up else 1  # prefer GR over GR:1, etc.
        return (exact, no_suffix, len(r))

    return sorted(raw_list, key=_pref_key)[0]


def find_curve_mnemonic(mnemonics: Sequence[str], preferred: Sequence[str]) -> Optional[str]:
    """
    Choose the first match in preferred order, where preferred is intended to be CANONICAL.

    Returns a RAW mnemonic from `mnemonics` suitable for las[raw] indexing.

    Backward tolerant: if preferred includes raw/alias names, they are canonicalized anyway.
    """
    if not mnemonics:
        return None

    canon_map: Dict[str, List[str]] = {}
    for raw in mnemonics:
        c = _norm_mnemonic(raw)
        if c:
            canon_map.setdefault(c, []).append(raw)

    def _best_raw(canon: str) -> Optional[str]:
        raws = canon_map.get(canon, [])
        if not raws:
            return None

        def _pref_key(raw: str) -> Tuple[int, int, int]:
            r = (raw or "").strip()
            r_up = r.upper()
            exact = 0 if r_up == canon else 1
            no_suffix = 0 if ":" not in r_up else 1
            return (exact, no_suffix, len(r))

        return sorted(raws, key=_pref_key)[0]

    for p in preferred:
        canon = _norm_mnemonic(p)
        raw = _best_raw(canon)
        if raw is not None:
            return raw
    return None


# =============================================================================
# Numeric helpers
# =============================================================================

def _as_float_array(x) -> np.ndarray:
    """
    Convert arbitrary array-like to float64, coercing non-numeric entries to NaN.
    Avoids repeated element-wise Python loops for already-numeric arrays.
    """
    arr = np.asarray(x)
    if arr.dtype.kind in {"f", "i", "u"}:
        return arr.astype("float64", copy=False)

    # Fast-ish coercion attempt first
    try:
        return arr.astype("float64")
    except Exception:
        out = np.full(arr.shape, np.nan, dtype="float64")
        it = np.nditer(arr, flags=["multi_index", "refs_ok"])
        for v in it:
            try:
                out[it.multi_index] = float(v.item())
            except Exception:
                out[it.multi_index] = np.nan
        return out


def extract_depth_and_curve(
    las,
    *,
    curve_mnemonic: str,
    depth_preferred: Sequence[str] = ("DEPT", "MD"),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (depth, curve_values) as float arrays.

    - `curve_mnemonic` is treated as CANONICAL; we resolve to RAW before indexing.
    - Depth selection:
        1) Use numeric las.index if it has at least 2 finite values and a non-zero range.
        2) Otherwise resolve a depth curve from canonical preferences.

    Output is not guaranteed monotonic; downstream resampling should sort by depth.
    """
    target_canon = _norm_mnemonic(curve_mnemonic)
    if not target_canon:
        raise ValueError("curve_mnemonic is empty")

    # 1) Depth from las.index if viable
    depth: Optional[np.ndarray] = None
    try:
        idx = np.asarray(getattr(las, "index"))
        d = _as_float_array(idx)
        fin = np.isfinite(d)
        if d.ndim == 1 and int(np.count_nonzero(fin)) >= 2:
            dmin = float(np.nanmin(d))
            dmax = float(np.nanmax(d))
            if np.isfinite(dmin) and np.isfinite(dmax) and (dmax > dmin):
                depth = d
    except Exception:
        depth = None

    # 2) Fallback: depth curve
    if depth is None:
        raw_mn = list_curve_mnemonics(las)
        dep_raw = find_curve_mnemonic(raw_mn, depth_preferred)
        if dep_raw is None:
            avail = sorted({c for c in (_norm_mnemonic(m) for m in raw_mn) if c})
            raise RuntimeError(
                "Could not determine depth (no usable las.index and no depth curve found). "
                f"Depth preferred={list(depth_preferred)}. Available canonical={avail[:50]}"
            )
        depth = _as_float_array(las[dep_raw])  # type: ignore[index]

    # Resolve target curve raw mnemonic (critical: do not assume canonical exists as raw)
    curve_raw = resolve_curve_mnemonic(las, target_canon)
    if curve_raw is None:
        raw_mn = list_curve_mnemonics(las)
        avail = sorted({c for c in (_norm_mnemonic(m) for m in raw_mn) if c})
        raise KeyError(
            f"Canonical curve {target_canon} not found in LAS. "
            f"Available canonical={avail[:50]}"
        )

    y = _as_float_array(las[curve_raw])  # type: ignore[index]

    # Align lengths defensively
    n = int(min(depth.size, y.size))
    if n <= 0:
        raise RuntimeError("Depth/curve arrays are empty after extraction.")
    depth = depth[:n]
    y = y[:n]
    return depth.astype("float64", copy=False), y.astype("float64", copy=False)


# =============================================================================
# Resampling + DTW
# =============================================================================

def resample_and_normalize_curve(
    depth: np.ndarray,
    y: np.ndarray,
    *,
    n_samples: int,
    p_lo: float = 1.0,
    p_hi: float = 99.0,
    min_finite: int = 10,
) -> Tuple[np.ndarray, float, float]:
    """
    Resample curve to a uniform depth grid and normalize to [0,1] using robust percentiles.

    Returns:
      x_norm (n_samples,), z_top, z_base
    """
    depth = np.asarray(depth, dtype="float64").reshape(-1)
    y = np.asarray(y, dtype="float64").reshape(-1)

    n_samples_i = int(n_samples)
    if n_samples_i < 8:
        raise ValueError("n_samples too small; use >= 8")
    if not (0.0 <= float(p_lo) < float(p_hi) <= 100.0):
        raise ValueError("Percentiles must satisfy 0 <= p_lo < p_hi <= 100")

    if depth.size < 2:
        raise RuntimeError("Insufficient depth samples.")

    # sort by depth; keep y aligned
    order = np.argsort(depth)
    depth = depth[order]
    y = y[order]

    # finite depth only (y may still be non-finite; handled below)
    m = np.isfinite(depth)
    depth = depth[m]
    y = y[m]
    if depth.size < 2:
        raise RuntimeError("Insufficient finite depth after filtering.")

    z_top = float(np.nanmin(depth))
    z_base = float(np.nanmax(depth))
    if not np.isfinite(z_top) or not np.isfinite(z_base) or z_base <= z_top:
        raise RuntimeError("Invalid depth range.")

    z = np.linspace(z_top, z_base, n_samples_i, dtype="float64")

    fin = np.isfinite(y)
    if int(np.count_nonzero(fin)) < int(min_finite):
        raise RuntimeError("Too few finite curve samples to resample.")

    d_fin = depth[fin]
    y_fin = y[fin]

    # Aggregate duplicates by mean value (more stable than 'first wins')
    uniq_d, inv = np.unique(d_fin, return_inverse=True)
    if uniq_d.size < 2:
        raise RuntimeError("Degenerate depth after removing duplicates.")
    y_sum = np.bincount(inv, weights=y_fin.astype("float64", copy=False))
    y_cnt = np.bincount(inv).astype("float64", copy=False)
    y_mean = y_sum / np.maximum(1.0, y_cnt)

    y_rs = np.interp(z, uniq_d, y_mean, left=float(y_mean[0]), right=float(y_mean[-1])).astype("float64", copy=False)

    plo = float(np.percentile(y_rs, float(p_lo)))
    phi = float(np.percentile(y_rs, float(p_hi)))

    # If percentiles collapse, fall back to min/max normalization (still robust)
    if (not np.isfinite(plo)) or (not np.isfinite(phi)) or (phi <= plo):
        plo = float(np.nanmin(y_rs))
        phi = float(np.nanmax(y_rs))
        if (not np.isfinite(plo)) or (not np.isfinite(phi)) or (phi <= plo):
            # fully degenerate: return zeros but preserve top/base
            return np.zeros((n_samples_i,), dtype="float64"), z_top, z_base

    x = (y_rs - plo) / (phi - plo)
    x = np.clip(x, 0.0, 1.0)
    return x.astype("float64", copy=False), z_top, z_base


def dtw_cost_and_path(
    x: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = 0.15,
    backtrack: bool = True,
) -> Tuple[float, float, Optional[np.ndarray]]:
    """
    DTW with local distance d = |x_i - y_j|^alpha (alpha < 1 de-emphasizes outliers)

    Returns:
      cost_total, cost_per_step, path (optional; shape [L,2] with (i,j))
    """
    require_librosa()

    x = np.asarray(x, dtype="float64").reshape(-1)
    y = np.asarray(y, dtype="float64").reshape(-1)

    if x.size < 8 or y.size < 8:
        raise ValueError("x and y too short for DTW (use >= 8 samples each).")
    if float(alpha) <= 0.0:
        raise ValueError("alpha must be > 0")

    C = np.abs(x[:, None] - y[None, :]) ** float(alpha)

    # librosa API is stable here, but keep return handling defensive
    res = librosa.sequence.dtw(C=C, backtrack=bool(backtrack))  # type: ignore[attr-defined]
    if isinstance(res, tuple) and len(res) == 2:
        D, wp = res
    else:
        raise RuntimeError("Unexpected librosa.sequence.dtw return value.")

    D = np.asarray(D, dtype="float64")
    cost_total = float(D[-1, -1]) if D.size else float("nan")

    if wp is None:
        # Conservative denominator if path not available
        denom = float(max(1, x.size + y.size))
        return float(cost_total), float(cost_total / denom), None

    path = np.asarray(wp[::-1], dtype="int64")
    L = int(path.shape[0]) if path.ndim == 2 else 0
    cost_per_step = float(cost_total) / float(max(1, L))
    return float(cost_total), float(cost_per_step), path


def downsample_path(path: np.ndarray, n_tiepoints: int = 64) -> np.ndarray:
    """
    Downsample a DTW path to a small set of tiepoints.

    Returns array shape (K,2), with K<=n_tiepoints.
    """
    path = np.asarray(path)
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError("path must be (L,2)")
    L = int(path.shape[0])

    n = int(n_tiepoints)
    if n <= 0:
        raise ValueError("n_tiepoints must be > 0")

    if L <= n:
        return path.astype("int64", copy=False)

    idx = np.linspace(0, L - 1, n, dtype="int64")
    # Avoid duplicates for some (L,n) combos; keep sorted order
    idx = np.unique(idx)
    return path[idx].astype("int64", copy=False)
