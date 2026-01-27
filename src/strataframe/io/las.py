# src/strataframe/io/las.py
from __future__ import annotations

import contextlib
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import lasio  # type: ignore
except Exception:  # pragma: no cover
    lasio = None  # type: ignore


def require_lasio() -> None:
    if lasio is None:
        raise RuntimeError("lasio is not installed. Install with: pip install lasio")


_WRAP_LINE_RE = re.compile(
    r"\bWRAP\b\s*(?:[.:=]|\s)\s*(YES|NO|TRUE|FALSE|0|1|Y|N)\b",
    re.IGNORECASE,
)


def is_wrapped_las(path: Path, *, max_bytes: int = 256_000) -> Optional[bool]:
    """
    Heuristic: scan the header for a WRAP line (YES/NO) without fully parsing.

    Returns:
      - True if WRAP=YES
      - False if WRAP=NO
      - None if cannot determine
    """
    try:
        with path.open("rb") as f:
            blob = f.read(max_bytes)
        txt = blob.decode("utf-8", errors="ignore")

        # Limit to header-ish content: stop if we see ~A (ASCII log data section)
        # This prevents accidental matches in data blocks.
        upper = txt.upper()
        cut = upper.find("~A")
        if cut > 0:
            txt = txt[:cut]

        for line in txt.splitlines():
            m = _WRAP_LINE_RE.search(line)
            if not m:
                continue
            val = m.group(1).strip().upper()
            if val in {"YES", "TRUE", "1", "Y"}:
                return True
            if val in {"NO", "FALSE", "0", "N"}:
                return False
            return None
        return None
    except Exception:
        return None


@contextlib.contextmanager
def quiet_stdio(enabled: bool = True):
    """
    Redirect stdout/stderr to suppress noisy third-party library prints.
    """
    if not enabled:
        yield
        return
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        yield


@dataclass(frozen=True)
class LasReadResult:
    las: "lasio.LASFile"
    engine_used: str
    wrapped: Optional[bool]


def _lasio_read(
    path: Path,
    *,
    engine: Optional[str],
    ignore_data: Optional[bool],
) -> "lasio.LASFile":
    """
    Handle lasio versions that may or may not accept engine= and/or ignore_data=.
    We progressively fall back by removing kwargs that trigger TypeError.
    """
    require_lasio()
    p = str(path)

    kwargs = {}
    if engine is not None:
        kwargs["engine"] = engine
    if ignore_data is not None:
        kwargs["ignore_data"] = bool(ignore_data)

    # Try full kwargs first, then progressively remove.
    try:
        return lasio.read(p, **kwargs)  # type: ignore[misc]
    except TypeError:
        pass

    if "engine" in kwargs:
        try:
            k2 = dict(kwargs)
            k2.pop("engine", None)
            return lasio.read(p, **k2)  # type: ignore[misc]
        except TypeError:
            pass

    if "ignore_data" in kwargs:
        try:
            k3 = dict(kwargs)
            k3.pop("ignore_data", None)
            return lasio.read(p, **k3)  # type: ignore[misc]
        except TypeError:
            pass

    # Final fallback: plain read
    return lasio.read(p)  # type: ignore[misc]


def read_las_safely(
    path: Path,
    *,
    prefer_fast_for_unwrapped: bool = True,
    ignore_data: bool = False,
    quiet: bool = True,
) -> LasReadResult:
    """
    Robust LAS reader:
      - if WRAP=YES -> engine='normal'
      - else try engine='fast' (optional) then fallback to 'normal'
      - tolerates lasio versions without engine=/ignore_data=
      - optionally silences stdout/stderr

    Parameters
    ----------
    ignore_data:
      True for header-only scanning (fast); False to read curves.
    """
    require_lasio()

    wrapped = is_wrapped_las(path)

    # Wrapped: prefer normal (wrapped files often require engine='normal')
    if wrapped is True:
        with quiet_stdio(quiet):
            las = _lasio_read(path, engine="normal", ignore_data=ignore_data)
        return LasReadResult(las=las, engine_used="normal", wrapped=wrapped)

    # Unwrapped or unknown: optionally try fast
    if prefer_fast_for_unwrapped:
        try:
            with quiet_stdio(quiet):
                las = _lasio_read(path, engine="fast", ignore_data=ignore_data)
            return LasReadResult(las=las, engine_used="fast", wrapped=wrapped)
        except Exception:
            pass

    # Fallback: normal
    with quiet_stdio(quiet):
        las = _lasio_read(path, engine="normal", ignore_data=ignore_data)
    return LasReadResult(las=las, engine_used="normal", wrapped=wrapped)


def read_las_header_only(
    path: Path,
    *,
    prefer_fast_for_unwrapped: bool = True,
    quiet: bool = True,
) -> LasReadResult:
    """
    Convenience wrapper for header-only reads.
    """
    return read_las_safely(
        path,
        prefer_fast_for_unwrapped=prefer_fast_for_unwrapped,
        ignore_data=True,
        quiet=quiet,
    )


def extract_curve_values(las: "lasio.LASFile", mnemonic: str) -> Optional[np.ndarray]:
    """
    Safely extract curve values by mnemonic (case-insensitive exact match).
    Returns None if missing or empty.

    Notes:
      - Does NOT apply alias-family matching; keep this low-level.
      - Higher-level code can try multiple mnemonics (e.g., aliases_for("GR")).
    """
    m = (mnemonic or "").strip()
    if not m:
        return None
    target = m.upper()

    try:
        lookup: dict[str, str] = {}
        for c in getattr(las, "curves", []) or []:
            try:
                mn = str(getattr(c, "mnemonic", "") or "").strip()
            except Exception:
                continue
            if not mn:
                continue
            lookup.setdefault(mn.upper(), mn)

        actual = lookup.get(target)
        if actual is None:
            return None

        arr = np.asarray(las[actual], dtype="float64")  # type: ignore[index]
        if arr.size == 0:
            return None
        return arr
    except Exception:
        return None


def list_curve_mnemonics(las: "lasio.LASFile") -> list[str]:
    """
    Return curve mnemonics in ~CURVE order (as stored in the LAS), e.g., ['DEPT', 'GR:1', ...].
    Safe for header-only reads (ignore_data=True).
    """
    out: list[str] = []
    try:
        for c in getattr(las, "curves", []) or []:
            mn = str(getattr(c, "mnemonic", "") or "").strip()
            if mn:
                out.append(mn)
    except Exception:
        pass
    return out


def find_curve_mnemonics_in_family(las: "lasio.LASFile", family_or_mnemonic: str) -> list[str]:
    """
    Header-only friendly: returns actual mnemonics present in the LAS whose canonical mnemonic
    matches the alias family (e.g., 'GR' will match 'GR', 'GAMMA', 'GR:1', etc.).
    """
    # Local import to keep io/las.py usable even if higher-level policy changes
    from strataframe.curves.normalize_header import aliases_for, norm_mnemonic

    fam = norm_mnemonic(family_or_mnemonic)
    if not fam:
        return []

    wanted = {norm_mnemonic(a) for a in aliases_for(fam) if norm_mnemonic(a)}
    out: list[str] = []
    for mn in list_curve_mnemonics(las):
        if norm_mnemonic(mn) in wanted:
            out.append(mn)
    return out


def header_has_depth_axis(las: "lasio.LASFile") -> bool:
    """
    Header-only check: true if we can plausibly resolve a depth axis from ~CURVE
    (DEPT/DEPTH/MD present in some form).
    """
    from strataframe.curves.normalize_header import norm_mnemonic

    wanted = {"DEPT", "DEPTH", "MD"}
    for mn in list_curve_mnemonics(las):
        if norm_mnemonic(mn) in wanted:
            return True
    return False
