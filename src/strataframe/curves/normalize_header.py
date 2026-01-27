# src/strataframe/curves/normalize_header.py
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

# =============================================================================
# Canonicalization tables (header-level: mnemonics + units)
# =============================================================================

# Alias (raw/variant) -> canonical mnemonic
#
# Design intent:
# - Canonical mnemonics are the join keys used across the codebase.
# - Keep RHOB/RHOC distinct (RHOC = density correction).
# - Keep resistivity families distinct.
# - Do NOT over-collapse porosity families (DPHI/DPOR/SPOR remain distinct),
#   but do map common "total neutron porosity" mnemonics onto NPHI.
MNEMONIC_ALIASES: Dict[str, str] = {
    # -----------------------------
    # Depth
    # -----------------------------
    "DEPTH": "DEPT",
    "DPTH": "DEPT",
    "DEPT": "DEPT",
    "MD": "MD",             # measured depth (sometimes used as depth curve)
    "TVD": "TVD",           # true vertical depth (occasionally present)

    # -----------------------------
    # Gamma ray
    # -----------------------------
    "GAM": "GR",
    "GAMMA": "GR",
    "GAMMARAY": "GR",
    "GAMMA_RAY": "GR",
    "GAMMA-RAY": "GR",
    "CGR": "GR",            # corrected GR
    "SGR": "GR",            # spectral GR (often still used as GR proxy)
    "GR": "GR",

    # -----------------------------
    # SP
    # -----------------------------
    "SP": "SP",

    # -----------------------------
    # Density (keep RHOB/RHOC distinct; RHOC is density correction)
    # -----------------------------
    "RHOB": "RHOB",
    "RHOC": "RHOC",
    "DRHO": "DRHO",
    # Common generic density mnemonics -> RHOB (bulk density)
    "DEN": "RHOB",
    "RHO": "RHOB",
    "RHOZ": "RHOB",
    "ZDEN": "RHOB",

    # -----------------------------
    # Neutron / porosity families (do not over-collapse; keep distinct)
    # -----------------------------
    "NPHI": "NPHI",
    "NPOR": "NPOR",
    "CNPOR": "CNPOR",
    "TNPH": "NPHI",         # total neutron porosity variants
    "TNPHI": "NPHI",
    "DPHI": "DPHI",
    "DPOR": "DPOR",
    "SPOR": "SPOR",
    "PHID": "DPHI",         # common alias for density porosity
    "PHIE": "DPHI",         # effective density porosity (often close enough for family scoring)

    # -----------------------------
    # Caliper
    # -----------------------------
    "CALI": "CALI",
    "CALIPER": "CALI",
    "DCAL": "DCAL",
    "MCAL": "MCAL",

    # -----------------------------
    # Sonic
    # -----------------------------
    "DT": "DT",
    "DTC": "DTC",
    "DTS": "DTS",
    "DTCO": "DTC",          # common compressional sonic mnemonic

    # -----------------------------
    # Photoelectric
    # -----------------------------
    "PE": "PE",
    "PEF": "PE",

    # -----------------------------
    # Resistivity families (keep distinct; normalize later by units if needed)
    # -----------------------------
    "ILD": "ILD",
    "ILM": "ILM",
    "LLD": "LLD",
    "LLS": "LLS",
    "RILD": "RILD",
    "RILM": "RILM",
    "RLL3": "RLL3",
    "RT": "RT",
    "RXO": "RXO",
    "RXRT": "RXRT",
    "RXORT": "RXORT",

    # -----------------------------
    # Conductivity variants seen in some corpora (optional)
    # -----------------------------
    "CILD": "CILD",
    "CILM": "CILM",
    "CLL3": "CLL3",
}

# Alias (raw/variant) -> canonical unit
UNIT_ALIASES: Dict[str, str] = {
    # Depth
    "F": "FT",
    "FT": "FT",
    "FEET": "FT",
    "M": "M",
    "METERS": "M",
    "METRES": "M",

    # Gamma ray
    "GAPI": "GAPI",
    "API": "GAPI",
    "API-GR": "GAPI",
    "CPS": "CPS",
    "CPM": "CPM",

    # SP
    "MV": "MV",
    "MILLIVOLTS": "MV",
    "V": "V",

    # Density
    "G/CC": "G/CC",
    "GM/CC": "G/CC",
    "G/C3": "G/CC",
    "GM/C3": "G/CC",
    "G/CM3": "G/CC",
    "GM/CM3": "G/CC",
    "KG/M3": "KG/M3",

    # Sonic
    "USEC/FT": "US/FT",
    "US/FT": "US/FT",
    "US/F": "US/FT",
    "USEC": "US",
    "US": "US",

    # Caliper
    "IN": "IN",
    "INCH": "IN",
    "INCHES": "IN",
    "MM": "MM",

    # Resistivity (use OHM-M as canonical when length-normalized)
    "OHM-M": "OHM-M",
    "OHMM": "OHM-M",
    "OHM/M": "OHM-M",
    "OHM.M": "OHM-M",
    "OHMS": "OHM",
    "OHM": "OHM",

    # Conductivity
    "MMHO/M": "MMHO/M",
    "MMHOS/M": "MMHO/M",
    "MMHO-M": "MMHO/M",
    "MMHO": "MMHO",

    # Porosity (canonicalize to PU)
    "PU": "PU",
    "PERC": "PU",
    "PERCENT": "PU",
    "%": "PU",          # effective because unit cleaning allows '%'
    "V/V": "PU",
    "VV": "PU",
    "FRAC": "PU",
    "DEC": "PU",
    "DECP": "PU",
    "CFCF": "PU",

    # Some LAS use NONE for unitless (norm_unit maps this to "")
    "NONE": "",
}

# Allowed canonical units per canonical mnemonic (optional but useful for QA/quarantine)
#
# IMPORTANT:
# - This is intended to be checked against *normalized* units (output of norm_unit()).
# - Porosity should be ("PU", "") (not pre-normalization variants).
# - Do not include "NONE" here; "NONE" is normalized to "".
ALLOWED_UNITS: Dict[str, Sequence[str]] = {
    # depth
    "DEPT": ("FT", "M", ""),
    "MD": ("FT", "M", ""),
    "TVD": ("FT", "M", ""),

    # GR / SP
    "GR": ("GAPI", "CPS", "CPM", ""),
    "SP": ("MV", "V", ""),

    # density family
    "RHOB": ("G/CC", "KG/M3", ""),
    "RHOC": ("G/CC", "KG/M3", ""),
    "DRHO": ("G/CC", "KG/M3", ""),

    # sonic
    "DT": ("US/FT", ""),
    "DTC": ("US/FT", ""),
    "DTS": ("US/FT", ""),

    # caliper
    "CALI": ("IN", "MM", ""),
    "DCAL": ("IN", "MM", ""),
    "MCAL": ("IN", "MM", ""),

    # photoelectric
    "PE": ("",),  # unit varies a lot; treat as unitless for QA unless you want to enforce

    # porosity families (post-normalization)
    "NPHI": ("PU", ""),
    "NPOR": ("PU", ""),
    "CNPOR": ("PU", ""),
    "DPHI": ("PU", ""),
    "DPOR": ("PU", ""),
    "SPOR": ("PU", ""),

    # resistivity families
    "ILD": ("OHM-M", "OHM", ""),
    "ILM": ("OHM-M", "OHM", ""),
    "LLD": ("OHM-M", "OHM", ""),
    "LLS": ("OHM-M", "OHM", ""),
    "RILD": ("OHM-M", "OHM", ""),
    "RILM": ("OHM-M", "OHM", ""),
    "RLL3": ("OHM-M", "OHM", ""),
    "RT": ("OHM-M", "OHM", ""),
    "RXO": ("OHM-M", "OHM", ""),
    "RXRT": ("OHM-M", "OHM", ""),     # "NONE" normalizes to ""
    "RXORT": ("OHM-M", "OHM", ""),

    # conductivity families
    "CILD": ("MMHO/M", "MMHO", ""),
    "CILM": ("MMHO/M", "MMHO", ""),
    "CLL3": ("MMHO/M", "MMHO", ""),
}


# =============================================================================
# Token cleaning (note: mnemonics and units have different character needs)
# =============================================================================

_WS_RE = re.compile(r"\s+")
# Mnemonics often appear as GR:1, GR:2, RHOB, GAMMA_RAY, etc. Keep ":" for suffix stripping.
_MNEM_BAD_RE = re.compile(r"[^A-Z0-9_:/\-]+")
# Units often include / . - (e.g., US/FT, OHM-M, G/CC). Allow '%' for porosity units.
_UNIT_BAD_RE = re.compile(r"[^A-Z0-9_/.\-%]+")
# Strip common LAS suffix patterns like GR:1, GR:2 (keeps canonical family stable)
_MNEM_SUFFIX_RE = re.compile(r":\d+$")


def _clean_mnemonic_token(x: Optional[str]) -> str:
    if x is None:
        return ""
    s = x.strip()
    if not s:
        return ""
    s = _WS_RE.sub("", s).upper()
    s = _MNEM_BAD_RE.sub("", s)
    return s


def _clean_unit_token(x: Optional[str]) -> str:
    if x is None:
        return ""
    s = x.strip()
    if not s:
        return ""
    s = _WS_RE.sub("", s).upper()
    s = s.replace("\\", "/")
    s = _UNIT_BAD_RE.sub("", s)
    return s


# =============================================================================
# Public API
# =============================================================================

def norm_mnemonic(mnemonic: Optional[str]) -> str:
    """
    Canonicalize a LAS curve mnemonic for joins/contracts.

    - Uppercases, removes whitespace
    - Preserves ":" long enough to strip suffix like ":1"
    - Maps known aliases to canonical mnemonic via MNEMONIC_ALIASES
    """
    m = _clean_mnemonic_token(mnemonic)
    if not m:
        return ""
    m = _MNEM_SUFFIX_RE.sub("", m)
    return MNEMONIC_ALIASES.get(m, m)


def norm_unit(unit: Optional[str]) -> str:
    """
    Canonicalize a LAS curve unit for joins/contracts.
    """
    u = _clean_unit_token(unit)
    if not u:
        return ""

    # Backward-compatible fast paths (kept explicit)
    if u in {"OHMM", "OHM/M", "OHM.M"}:
        u = "OHM-M"
    if u in {"G/C3", "GM/C3", "GM/CC", "G/CM3", "GM/CM3"}:
        u = "G/CC"
    if u in {"USEC/FT", "US/F"}:
        u = "US/FT"
    if u in {"INCH", "INCHES"}:
        u = "IN"

    return UNIT_ALIASES.get(u, u)


def canon_key(mnemonic: Optional[str], unit: Optional[str]) -> Tuple[str, str, str]:
    """
    Return (mnemonic_n, unit_n, "mnemonic_n|unit_n") suitable for deterministic grouping/joining.
    """
    mn = norm_mnemonic(mnemonic)
    un = norm_unit(unit)
    return mn, un, f"{mn}|{un}"


def aliases_for(mnemonic_or_canon: str, *, normalized: bool = True) -> List[str]:
    canon = norm_mnemonic(mnemonic_or_canon)
    if not canon:
        return []
    fam = list(_ALIASES_BY_CANON.get(canon, [canon]))
    if not normalized:
        return fam
    # Return cleaned/upper tokens (callers can do exact compare vs normalized header)
    out: List[str] = []
    for x in fam:
        y = _clean_mnemonic_token(x)
        if y:
            out.append(y)
    return out


# =============================================================================
# Internal: canonical -> alias family (preserves insertion order from MNEMONIC_ALIASES)
# =============================================================================

def _build_aliases_by_canon() -> Dict[str, List[str]]:
    by: Dict[str, List[str]] = {}
    seen: Dict[str, set[str]] = {}

    # Seed with canonical mnemonics we know about (keeps canonical first)
    for _alias, canon in MNEMONIC_ALIASES.items():
        c = _clean_mnemonic_token(canon)
        if not c:
            continue
        by.setdefault(c, [])
        seen.setdefault(c, set())

    # Ensure canonical itself is first
    for c in list(by.keys()):
        by[c].append(c)
        seen[c].add(c)

    # Add aliases in definition order (stable, predictable)
    for alias, canon in MNEMONIC_ALIASES.items():
        a = _clean_mnemonic_token(alias)
        c = _clean_mnemonic_token(canon)
        if not a or not c:
            continue
        if a not in seen[c]:
            by[c].append(a)
            seen[c].add(a)

    return by


_ALIASES_BY_CANON: Dict[str, List[str]] = _build_aliases_by_canon()
