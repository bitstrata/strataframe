# src/strataframe/chronolog/shared.py
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from strataframe.curves.normalize_header import aliases_for, norm_mnemonic


# Canonical log families we plan to support. Keep this small and explicit.
DEFAULT_LOG_FAMILIES: Tuple[str, ...] = ("GR", "DPHI", "NPHI", "PE")


def resolve_family_candidates(
    family: str,
    *,
    custom_families: Optional[Dict[str, Sequence[str]]] = None,
) -> List[str]:
    """
    Return candidate mnemonics for a log family.

    - Uses custom_families if provided.
    - Falls back to aliases_for() from normalize_header (canonical + aliases).
    """
    fam = norm_mnemonic(family)
    if not fam:
        return []

    if custom_families and fam in custom_families:
        return [norm_mnemonic(x) for x in custom_families.get(fam, []) if norm_mnemonic(x)]

    return [norm_mnemonic(x) for x in aliases_for(fam, normalized=True) if norm_mnemonic(x)]


def family_candidates_set(
    family: str,
    *,
    custom_families: Optional[Dict[str, Sequence[str]]] = None,
) -> Iterable[str]:
    return set(resolve_family_candidates(family, custom_families=custom_families))
