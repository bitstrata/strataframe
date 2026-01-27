# src/strataframe/util/hash_utils.py
from __future__ import annotations

import hashlib


def stable_hash32(s: str) -> int:
    """
    Stable 32-bit hash for deterministic RNG seeding across Python processes.
    (Avoids Python's salted hash().)
    """
    h = hashlib.md5((s or "").encode("utf-8")).hexdigest()[:8]
    return int(h, 16) & 0xFFFFFFFF
