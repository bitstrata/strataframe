# src/strataframe/utils/fingerprint.py
from __future__ import annotations

import hashlib
from pathlib import Path


def sha1_file_prefix(path: Path, *, max_bytes: int = 2_000_000) -> str:
    """
    Cheap-ish fingerprint: sha1(first max_bytes) + file size.

    Notes:
    - Intentionally does NOT hash full files (fast + stable enough for manifests).
    - If the file is smaller than max_bytes, this is effectively sha1(file) + size.
    - If the file cannot be read, returns sha1(size-only) so callers still get a deterministic value.
    """
    h = hashlib.sha1()

    try:
        size = int(path.stat().st_size)
    except Exception:
        size = -1

    # Include the size up front as well as at the end; this makes the fallback case
    # (unreadable file) still distinct per-size, and does not change normal behavior materially.
    h.update(str(size).encode("utf-8"))
    h.update(b"|")

    try:
        with path.open("rb") as f:
            h.update(f.read(int(max_bytes)))
    except Exception:
        # leave as size-only fingerprint
        pass

    h.update(b"|")
    h.update(str(size).encode("utf-8"))
    return h.hexdigest()
