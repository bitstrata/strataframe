# src/strataframe/utils/heartbeat.py
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict


def _rss_mb_ps() -> float:
    """
    macOS-friendly RSS using `ps`. Returns MB, or NaN on failure.
    """
    try:
        pid = str(os.getpid())
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", pid], text=True).strip()
        kb = float(out)
        return kb / 1024.0
    except Exception:
        return float("nan")


def _open_fds() -> int:
    """
    Rough open-fd count for this process on macOS.
    """
    try:
        return len(os.listdir("/dev/fd"))
    except Exception:
        return -1


def write_heartbeat(path: Path, payload: Dict[str, Any]) -> None:
    """
    Atomic JSON write (tmp + replace) so tailing readers never see partial JSON.
    Adds ts/pid/rss_mb/open_fds automatically.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    obj = dict(payload)
    obj["ts"] = time.time()
    obj["pid"] = os.getpid()
    obj["rss_mb"] = _rss_mb_ps()
    obj["open_fds"] = _open_fds()

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)
