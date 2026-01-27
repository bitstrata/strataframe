# src/strataframe/util/state_json.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def write_state_json(out_dir: Path, payload: Dict[str, Any], *, filename: str = "step2_state.json") -> None:
    try:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass
