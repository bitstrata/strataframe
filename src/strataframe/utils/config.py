# src/strataframe/utils/config.py
from __future__ import annotations

from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


def require_yaml() -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required for --config. Install with: pip install pyyaml")


def _find_repo_root(start: Path) -> Path:
    """
    Heuristic: walk up from start looking for a 'scripts' directory.
    Falls back to start if not found.
    """
    p = start.resolve()
    for _ in range(10):
        if (p / "scripts").exists():
            return p
        if p.parent == p:
            break
        p = p.parent
    return start.resolve()


def resolve_config_path(path: Path) -> Path:
    """
    Resolve config path robustly:
      1) as given (absolute or relative to CWD)
      2) relative to repo root (parent chain containing ./scripts)
    """
    p = Path(path)
    if p.exists():
        return p.resolve()

    repo = _find_repo_root(Path.cwd())
    p2 = (repo / p).resolve()
    if p2.exists():
        return p2

    return p  # caller will raise with the original


def load_yaml(path: Path) -> Dict[str, Any]:
    require_yaml()
    p = resolve_config_path(Path(path))
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    obj = yaml.safe_load(p.read_text(encoding="utf-8"))
    return obj if isinstance(obj, dict) else {}


def deep_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    cur: Any = d
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out


def as_plain_dict(x: Any) -> Any:
    if is_dataclass(x):
        return {k: as_plain_dict(v) for k, v in x.__dict__.items()}
    if isinstance(x, dict):
        return {k: as_plain_dict(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [as_plain_dict(v) for v in x]
    return x
