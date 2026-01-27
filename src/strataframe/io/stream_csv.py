# src/strataframe/io/stream_csv.py
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Sequence


class CsvSink:
    """
    Streaming CSV writer that flushes on each write.

    Use for long-running pipelines (typewell runs) without buffering huge tables.
    """

    def __init__(self, path: Path, fieldnames: Sequence[str], *, append: bool) -> None:
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        exists = self.path.exists()
        mode = "a" if append else "w"
        self.f = self.path.open(mode, encoding="utf-8", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=self.fieldnames)

        if (not exists) or (not append):
            self.w.writeheader()
            self.f.flush()

    def write(self, row: Dict[str, Any]) -> None:
        self.w.writerow({k: row.get(k, "") for k in self.fieldnames})
        self.f.flush()

    def close(self) -> None:
        try:
            self.f.flush()
            self.f.close()
        except Exception:
            pass
