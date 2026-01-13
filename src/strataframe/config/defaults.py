from __future__ import annotations

from pathlib import Path

from .schema import IOConfig, RunConfig


def default_config() -> RunConfig:
    base = Path.cwd()
    return RunConfig(
        io=IOConfig(
            wells_parquet=base / "data" / "wells.parquet",
            ihs_tops_parquet=base / "data" / "ihs_tops.parquet",
            out_dir=base / "out",
            work_dir=base / "work",
        )
    )
