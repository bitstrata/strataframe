from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


@dataclass(frozen=True)
class IOConfig:
    wells_parquet: Path
    ihs_tops_parquet: Path
    out_dir: Path
    work_dir: Path


@dataclass(frozen=True)
class BlockConfig:
    mode: str = "tiles"
    tile_km: float = 20.0
    halo_km: float = 5.0
    min_control_wells: int = 20
    max_control_wells: int = 80


@dataclass(frozen=True)
class GraphConfig:
    k_nn: int = 10
    max_radius_km: float = 15.0
    add_mst: bool = True
    add_random_long_edges: int = 1
    embed_prune_tau: float = 0.42


@dataclass(frozen=True)
class DTWConfig:
    alpha: float = 0.20
    band_frac: float = 0.10
    use_lb_keogh: bool = True
    n_jobs: int = 8


@dataclass(frozen=True)
class RGTConfig:
    lambda_anchor: float = 50.0
    enforce_monotonic: bool = True
    pinch_rule: str = "zone_default"
    downsample: int = 10


@dataclass(frozen=True)
class StitchConfig:
    model: str = "affine_per_zone"
    robust_loss: str = "huber"
    huber_delta: float = 1.5
    regularize_a: float = 10.0
    regularize_b: float = 1.0


@dataclass(frozen=True)
class RunConfig:
    io: IOConfig
    blocks: BlockConfig = BlockConfig()
    graph: GraphConfig = GraphConfig()
    dtw: DTWConfig = DTWConfig()
    rgt: RGTConfig = RGTConfig()
    stitch: StitchConfig = StitchConfig()
    zones: Optional[Sequence[str]] = None
