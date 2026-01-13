from __future__ import annotations

from pathlib import Path
import json
import typer
from rich import print

from strataframe.config.defaults import default_config
from strataframe.config.schema import IOConfig, RunConfig
from strataframe.io.wells import load_wells_parquet
from strataframe.io.ihs_tops import load_ihs_tops
from strataframe.spatial.blocks import make_tiles_with_halo

app = typer.Typer(add_completion=False)


@app.command()
def run(
    wells: Path = typer.Option(..., exists=True, help="Parquet: well_id,x,y,md,GR,..."),
    tops: Path = typer.Option(..., exists=True, help="Parquet: well_id,top_name,depth,sigma_m"),
    out_dir: Path = typer.Option(Path("out")),
    work_dir: Path = typer.Option(Path("work")),
    tile_km: float = typer.Option(20.0),
    halo_km: float = typer.Option(5.0),
):
    base = default_config()
    cfg = RunConfig(
        io=IOConfig(wells, tops, out_dir, work_dir),
        blocks=base.blocks.__class__(mode="tiles", tile_km=tile_km, halo_km=halo_km),
        graph=base.graph,
        dtw=base.dtw,
        rgt=base.rgt,
        stitch=base.stitch,
        zones=base.zones,
    )

    cfg.io.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.io.work_dir.mkdir(parents=True, exist_ok=True)

    print("[bold]Loading wells...[/bold]")
    wells_list = load_wells_parquet(str(cfg.io.wells_parquet), log_cols=["GR"])
    print(f"Loaded wells: {len(wells_list)}")

    print("[bold]Loading IHS tops...[/bold]")
    tops_map = load_ihs_tops(str(cfg.io.ihs_tops_parquet))
    print(f"Loaded tops for wells: {len(tops_map)}")

    print("[bold]Building blocks (tiles + halo)...[/bold]")
    blocks, adj = make_tiles_with_halo(wells_list, tile_km=cfg.blocks.tile_km, halo_km=cfg.blocks.halo_km)
    print(f"Blocks: {len(blocks)} | Adjacencies: {len(adj)}")

    manifest = {
        "wells": str(cfg.io.wells_parquet),
        "tops": str(cfg.io.ihs_tops_parquet),
        "n_wells": len(wells_list),
        "n_blocks": len(blocks),
        "n_adjacencies": len(adj),
        "tile_km": tile_km,
        "halo_km": halo_km,
    }
    (cfg.io.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("[green]Wrote[/green]", cfg.io.out_dir / "manifest.json")


if __name__ == "__main__":
    app()
