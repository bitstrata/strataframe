# strataframe

strataframe is a scalable, unconformity-aware multi-well correlation and stratigraphic framework builder.

It uses:
- anchor-bounded zones (e.g., IHS tops) to prevent cross-unconformity warping
- sparse graphs with deliberate overlap (halo) for block-scale solves
- DTW-based pairwise alignment + anchored RGT optimization
- block stitching to reconcile adjacent analysis tiles

## quickstart
1) Create venv and install:
   python -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -e ".[dtw]"

2) Run (placeholders):
   strataframe-block run --wells data/wells.parquet --tops data/ihs_tops.parquet
