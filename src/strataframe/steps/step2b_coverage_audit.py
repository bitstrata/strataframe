# src/strataframe/steps/step2b_coverage_audit.py
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CoverageConfig:
    short_frac: float = 0.90
    long_frac: float = 1.80
    min_expected_thk: float = 50.0
    verbose: bool = True


def run_coverage_audit(
    *,
    nodes_csv: Path,
    surface_predictions_csv: Path,
    out_dir: Path,
    cfg: CoverageConfig,
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes = pd.read_csv(nodes_csv)
    nodes["node_id"] = pd.to_numeric(nodes["node_id"], errors="coerce").astype("Int64")
    nodes = nodes[nodes["node_id"].notna()].copy()
    nodes["node_id"] = nodes["node_id"].astype(int)

    pred = pd.read_csv(surface_predictions_csv)
    pred["node_id"] = pd.to_numeric(pred["node_id"], errors="coerce").astype("Int64")
    pred = pred[pred["node_id"].notna()].copy()
    pred["node_id"] = pred["node_id"].astype(int)

    df = nodes.merge(pred, on="node_id", how="left", suffixes=("", "_pred"))
    if bool(cfg.verbose):
        print(f"[step2b] loaded nodes={nodes.shape[0]} preds={pred.shape[0]} merged={df.shape[0]}")

    z_exp_top = pd.to_numeric(df.get("z_exp_top"), errors="coerce")
    z_exp_base = pd.to_numeric(df.get("z_exp_base"), errors="coerce")
    z_act_top = pd.to_numeric(df.get("z_gr_top_trim"), errors="coerce")
    z_act_base = pd.to_numeric(df.get("z_gr_base_trim"), errors="coerce")

    exp_thk = z_exp_base - z_exp_top
    act_thk = z_act_base - z_act_top

    df["expected_thickness"] = exp_thk
    df["actual_thickness"] = act_thk
    df["missing_top_ft"] = (z_act_top - z_exp_top).clip(lower=0)
    df["missing_base_ft"] = (z_exp_base - z_act_base).clip(lower=0)
    df["coverage_ratio"] = act_thk / exp_thk.replace(0, np.nan)

    df["short_flag"] = df["coverage_ratio"] < float(cfg.short_frac)
    df["long_flag"] = df["coverage_ratio"] > float(cfg.long_frac)
    df["expected_bad"] = (~np.isfinite(exp_thk)) | (exp_thk < float(cfg.min_expected_thk))

    qc_ok = df.get("fit_ok")
    if qc_ok is None:
        qc_ok = df.get("qc_ok")
    if qc_ok is None:
        qc_ok = True

    if isinstance(qc_ok, (bool, np.bool_)):
        qc_ok_series = pd.Series([bool(qc_ok)] * int(df.shape[0]), index=df.index)
    else:
        qc_ok_series = pd.Series(qc_ok, index=df.index, dtype="boolean").fillna(False).astype(bool)

    df["exclude_flag"] = (~qc_ok_series) | df["expected_bad"] | df["long_flag"]

    # Create expected nodes CSV for downstream (depth_min/max replaced)
    nodes_expected = df.copy()
    nodes_expected["depth_min"] = z_exp_top
    nodes_expected["depth_max"] = z_exp_base
    nodes_expected_path = out_dir / "step2b_nodes_expected.csv"
    nodes_expected.to_csv(nodes_expected_path, index=False)

    audit_csv = out_dir / "step2b_coverage_audit.csv"
    df.to_csv(audit_csv, index=False)

    exclude_csv = out_dir / "step2b_excluded_nodes.csv"
    df[df["exclude_flag"]].loc[:, ["node_id"]].drop_duplicates().to_csv(exclude_csv, index=False)

    out_json = out_dir / "step2b_coverage.json"
    out_json.write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "counts": {
                    "n_nodes": int(df.shape[0]),
                    "n_exclude": int(df["exclude_flag"].sum()),
                    "n_short": int(df["short_flag"].sum()),
                    "n_long": int(df["long_flag"].sum()),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if bool(cfg.verbose):
        print(
            f"[step2b] exclude={int(df['exclude_flag'].sum())} short={int(df['short_flag'].sum())} long={int(df['long_flag'].sum())}"
        )

    return {
        "nodes_expected_csv": str(nodes_expected_path),
        "audit_csv": str(audit_csv),
        "excluded_csv": str(exclude_csv),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Step2b: coverage audit + expected depth window.")
    ap.add_argument("--nodes-csv", required=True)
    ap.add_argument("--surface-predictions-csv", required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--short-frac", type=float, default=0.90)
    ap.add_argument("--long-frac", type=float, default=1.80)
    ap.add_argument("--min-expected-thk", type=float, default=50.0)
    ap.add_argument("--quiet", action="store_true")

    args = ap.parse_args()
    cfg = CoverageConfig(
        short_frac=float(args.short_frac),
        long_frac=float(args.long_frac),
        min_expected_thk=float(args.min_expected_thk),
        verbose=(not bool(args.quiet)),
    )
    run_coverage_audit(
        nodes_csv=Path(args.nodes_csv),
        surface_predictions_csv=Path(args.surface_predictions_csv),
        out_dir=Path(args.out_dir),
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
