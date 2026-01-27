# scripts/curve_vocab.py
#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd
import lasio


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/las"))
    ap.add_argument("--out", type=Path, default=Path("out/las_curve_vocab.csv"))
    args = ap.parse_args()

    rows = []
    rejects = []

    for p in sorted(args.root.rglob("*.las")):
        try:
            las = lasio.read(p, ignore_data=True)
            for c in las.curves:
                rows.append(
                    {
                        "file": str(p),
                        "mnemonic": (c.mnemonic or "").strip(),
                        "unit": (c.unit or "").strip(),
                        "descr": (c.descr or "").strip(),
                    }
                )
        except Exception as e:
            rejects.append({"file": str(p), "error": str(e)})

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    # Aggregates
    agg = (
        df.groupby(["mnemonic", "unit"], dropna=False)
          .size()
          .reset_index(name="count")
          .sort_values("count", ascending=False)
    )
    agg_out = args.out.with_name("las_curve_vocab_counts.csv")
    agg.to_csv(agg_out, index=False)

    if rejects:
        rej_out = args.out.with_name("las_curve_vocab_rejects.csv")
        pd.DataFrame(rejects).to_csv(rej_out, index=False)

    print(f"Wrote: {args.out}")
    print(f"Wrote: {agg_out}")
    if rejects:
        print(f"Wrote rejects: {len(rejects)}")

if __name__ == "__main__":
    main()
