# src/strataframe/viz/step3_viz.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from strataframe.steps.step3b_rep_arrays import load_rep_arrays_npz

from .step3_common import ensure_dir, find_first, load_edges_csv, load_reps
from .step3a_candidates_map import plot_step3a_candidates_map
from .step3b_panel import plot_step3b_rep_arrays_panel
from .step3c_dtw_map import plot_step3c_dtw_map
from .step3d_fence import plot_step3d_framework_and_fence
from .step3e_chronostrat import plot_step3e_chronostrat


# -----------------------------------------------------------------------------
# CLI config
# -----------------------------------------------------------------------------

_STEP_ALIASES = {
    "3a": "3a",
    "3b": "3b",
    "3c": "3c",
    "3d": "3d",
    "3e": "3e",
    "all": "all",
}


@dataclass(frozen=True)
class Step3VizConfig:
    max_edges: int = 80_000
    max_wells_fence: int = 60
    seeds: Tuple[int, ...] = (42,)
    steps: Tuple[str, ...] = ("3a", "3b", "3c", "3d", "3e")
    # difference cross-sections (image-space diff between fences)
    write_diffs: bool = False
    diff_ref_seed: Optional[int] = None
    diff_mode: str = "abs"  # "abs" or "signed"
    diff_clip_pctl: float = 99.0  # contrast stretch for output png


def _parse_steps(s: str) -> Tuple[str, ...]:
    raw = [t.strip().lower() for t in s.split(",") if t.strip()]
    if not raw:
        return ("3a", "3b", "3c", "3d", "3e")
    out: List[str] = []
    for t in raw:
        t2 = _STEP_ALIASES.get(t)
        if t2 is None:
            raise ValueError(f"Unknown step token: {t!r}. Use 3a,3b,3c,3d,3e,all.")
        if t2 == "all":
            return ("3a", "3b", "3c", "3d", "3e")
        out.append(t2)
    # de-dupe preserving order
    seen = set()
    out2 = []
    for t in out:
        if t not in seen:
            seen.add(t)
            out2.append(t)
    return tuple(out2)


def _parse_seeds(
    *,
    seed: int,
    seeds_csv: str,
    n_fences: int,
    seed_step: int,
) -> Tuple[int, ...]:
    if seeds_csv.strip():
        vals: List[int] = []
        for tok in seeds_csv.split(","):
            tok = tok.strip()
            if not tok:
                continue
            vals.append(int(tok))
        if not vals:
            raise ValueError("--seeds was provided but parsed empty.")
        return tuple(vals)

    if n_fences <= 1:
        return (int(seed),)

    step = int(seed_step) if int(seed_step) != 0 else 1
    return tuple(int(seed) + i * step for i in range(int(n_fences)))


# -----------------------------------------------------------------------------
# Difference cross-section utilities (image-space diffs)
# -----------------------------------------------------------------------------

def _read_png_float(path: Path) -> np.ndarray:
    """
    Read a PNG into float32 in [0,1], shape (H,W,3).
    Robust to grayscale and alpha.
    """
    im = plt.imread(str(path))
    if im.ndim == 2:
        im = np.stack([im, im, im], axis=-1)
    elif im.ndim == 3 and im.shape[2] == 4:
        im = im[..., :3]
    im = im.astype("float32", copy=False)
    if im.max() > 1.5:
        im = im / 255.0
    return im


def _center_crop_to_common(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Center-crop both images to the min common H,W so diffs are meaningful even when
    fence figure widths vary by seed (common in step3d with dynamic figsize).
    """
    ha, wa = a.shape[0], a.shape[1]
    hb, wb = b.shape[0], b.shape[1]
    h = min(ha, hb)
    w = min(wa, wb)

    def crop(im: np.ndarray, h: int, w: int) -> np.ndarray:
        H, W = im.shape[0], im.shape[1]
        y0 = max(0, (H - h) // 2)
        x0 = max(0, (W - w) // 2)
        return im[y0 : y0 + h, x0 : x0 + w, :]

    return crop(a, h, w), crop(b, h, w)


def _write_diff_png(
    *,
    ref_png: Path,
    other_png: Path,
    out_png: Path,
    mode: str = "abs",
    clip_pctl: float = 99.0,
) -> None:
    """
    Create a "difference cross-section" PNG by image-space differencing between two
    step3d fence renders. This is primarily for comparing how A–A' changes with seed.
    """
    a = _read_png_float(ref_png)
    b = _read_png_float(other_png)
    a2, b2 = _center_crop_to_common(a, b)

    if mode == "signed":
        d = (b2 - a2)  # [-1,1]
        # convert to grayscale-like magnitude while preserving sign via mid-gray baseline
        g = d.mean(axis=2)
        # map [-p,p] -> [0,1] around 0.5
        p = float(np.nanpercentile(np.abs(g), float(clip_pctl))) if np.isfinite(g).any() else 1.0
        p = p if p > 1e-9 else 1.0
        out = np.clip(0.5 + 0.5 * (g / p), 0.0, 1.0)
    else:
        d = np.abs(b2 - a2)
        g = d.mean(axis=2)
        p = float(np.nanpercentile(g, float(clip_pctl))) if np.isfinite(g).any() else 1.0
        p = p if p > 1e-9 else 1.0
        out = np.clip(g / p, 0.0, 1.0)

    ensure_dir(out_png.parent)
    plt.imsave(str(out_png), out, cmap="gray", vmin=0.0, vmax=1.0)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Generate Step 3 (3a–3e) visualizations from cached outputs.\n\n"
            "Key additions:\n"
            "  - Generate multiple Step 3d fence cross-sections (A–A') across seeds.\n"
            "  - Optionally write 'difference cross-sections' (image diffs) vs a reference seed.\n"
            "  - Optionally run only specific steps via --steps (e.g., 3b only, 3d only).\n"
        )
    )
    ap.add_argument("--reps-csv", type=str, required=True)
    ap.add_argument("--step3-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default="")

    ap.add_argument("--max-edges", type=int, default=80_000)
    ap.add_argument("--max-wells-fence", type=int, default=60)

    # seed controls
    ap.add_argument("--seed", type=int, default=42, help="Base seed (used if --seeds not provided).")
    ap.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Comma-separated seed list (e.g., 1,2,3). Overrides --seed/--n-fences.",
    )
    ap.add_argument(
        "--n-fences",
        type=int,
        default=1,
        help="Generate N step3d fences using seeds seed + i*seed_step (ignored if --seeds is provided).",
    )
    ap.add_argument(
        "--seed-step",
        type=int,
        default=1,
        help="Step size for generated seeds when using --n-fences.",
    )

    # step selection
    ap.add_argument(
        "--steps",
        type=str,
        default="all",
        help="Comma-separated subset: 3a,3b,3c,3d,3e,all. Example: --steps 3b or --steps 3d.",
    )

    # difference cross-sections (image diffs between fence renders)
    ap.add_argument(
        "--write-diff-fences",
        action="store_true",
        help="If set and multiple seeds are produced, write diff images vs reference seed.",
    )
    ap.add_argument(
        "--diff-ref-seed",
        type=int,
        default=None,
        help="Reference seed for diffs. Default: first seed in the list.",
    )
    ap.add_argument(
        "--diff-mode",
        type=str,
        default="abs",
        choices=("abs", "signed"),
        help="Diff mode: abs (magnitude) or signed (mapped around mid-gray).",
    )
    ap.add_argument(
        "--diff-clip-pctl",
        type=float,
        default=99.0,
        help="Percentile for contrast stretch of diff output (higher = less aggressive).",
    )

    args = ap.parse_args(list(argv) if argv is not None else None)

    reps_csv = Path(args.reps_csv)
    step3_dir = Path(args.step3_dir)
    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else (step3_dir / "viz")
    ensure_dir(out_dir)

    steps = _parse_steps(str(args.steps))
    seeds = _parse_seeds(
        seed=int(args.seed),
        seeds_csv=str(args.seeds),
        n_fences=int(args.n_fences),
        seed_step=int(args.seed_step),
    )

    cfg = Step3VizConfig(
        max_edges=int(args.max_edges),
        max_wells_fence=int(args.max_wells_fence),
        seeds=tuple(int(s) for s in seeds),
        steps=tuple(steps),
        write_diffs=bool(args.write_diff_fences),
        diff_ref_seed=(int(args.diff_ref_seed) if args.diff_ref_seed is not None else None),
        diff_mode=str(args.diff_mode),
        diff_clip_pctl=float(args.diff_clip_pctl),
    )

    reps = load_reps(reps_csv)

    cand_csv = find_first(step3_dir, ["candidate_edges.csv", "candidates.csv", "step3a_candidate_edges.csv"])
    rep_arrays_npz = find_first(step3_dir, ["rep_arrays.npz", "rep_arrays_cache.npz", "step3b_rep_arrays.npz"])
    dtw_edges_csv = find_first(step3_dir, ["dtw_edges.csv", "step3c_dtw_edges.csv"])
    dtw_paths_npz = find_first(step3_dir, ["dtw_paths.npz", "step3c_dtw_paths.npz"])
    fw_edges_csv = find_first(step3_dir, ["framework_edges.csv", "step3d_framework_edges.csv", "framework.csv"])
    chron_npz = find_first(step3_dir, ["chronostrat.npz", "step3e_chronostrat.npz"])

    candidate_edges = load_edges_csv(cand_csv) if cand_csv else None

    # 3a
    if "3a" in cfg.steps:
        plot_step3a_candidates_map(
            reps=reps,
            candidate_edges=candidate_edges,
            out_png=out_dir / "step3a_candidates_map.png",
            max_edges=cfg.max_edges,
        )

    # 3b
    if "3b" in cfg.steps:
        if rep_arrays_npz and rep_arrays_npz.exists():
            rep_arrays = load_rep_arrays_npz(str(rep_arrays_npz))
            rep_ids = sorted(list(rep_arrays.keys()))  # deterministic
            plot_step3b_rep_arrays_panel(
                rep_arrays,
                rep_ids,
                str(out_dir / "step3b_rep_arrays_panel.png"),
                max_wells=200,
                show_trace=True,
            )

    # 3c
    if "3c" in cfg.steps:
        if dtw_edges_csv and dtw_edges_csv.exists():
            dtw_edges = load_edges_csv(dtw_edges_csv)
            plot_step3c_dtw_map(
                reps=reps,
                candidate_edges=candidate_edges,
                dtw_edges=dtw_edges,
                out_png=out_dir / "step3c_dtw_map.png",
                max_edges=cfg.max_edges,
            )

    # 3d (multi-seed fences)
    fence_png_by_seed: List[Tuple[int, Path]] = []
    if "3d" in cfg.steps:
        if fw_edges_csv and fw_edges_csv.exists() and rep_arrays_npz and rep_arrays_npz.exists():
            fw_edges = load_edges_csv(fw_edges_csv)

            for s in cfg.seeds:
                seed_dir = out_dir  # keep flat; seed is embedded in filenames
                out_map_png = seed_dir / f"step3d_framework_map_seed{s}.png"
                out_fence_png = seed_dir / f"step3d_fence_seed{s}.png"
                out_order_csv = seed_dir / f"step3d_fence_order_seed{s}.csv"

                plot_step3d_framework_and_fence(
                    reps=reps,
                    rep_arrays_npz=rep_arrays_npz,
                    framework_edges=fw_edges,
                    dtw_paths_npz=dtw_paths_npz,
                    out_map_png=out_map_png,
                    out_fence_png=out_fence_png,
                    max_edges=max(cfg.max_edges, 120_000),
                    max_wells_fence=cfg.max_wells_fence,
                    seed=int(s),
                    out_order_csv=out_order_csv,
                )
                fence_png_by_seed.append((int(s), out_fence_png))

        else:
            print("Step 3d skipped: missing framework_edges.csv and/or rep_arrays.npz.")

    # Optional: difference cross-sections (image diffs between fences across seeds)
    if cfg.write_diffs and len(fence_png_by_seed) >= 2:
        ref_seed = cfg.diff_ref_seed
        if ref_seed is None:
            ref_seed = fence_png_by_seed[0][0]

        ref_png = None
        for s, p in fence_png_by_seed:
            if s == ref_seed:
                ref_png = p
                break
        if ref_png is None:
            ref_seed, ref_png = fence_png_by_seed[0]

        for s, p in fence_png_by_seed:
            if s == ref_seed:
                continue
            out_diff = out_dir / f"step3d_fence_diff_seed{s}_vs_seed{ref_seed}_{cfg.diff_mode}.png"
            try:
                _write_diff_png(
                    ref_png=ref_png,
                    other_png=p,
                    out_png=out_diff,
                    mode=cfg.diff_mode,
                    clip_pctl=cfg.diff_clip_pctl,
                )
            except Exception as e:
                print(f"[WARN] Failed to write diff fence for seed={s} vs ref={ref_seed}: {e}")

    # 3e
    if "3e" in cfg.steps:
        if chron_npz and chron_npz.exists():
            plot_step3e_chronostrat(
                chronostrat_npz=chron_npz,
                out_png=out_dir / "step3e_chronostrat.png",
            )

    print(f"Wrote Step 3 viz to: {out_dir}")
    if "3d" in cfg.steps:
        print(f"Step 3d seeds: {list(cfg.seeds)}")
        if cfg.write_diffs and len(cfg.seeds) > 1:
            print(f"Diff fences: enabled (ref_seed={cfg.diff_ref_seed if cfg.diff_ref_seed is not None else cfg.seeds[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
