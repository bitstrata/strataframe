# src/strataframe/steps/step0_index_gr.py
from __future__ import annotations

import csv
import gc
import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from strataframe.curves.normalize_header import aliases_for, norm_mnemonic
from strataframe.io.csv import to_float, read_csv_rows
from strataframe.utils.fingerprint import sha1_file_prefix
from strataframe.utils.las_scan import compute_percentiles


# -----------------------------------------------------------------------------
# Step 0 config + paths
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Step0Config:
    curve_family: str = "GR"
    min_finite: int = 200
    pct: Tuple[float, float, float] = (1.0, 50.0, 99.0)

    # Legacy / compatibility with CLI + YAML (Step0 no longer uses lasio)
    quiet_lasio: bool = True

    # Progress reporting
    progress: bool = True
    progress_every: int = 250  # print + heartbeat every N wells (0 disables)

    # Streaming / memory control
    flush_every: int = 1000  # write parquet part every N OK wells (0 disables chunking)

    # Safety valves
    max_las_mb: int = 512          # skip files larger than this (defensive)
    reservoir_max: int = 20000     # max samples kept for percentile estimation (stream mode)

    # GC cadence (avoid per-well gc.collect() churn)
    gc_every: int = 300


@dataclass(frozen=True)
class Step0Paths:
    out_dir: Path

    @property
    def wells_gr_parquet(self) -> Path:
        # NOTE: directory dataset of part-*.parquet
        return self.out_dir / "wells_gr.parquet"

    @property
    def wells_gr_qc_csv(self) -> Path:
        return self.out_dir / "wells_gr_qc.csv"

    @property
    def manifest_json(self) -> Path:
        return self.out_dir / "manifest.json"

    @property
    def diagnostics_json(self) -> Path:
        return self.out_dir / "diagnostics.json"

    @property
    def progress_json(self) -> Path:
        return self.out_dir / "progress.json"

    @property
    def state_json(self) -> Path:
        return self.out_dir / "state.json"


# -----------------------------------------------------------------------------
# JSON + runtime stats
# -----------------------------------------------------------------------------

def _write_json_atomic(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as tf:
        tf.write(payload)
        tmp = tf.name
    os.replace(tmp, str(path))


def _now_utc_iso() -> str:
    import datetime as _dt
    return _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _rss_mb_and_fds() -> Tuple[Optional[float], Optional[int]]:
    """
    Best effort. If psutil isn't installed, returns (None, None).
    """
    try:
        import psutil  # type: ignore
        p = psutil.Process()
        rss = float(p.memory_info().rss) / (1024.0 * 1024.0)
        fds = int(getattr(p, "num_fds")()) if hasattr(p, "num_fds") else None
        return rss, fds
    except Exception:
        return None, None


def _fmt_eta(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return ""
    s = int(seconds)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# -----------------------------------------------------------------------------
# Manifest helpers
# -----------------------------------------------------------------------------

def _url_to_local_path(url: str, las_root: Path) -> Optional[Path]:
    u = (url or "").strip()
    if not u:
        return None
    name = u.split("/")[-1].strip()
    if not name:
        return None
    return Path(las_root) / name


def _api_nodash(api: str, api_num_nodash: str) -> str:
    if (api_num_nodash or "").strip():
        return (api_num_nodash or "").strip()
    return "".join(ch for ch in (api or "") if ch.isdigit())


# -----------------------------------------------------------------------------
# LAS header parsing (NO lasio)
# -----------------------------------------------------------------------------

def _parse_las_header_minimal(path: Path, *, max_header_bytes: int = 2_000_000) -> Dict[str, Any]:
    """
    Minimal LAS header parse:
      - WRAP (YES/NO)
      - NULL value (float or None)
      - curve mnemonics in order from ~C
    Stops once ~A is reached. Uses constant memory.
    """
    wrap = False
    null_value: Optional[float] = None
    curves: List[str] = []

    section = ""
    n_read = 0

    def _is_section(line: str) -> bool:
        s = line.lstrip()
        return s.startswith("~") and len(s) >= 2

    def _section_name(line: str) -> str:
        s = line.lstrip()
        # "~C" / "~Curve" etc
        t = s[1:].strip()
        return t[:1].upper() if t else ""

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            n_read += len(line.encode("utf-8", errors="ignore"))
            if n_read > max_header_bytes:
                # header is absurdly large; bail (still safe)
                break

            s = line.strip()
            if not s or s.startswith("#"):
                continue

            if _is_section(line):
                sec = _section_name(line)
                section = sec
                if sec == "A":
                    break
                continue

            # only parse within relevant sections
            if section == "W":
                # e.g. "WRAP.     YES : Wrapped"
                #      "NULL.  -999.25 : Null value"
                left = s.split(":", 1)[0].strip()
                if "." in left:
                    key, rest = left.split(".", 1)
                    key_n = key.strip().upper()
                    val = rest.strip().split()[0] if rest.strip() else ""
                    if key_n == "WRAP":
                        wrap = val.upper().startswith("Y")
                    elif key_n == "NULL":
                        try:
                            null_value = float(val)
                        except Exception:
                            null_value = None

            elif section == "C":
                # curve line like "DEPT .M : Depth"
                left = s.split(":", 1)[0].strip()
                if "." in left:
                    key = left.split(".", 1)[0].strip()
                else:
                    key = left.split()[0].strip()
                if key and not key.startswith("~"):
                    curves.append(key)

    return {"wrap": wrap, "null": null_value, "curves": curves}


def _choose_depth_index(curves: List[str]) -> Optional[int]:
    """
    Choose the depth curve index from header mnemonics.

    Note: norm_mnemonic("DEPTH") -> "DEPT" in this codebase, so we check canonical tokens.
    """
    for i, mn in enumerate(curves):
        n = norm_mnemonic(mn)
        if n in {"DEPT", "MD"}:
            return i
    return None


def _choose_gr_index(curves: List[str], fam_can: set[str]) -> Optional[int]:
    # Prefer exact GR first, else first family match
    for i, mn in enumerate(curves):
        if norm_mnemonic(mn) == "GR":
            return i
    for i, mn in enumerate(curves):
        if norm_mnemonic(mn) in fam_can:
            return i
    return None


# -----------------------------------------------------------------------------
# Streaming ASCII scan of ~A (supports wrapped + unwrapped)
# -----------------------------------------------------------------------------

def _reservoir_add(sample: List[float], x: float, n_seen: int, k: int) -> None:
    if k <= 0:
        return
    if len(sample) < k:
        sample.append(x)
        return
    # deterministic pseudo-random index based on n_seen
    j = (1103515245 * n_seen + 12345) & 0x7FFFFFFF
    idx = j % n_seen
    if idx < k:
        sample[idx] = x


def _scan_las_ascii_depth_gr(
    path: Path,
    *,
    n_curves: int,
    depth_idx: int,
    gr_idx: int,
    wrapped: bool,
    null_value: Optional[float],
    pct: Tuple[float, float, float],
    reservoir_max: int,
) -> Tuple[int, int, float, float, float, float, float, float]:
    """
    Stream parse ~A section for depth + GR.
    Returns:
      (n_total_rows, n_finite_gr, finite_frac, depth_min, depth_max, p01, p50, p99)

    For wrapped files, tokens are accumulated and grouped into rows of n_curves.
    """
    in_data = False
    n_total = 0
    n_fin = 0

    dmin = np.inf
    dmax = -np.inf

    sample: List[float] = []
    max_idx = max(depth_idx, gr_idx)

    token_buf: List[str] = []

    def _process_row(tokens: List[str]) -> None:
        nonlocal n_total, n_fin, dmin, dmax

        if len(tokens) <= max_idx:
            return

        # depth
        try:
            d = float(tokens[depth_idx])
        except Exception:
            d = np.nan
        if np.isfinite(d):
            if d < dmin:
                dmin = d
            if d > dmax:
                dmax = d

        # gr
        try:
            g = float(tokens[gr_idx])
        except Exception:
            g = np.nan
        if null_value is not None and np.isfinite(g) and g == null_value:
            g = np.nan

        n_total += 1
        if np.isfinite(g):
            n_fin += 1
            _reservoir_add(sample, float(g), n_fin, reservoir_max)

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not in_data:
                s0 = line.lstrip()
                if s0.upper().startswith("~A"):
                    in_data = True
                continue

            s = line.strip()
            if not s or s.startswith("#"):
                continue

            # Make splitting robust: treat commas as whitespace
            s = s.replace(",", " ")
            parts = s.split()
            if not parts:
                continue

            if wrapped:
                token_buf.extend(parts)
                while len(token_buf) >= n_curves:
                    row = token_buf[:n_curves]
                    del token_buf[:n_curves]
                    _process_row(row)
            else:
                _process_row(parts)

    frac = float(n_fin / max(1, n_total))

    if n_fin <= 0 or not sample:
        return n_total, n_fin, frac, np.nan, np.nan, np.nan, np.nan, np.nan

    arr = np.asarray(sample, dtype="float64")
    p01, p50, p99 = compute_percentiles(arr, pct)

    depth_min = float(dmin) if np.isfinite(dmin) else np.nan
    depth_max = float(dmax) if np.isfinite(dmax) else np.nan
    return n_total, n_fin, frac, depth_min, depth_max, float(p01), float(p50), float(p99)


# -----------------------------------------------------------------------------
# Step 0 runner
# -----------------------------------------------------------------------------

def run_step0_index_gr(
    *,
    ks_manifest_path: Path,
    las_root: Path,
    out_dir: Path,
    cfg: Step0Config,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Step 0:
      - Read manifest
      - Parse LAS header text (no lasio)
      - Stream parse ~A for depth + one GR curve
      - Enforce min_finite
      - Write parquet parts + QC CSV + diagnostics + progress/state JSON

    Key robustness:
      - state.json is updated BEFORE touching each LAS file so you can pinpoint the killer file
        even if the OS SIGKILLs the process mid-parse.
      - resume works if state.json exists and outputs exist (run again without --overwrite).
    """
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception as e:
        raise RuntimeError("pyarrow is required for streaming parquet output. Install: pip install pyarrow") from e

    paths = Step0Paths(out_dir=out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wells_gr_ds = paths.wells_gr_parquet
    qc_csv_path = paths.wells_gr_qc_csv

    manifest_rows = read_csv_rows(Path(ks_manifest_path))
    if not manifest_rows:
        raise ValueError(f"No rows in ks manifest: {ks_manifest_path}")

    fam = norm_mnemonic(cfg.curve_family)
    fam_aliases = aliases_for(fam) if fam else [cfg.curve_family]
    fam_can = {norm_mnemonic(a) for a in fam_aliases if norm_mnemonic(a)}
    if not fam_can:
        fam_can = {fam} if fam else {"GR"}

    # -------------------------------------------------------------------------
    # Resume / overwrite
    # -------------------------------------------------------------------------
    start_idx = 0
    part_idx = 0
    existing_counts: Optional[Dict[str, int]] = None

    if overwrite:
        if wells_gr_ds.exists():
            if wells_gr_ds.is_dir():
                shutil.rmtree(wells_gr_ds)
            else:
                wells_gr_ds.unlink()
        for p in (qc_csv_path, paths.manifest_json, paths.diagnostics_json, paths.progress_json, paths.state_json):
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass

    if (not overwrite) and wells_gr_ds.exists() and paths.state_json.exists():
        st = json.loads(paths.state_json.read_text(encoding="utf-8"))
        start_idx = int(st.get("next_row", 0))
        part_idx = int(st.get("part_idx", 0))
        existing_counts = st.get("counts", None)
    elif (not overwrite) and (wells_gr_ds.exists() or qc_csv_path.exists()):
        raise RuntimeError(
            f"Step0 outputs already exist in {out_dir} but no state.json found to resume. "
            f"Use --overwrite or delete the run directory."
        )

    wells_gr_ds.mkdir(parents=True, exist_ok=True)

    # If part_idx is missing/wrong, derive from disk
    try:
        parts_on_disk = sorted(wells_gr_ds.glob("part-*.parquet"))
        part_idx = max(part_idx, len(parts_on_disk))
    except Exception:
        pass

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------
    diag: Dict[str, Any] = {
        "step": "step0_index_gr",
        "inputs": {
            "ks_manifest_path": str(ks_manifest_path),
            "las_root": str(las_root),
            "ks_manifest_fingerprint": sha1_file_prefix(Path(ks_manifest_path)),
        },
        "config": asdict(cfg),
        "counts": {
            "n_rows_in": int(len(manifest_rows)),
            "n_missing_las": 0,
            "n_header_fail": 0,
            "n_no_depth": 0,
            "n_no_gr": 0,
            "n_skip_too_large": 0,
            "n_stream_fail": 0,
            "n_insufficient_gr": 0,
            "n_keep": 0,
        },
        "gr_family_canons": sorted(fam_can),
        "examples": {"failures": []},
        "outputs": {
            "parquet_dataset_dir": str(wells_gr_ds),
            "qc_csv": str(qc_csv_path),
            "parquet_parts_written": int(part_idx),
        },
        "resume": {"start_row": int(start_idx), "initial_part_idx": int(part_idx)},
    }

    # Restore counts when resuming so progress output stays meaningful
    if existing_counts:
        try:
            for k, v in existing_counts.items():
                if k in diag["counts"]:
                    diag["counts"][k] = int(v)
        except Exception:
            pass

    qc_fields = [
        "well_id", "url", "las_path",
        "status", "reason",
        "lat", "lon",
        "engine_used", "wrapped",
        "las_size_mb",
        "depth_source", "depth_min", "depth_max",
        "gr_mnemonic_used", "gr_candidates",
        "gr_n_total", "gr_n_finite", "gr_finite_frac",
        "gr_p01", "gr_p50", "gr_p99",
        "mode",
    ]

    keep_fields = [
        "well_id", "url", "las_path",
        "lat", "lon",
        "depth_source", "depth_min", "depth_max",
        "gr_mnemonic_used",
        "gr_n_finite", "gr_finite_frac",
        "gr_p01", "gr_p50", "gr_p99",
        "engine_used", "wrapped",
        "mode",
    ]

    # QC CSV writer (append if resuming)
    qc_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if qc_csv_path.exists() and start_idx > 0:
        qc_f = qc_csv_path.open("a", encoding="utf-8", newline="")
        qc_w = csv.DictWriter(qc_f, fieldnames=qc_fields)
    else:
        qc_f = qc_csv_path.open("w", encoding="utf-8", newline="")
        qc_w = csv.DictWriter(qc_f, fieldnames=qc_fields)
        qc_w.writeheader()

    keep_buf: List[Dict[str, Any]] = []

    def _flush_keep_buf(force: bool) -> None:
        nonlocal keep_buf, part_idx
        if not keep_buf:
            return
        flush_every = int(cfg.flush_every or 0)
        if (not force) and flush_every > 0 and len(keep_buf) < flush_every:
            return

        part_path = wells_gr_ds / f"part-{part_idx:05d}.parquet"
        part_idx += 1

        rows = [{k: r.get(k, None) for k in keep_fields} for r in keep_buf]
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, part_path, compression="snappy")

        diag["outputs"]["parquet_parts_written"] = int(part_idx)
        diag["counts"]["n_keep"] += int(len(keep_buf))

        keep_buf.clear()
        qc_f.flush()
        del table
        gc.collect()

    n_total = int(len(manifest_rows))
    t0 = time.perf_counter()

    def _write_progress(done: int, keep_est: int, last_well_id: str, last_las_path: str, phase: str) -> None:
        elapsed = max(1e-9, (time.perf_counter() - t0))
        rate = float((done - start_idx) / elapsed) if done > start_idx else 0.0
        eta = float((n_total - done) / rate) if rate > 0 else float("nan")
        rss_mb, fds = _rss_mb_and_fds()

        _write_json_atomic(
            paths.progress_json,
            {
                "ts_utc": _now_utc_iso(),
                "step": "0",
                "name": "step0_index_gr",
                "done": int(done),
                "total": int(n_total),
                "keep_est": int(keep_est),
                "parts_written": int(part_idx),
                "elapsed_s": float(elapsed),
                "rate_per_s": float(rate),
                "eta_s": float(eta) if np.isfinite(eta) else None,
                "rss_mb": rss_mb,
                "num_fds": fds,
                "counts": dict(diag["counts"]),
                "phase": phase,  # "pre" before opening LAS, "post" after row finished
                "last_well_id": str(last_well_id),
                "last_las_path": str(last_las_path),
                "out_dir": str(out_dir),
                "status": "running",
            },
        )

        _write_json_atomic(
            paths.state_json,
            {
                "ts_utc": _now_utc_iso(),
                "next_row": int(done),
                "part_idx": int(part_idx),
                "counts": dict(diag["counts"]),
                "phase": phase,
                "last_well_id": str(last_well_id),
                "last_las_path": str(last_las_path),
            },
        )

    if bool(cfg.progress):
        print(f"[step0] scanning {n_total} manifest rows…", flush=True)
        print(f"[step0] ks_manifest={ks_manifest_path}  las_root={las_root}", flush=True)
        print(
            f"[step0] curve_family={cfg.curve_family} min_finite={cfg.min_finite} "
            f"flush_every={cfg.flush_every} max_las_mb={cfg.max_las_mb}",
            flush=True,
        )
        if start_idx > 0:
            print(f"[step0] RESUME: starting at row {start_idx}  part_idx={part_idx}", flush=True)

    _write_progress(start_idx, keep_est=int(diag["counts"]["n_keep"]), last_well_id="", last_las_path="", phase="init")

    try:
        for i in range(start_idx, n_total):
            mr = manifest_rows[i]

            url = str(mr.get("URL", "") or "").strip()
            api = str(mr.get("API", "") or "").strip()
            api_num_nodash = str(mr.get("API_NUM_NODASH", "") or "").strip()
            kgs_id = str(mr.get("KGS_ID", "") or "").strip()
            well_id = _api_nodash(api, api_num_nodash) or kgs_id or api

            las_path = _url_to_local_path(url, Path(las_root))
            las_path_s = str(las_path) if las_path is not None else ""

            # IMPORTANT: checkpoint BEFORE touching the LAS file
            try:
                keep_est_pre = int(diag["counts"]["n_keep"]) + int(len(keep_buf))
                _write_progress(i, keep_est_pre, well_id, las_path_s, phase="pre")
            except Exception:
                pass

            lat = to_float(str(mr.get("Latitude", "") or ""))
            lon = to_float(str(mr.get("Longitude", "") or ""))

            qc: Dict[str, Any] = {
                "well_id": well_id,
                "url": url,
                "las_path": las_path_s,
                "status": "",
                "reason": "",
                "lat": "" if lat is None else f"{float(lat):.8f}",
                "lon": "" if lon is None else f"{float(lon):.8f}",
                "engine_used": "text",
                "wrapped": "",
                "las_size_mb": "",
                "depth_source": "",
                "depth_min": "",
                "depth_max": "",
                "gr_mnemonic_used": "",
                "gr_candidates": "",
                "gr_n_total": "",
                "gr_n_finite": "",
                "gr_finite_frac": "",
                "gr_p01": "",
                "gr_p50": "",
                "gr_p99": "",
                "mode": "stream_text",
            }

            if las_path is None or (not las_path.exists()) or (not las_path.is_file()):
                qc["status"] = "MISSING_LAS"
                qc["reason"] = "LAS path missing or file not found"
                diag["counts"]["n_missing_las"] += 1
                qc_w.writerow({k: qc.get(k, "") for k in qc_fields})
            else:
                # size guard
                try:
                    size_mb = float(las_path.stat().st_size) / (1024.0 * 1024.0)
                except Exception:
                    size_mb = float("nan")
                qc["las_size_mb"] = "" if not np.isfinite(size_mb) else f"{size_mb:.2f}"

                if np.isfinite(size_mb) and size_mb > float(cfg.max_las_mb):
                    qc["status"] = "SKIP_TOO_LARGE"
                    qc["reason"] = f"LAS size {size_mb:.2f} MB > max_las_mb {cfg.max_las_mb}"
                    diag["counts"]["n_skip_too_large"] += 1
                    qc_w.writerow({k: qc.get(k, "") for k in qc_fields})
                else:
                    try:
                        hdr = _parse_las_header_minimal(las_path)
                        curves = list(hdr.get("curves") or [])
                        wrapped = bool(hdr.get("wrap", False))
                        null_value = hdr.get("null", None)
                        qc["wrapped"] = str(bool(wrapped))
                    except Exception as e:
                        qc["status"] = "HEADER_FAIL"
                        qc["reason"] = f"{type(e).__name__}: {e}"
                        diag["counts"]["n_header_fail"] += 1
                        if len(diag["examples"]["failures"]) < 20:
                            diag["examples"]["failures"].append(
                                {"well_id": well_id, "status": qc["status"], "error": qc["reason"], "las_path": las_path_s}
                            )
                        qc_w.writerow({k: qc.get(k, "") for k in qc_fields})
                    else:
                        depth_idx = _choose_depth_index(curves)
                        gr_idx = _choose_gr_index(curves, fam_can)

                        gr_candidates = [mn for mn in curves if norm_mnemonic(mn) in fam_can]
                        qc["gr_candidates"] = " ".join(gr_candidates)

                        if depth_idx is None:
                            qc["status"] = "NO_DEPTH"
                            qc["reason"] = "No DEPT/MD curve found in header (~C)"
                            diag["counts"]["n_no_depth"] += 1
                            qc_w.writerow({k: qc.get(k, "") for k in qc_fields})
                        elif gr_idx is None:
                            qc["status"] = "NO_GR"
                            qc["reason"] = "No GR-family curve found in header (~C)"
                            diag["counts"]["n_no_gr"] += 1
                            qc_w.writerow({k: qc.get(k, "") for k in qc_fields})
                        else:
                            gr_mn = curves[gr_idx]
                            qc["gr_mnemonic_used"] = str(gr_mn)

                            try:
                                n_total_rows, n_fin, frac, dmin, dmax, p01, p50, p99 = _scan_las_ascii_depth_gr(
                                    las_path,
                                    n_curves=len(curves),
                                    depth_idx=int(depth_idx),
                                    gr_idx=int(gr_idx),
                                    wrapped=bool(wrapped),
                                    null_value=null_value if isinstance(null_value, (int, float)) else None,
                                    pct=cfg.pct,
                                    reservoir_max=int(cfg.reservoir_max),
                                )
                            except Exception as e:
                                qc["status"] = "STREAM_FAIL"
                                qc["reason"] = f"{type(e).__name__}: {e}"
                                diag["counts"]["n_stream_fail"] += 1
                                if len(diag["examples"]["failures"]) < 20:
                                    diag["examples"]["failures"].append(
                                        {"well_id": well_id, "status": qc["status"], "error": qc["reason"], "las_path": las_path_s}
                                    )
                                qc_w.writerow({k: qc.get(k, "") for k in qc_fields})
                            else:
                                qc["depth_source"] = "ASCII(~A)"
                                qc["gr_n_total"] = str(int(n_total_rows))
                                qc["gr_n_finite"] = str(int(n_fin))
                                qc["gr_finite_frac"] = f"{float(frac):.6f}"

                                qc["depth_min"] = "" if not np.isfinite(dmin) else f"{float(dmin):.3f}"
                                qc["depth_max"] = "" if not np.isfinite(dmax) else f"{float(dmax):.3f}"

                                qc["gr_p01"] = "" if not np.isfinite(p01) else f"{float(p01):.6g}"
                                qc["gr_p50"] = "" if not np.isfinite(p50) else f"{float(p50):.6g}"
                                qc["gr_p99"] = "" if not np.isfinite(p99) else f"{float(p99):.6g}"

                                if int(n_fin) < int(cfg.min_finite):
                                    qc["status"] = "INSUFFICIENT_GR"
                                    qc["reason"] = f"GR finite samples {n_fin} < min_finite {cfg.min_finite}"
                                    diag["counts"]["n_insufficient_gr"] += 1
                                    qc_w.writerow({k: qc.get(k, "") for k in qc_fields})
                                else:
                                    qc["status"] = "OK"
                                    qc_w.writerow({k: qc.get(k, "") for k in qc_fields})

                                    keep_buf.append(
                                        {
                                            "well_id": well_id,
                                            "url": url,
                                            "las_path": las_path_s,
                                            "lat": np.nan if lat is None else float(lat),
                                            "lon": np.nan if lon is None else float(lon),
                                            "depth_source": qc["depth_source"],
                                            "depth_min": float(qc["depth_min"]) if qc["depth_min"] else np.nan,
                                            "depth_max": float(qc["depth_max"]) if qc["depth_max"] else np.nan,
                                            "gr_mnemonic_used": str(gr_mn),
                                            "gr_n_finite": int(n_fin),
                                            "gr_finite_frac": float(frac),
                                            "gr_p01": float(p01) if np.isfinite(p01) else np.nan,
                                            "gr_p50": float(p50) if np.isfinite(p50) else np.nan,
                                            "gr_p99": float(p99) if np.isfinite(p99) else np.nan,
                                            "engine_used": "text",
                                            "wrapped": bool(wrapped),
                                            "mode": "stream_text",
                                        }
                                    )
                                    _flush_keep_buf(force=False)

            # progress tick
            if bool(cfg.progress):
                every = int(cfg.progress_every or 0)
                done = i + 1
                if every > 0 and (done == start_idx + 1 or done % every == 0 or done == n_total):
                    elapsed = max(1e-9, (time.perf_counter() - t0))
                    rate = (done - start_idx) / elapsed if done > start_idx else 0.0
                    eta = (n_total - done) / rate if rate > 0 else float("nan")

                    c = diag["counts"]
                    keep_est = int(c["n_keep"]) + int(len(keep_buf))
                    parts = int(part_idx)

                    print(
                        f"[step0] {done}/{n_total} "
                        f"keep~={keep_est} miss={c['n_missing_las']} "
                        f"hdr_fail={c['n_header_fail']} no_depth={c['n_no_depth']} no_gr={c['n_no_gr']} "
                        f"skip_big={c['n_skip_too_large']} stream_fail={c['n_stream_fail']} "
                        f"insuf={c['n_insufficient_gr']} parts={parts} "
                        f"rate={rate:.2f}/s eta={_fmt_eta(eta)}",
                        flush=True,
                    )

                    try:
                        _write_progress(done, keep_est, well_id, las_path_s, phase="post")
                    except Exception:
                        pass

                    ge = int(cfg.gc_every or 0)
                    if ge > 0 and (done % ge == 0):
                        gc.collect()

        _flush_keep_buf(force=True)

    finally:
        try:
            qc_f.flush()
            qc_f.close()
        except Exception:
            pass

    manifest = {
        "step": "step0_index_gr",
        "inputs": {
            "ks_manifest_path": str(ks_manifest_path),
            "las_root": str(las_root),
            "ks_manifest_fingerprint": diag["inputs"]["ks_manifest_fingerprint"],
        },
        "outputs": {
            "wells_gr_parquet": str(paths.wells_gr_parquet),
            "wells_gr_qc_csv": str(paths.wells_gr_qc_csv),
            "diagnostics_json": str(paths.diagnostics_json),
        },
        "config": asdict(cfg),
    }
    paths.manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    paths.diagnostics_json.write_text(json.dumps(diag, indent=2), encoding="utf-8")

    try:
        rss_mb, fds = _rss_mb_and_fds()
        _write_json_atomic(
            paths.progress_json,
            {
                "ts_utc": _now_utc_iso(),
                "step": "0",
                "name": "step0_index_gr",
                "done": int(n_total),
                "total": int(n_total),
                "keep_est": int(diag["counts"]["n_keep"]),
                "parts_written": int(part_idx),
                "rss_mb": rss_mb,
                "num_fds": fds,
                "counts": dict(diag["counts"]),
                "out_dir": str(out_dir),
                "status": "done",
            },
        )
        _write_json_atomic(
            paths.state_json,
            {
                "ts_utc": _now_utc_iso(),
                "next_row": int(n_total),
                "part_idx": int(part_idx),
                "counts": dict(diag["counts"]),
                "phase": "done",
            },
        )
    except Exception:
        pass

    return diag
