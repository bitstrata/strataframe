# src/strataframe/preprocess/normalize_las_header.py
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Iterable

# Optional pandas (nice output). Module works without it.
try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

# New shared IO
from strataframe.io import parse_las_curve_section, read_csv_rows, write_csv

# Shared curve-header canonicalization policy (single source of truth)
from strataframe.curves.normalize_header import (
    MNEMONIC_ALIASES,
    UNIT_ALIASES,
    ALLOWED_UNITS,
    aliases_for,      # not used in this file yet, but imported for parity / future-proofing
    norm_mnemonic,
    norm_unit,
    canon_key,
)


# =============================================================================
# Configuration: normalization tables + quarantine rules
# =============================================================================

# “Impossible” pair heuristics
IMPOSSIBLE_PAIRS: Sequence[Tuple[str, str]] = (
    ("DCAL", "GAPI"),
    ("MCAL", "GAPI"),
    ("CALI", "GAPI"),
    ("DEPT", "GAPI"),
    ("DEPT", "MV"),
    ("GR", "IN"),
    ("GR", "OHM-M"),
)


# =============================================================================
# Normalization helpers
# =============================================================================

_WS_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^A-Z0-9_/.\-]+")
_MNEM_SUFFIX_RE = re.compile(r":\d+$")


def _clean_token(x: Optional[str]) -> str:
    if x is None:
        return ""
    x = x.strip()
    if not x:
        return ""
    x = _WS_RE.sub("", x)
    x = x.upper()
    x = _NON_ALNUM_RE.sub("", x)
    return x


def norm_mnemonic(mnemonic: Optional[str]) -> str:
    m = _clean_token(mnemonic)
    if not m:
        return ""
    m = _MNEM_SUFFIX_RE.sub("", m)
    return MNEMONIC_ALIASES.get(m, m)


def norm_unit(unit: Optional[str]) -> str:
    u = _clean_token(unit)
    if not u:
        return ""
    u = u.replace("\\", "/")

    # fast-path canonicalizations (kept for backward compatibility)
    if u in {"OHMM", "OHM/M", "OHM.M"}:
        u = "OHM-M"
    if u in {"G/C3", "GM/C3", "GM/CC", "G/CM3", "GM/CM3"}:
        u = "G/CC"
    if u in {"USEC/FT", "US/F"}:
        u = "US/FT"
    if u in {"INCH", "INCHES"}:
        u = "IN"

    return UNIT_ALIASES.get(u, u)


def canon_key(mnemonic: Optional[str], unit: Optional[str]) -> Tuple[str, str, str]:
    mn = norm_mnemonic(mnemonic)
    un = norm_unit(unit)
    return mn, un, f"{mn}|{un}"


def api_nodash(api: str, api_num_nodash: str) -> str:
    if api_num_nodash.strip():
        return api_num_nodash.strip()
    return "".join(ch for ch in (api or "") if ch.isdigit())


def read_ks_manifest(path: Path) -> List[Dict[str, str]]:
    """
    Reads data/ks_las_files.txt (quoted CSV) and returns rows as dicts.
    Kept local because it's schema-specific to the KGS manifest.
    """
    with path.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        if rdr.fieldnames is None:
            raise ValueError(f"No header row found in {path}")
        rows: List[Dict[str, str]] = []
        for r in rdr:
            rows.append({k: (v if v is not None else "") for k, v in r.items()})
        return rows


def url_to_local_path(url: str, las_root: Path) -> Path:
    """
    Deterministic mapping from KGS URL to local file path.
    Default: use trailing filename from URL.
    """
    name = url.strip().split("/")[-1]
    return las_root / name


# =============================================================================
# Quarantine logic
# =============================================================================

@dataclass(frozen=True)
class QuarantineHit:
    url: str
    file: str
    kgs_id: str
    api_num_nodash: str
    operator: str
    lease: str

    mnemonic: str
    unit: str
    descr: str
    mnemonic_n: str
    unit_n: str

    severity: str  # INFO/WARN/ERROR
    reason: str


def check_quarantine(
    *,
    url: str,
    file: str,
    kgs_id: str,
    api_num_nodash: str,
    operator: str,
    lease: str,
    mnemonic: str,
    unit: str,
    descr: str,
    mnemonic_n: str,
    unit_n: str,
) -> List[QuarantineHit]:
    hits: List[QuarantineHit] = []

    # 1) impossible pairs => ERROR
    for (mn_bad, un_bad) in IMPOSSIBLE_PAIRS:
        if mnemonic_n == mn_bad and unit_n == un_bad:
            hits.append(
                QuarantineHit(
                    url=url,
                    file=file,
                    kgs_id=kgs_id,
                    api_num_nodash=api_num_nodash,
                    operator=operator,
                    lease=lease,
                    mnemonic=mnemonic,
                    unit=unit,
                    descr=descr,
                    mnemonic_n=mnemonic_n,
                    unit_n=unit_n,
                    severity="ERROR",
                    reason=f"Impossible mnemonic+unit: {mnemonic_n} with {unit_n}",
                )
            )

    # 2) unit not in allowed list
    allowed = ALLOWED_UNITS.get(mnemonic_n)
    if allowed is not None:
        if unit_n not in allowed:
            sev = "WARN"
            if unit_n == "" and "" in allowed:
                sev = "INFO"
            hits.append(
                QuarantineHit(
                    url=url,
                    file=file,
                    kgs_id=kgs_id,
                    api_num_nodash=api_num_nodash,
                    operator=operator,
                    lease=lease,
                    mnemonic=mnemonic,
                    unit=unit,
                    descr=descr,
                    mnemonic_n=mnemonic_n,
                    unit_n=unit_n,
                    severity=sev,
                    reason=f"Unit not allowed for mnemonic: {mnemonic_n} has {unit_n}; allowed={list(allowed)}",
                )
            )
    else:
        hits.append(
            QuarantineHit(
                url=url,
                file=file,
                kgs_id=kgs_id,
                api_num_nodash=api_num_nodash,
                operator=operator,
                lease=lease,
                mnemonic=mnemonic,
                unit=unit,
                descr=descr,
                mnemonic_n=mnemonic_n,
                unit_n=unit_n,
                severity="INFO",
                reason="Unknown mnemonic (no allowlist entry); consider adding to MNEMONIC_ALIASES/ALLOWED_UNITS if expected.",
            )
        )

    # 3) Description-based sanity (lightweight)
    d = (descr or "").strip().lower()
    if mnemonic_n == "GR" and unit_n in {"OHM-M", "IN"}:
        hits.append(
            QuarantineHit(
                url=url,
                file=file,
                kgs_id=kgs_id,
                api_num_nodash=api_num_nodash,
                operator=operator,
                lease=lease,
                mnemonic=mnemonic,
                unit=unit,
                descr=descr,
                mnemonic_n=mnemonic_n,
                unit_n=unit_n,
                severity="ERROR",
                reason="Gamma ray curve carrying resistivity/caliper units.",
            )
        )
    if mnemonic_n in {"CALI", "DCAL", "MCAL"} and ("gamma" in d or "gr" in d) and unit_n == "GAPI":
        hits.append(
            QuarantineHit(
                url=url,
                file=file,
                kgs_id=kgs_id,
                api_num_nodash=api_num_nodash,
                operator=operator,
                lease=lease,
                mnemonic=mnemonic,
                unit=unit,
                descr=descr,
                mnemonic_n=mnemonic_n,
                unit_n=unit_n,
                severity="ERROR",
                reason="Caliper mnemonic but description suggests gamma ray.",
            )
        )

    return hits


def quarantine_report(qhits: List[QuarantineHit]) -> List[Dict[str, object]]:
    sev_rank = {"ERROR": 0, "WARN": 1, "INFO": 2}
    qhits_sorted = sorted(
        qhits,
        key=lambda h: (sev_rank.get(h.severity, 9), h.url, h.mnemonic_n, h.unit_n),
    )

    out: List[Dict[str, object]] = []
    for h in qhits_sorted:
        out.append(
            {
                "severity": h.severity,
                "reason": h.reason,
                "url": h.url,
                "file": h.file,
                "kgs_id": h.kgs_id,
                "api_num_nodash": h.api_num_nodash,
                "operator": h.operator,
                "lease": h.lease,
                "mnemonic": h.mnemonic,
                "unit": h.unit,
                "descr": h.descr,
                "mnemonic_n": h.mnemonic_n,
                "unit_n": h.unit_n,
            }
        )
    return out


# =============================================================================
# Core transforms (manifest mode + legacy CSV mode)
# =============================================================================

def normalize_curve_index_rows(rows: List[Dict[str, str]]) -> Tuple[List[Dict[str, object]], List[QuarantineHit]]:
    """
    Legacy mode input schema: file,mnemonic,unit,descr
    Legacy mode does not have URL/well context; quarantine will use blank context fields.
    """
    out_rows: List[Dict[str, object]] = []
    qhits: List[QuarantineHit] = []

    for r in rows:
        f = r.get("file", "") or ""
        mn = r.get("mnemonic", "") or ""
        un = r.get("unit", "") or ""
        ds = r.get("descr", "") or ""
        mn_n, un_n, key = canon_key(mn, un)

        out_rows.append(
            {
                "file": f,
                "mnemonic": mn,
                "unit": un,
                "descr": ds,
                "mnemonic_n": mn_n,
                "unit_n": un_n,
                "canon_key": key,
            }
        )

        qhits.extend(
            check_quarantine(
                url="",
                file=f,
                kgs_id="",
                api_num_nodash="",
                operator="",
                lease="",
                mnemonic=mn,
                unit=un,
                descr=ds,
                mnemonic_n=mn_n,
                unit_n=un_n,
            )
        )

    return out_rows, qhits


def normalize_vocal_counts_rows(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    """
    Input schema: mnemonic,unit,count
    Output aggregated by (mnemonic_n, unit_n).
    """
    normed: List[Dict[str, object]] = []
    for r in rows:
        mn = r.get("mnemonic", "") or ""
        un = r.get("unit", "") or ""
        cnt_raw = r.get("count", "") or "0"
        try:
            cnt = int(float(cnt_raw))
        except Exception:
            cnt = 0

        mn_n, un_n, key = canon_key(mn, un)
        normed.append(
            {
                "mnemonic": mn,
                "unit": un,
                "count": cnt,
                "mnemonic_n": mn_n,
                "unit_n": un_n,
                "canon_key": key,
            }
        )

    agg: Dict[Tuple[str, str], int] = {}
    for r in normed:
        k = (str(r["mnemonic_n"]), str(r["unit_n"]))
        agg[k] = agg.get(k, 0) + int(r["count"])

    out = [
        {"mnemonic_n": mn, "unit_n": un, "count": cnt, "canon_key": f"{mn}|{un}"}
        for (mn, un), cnt in sorted(agg.items(), key=lambda x: x[1], reverse=True)
    ]
    return out


def summarize_by_group(rows: List[Dict[str, object]], group_key: str) -> List[Dict[str, object]]:
    """
    Produces one row per group listing canonical curve keys and counts.
    group_key is typically "url" (manifest mode) or "file" (legacy mode).
    """
    by_g: Dict[str, List[str]] = {}
    for r in rows:
        g = str(r.get(group_key, "") or "")
        k = str(r.get("canon_key", "") or "")
        if not g or not k:
            continue
        by_g.setdefault(g, []).append(k)

    out: List[Dict[str, object]] = []
    for g, keys in sorted(by_g.items()):
        uniq = sorted(set(keys))
        out.append(
            {
                group_key: g,
                "n_curves": len(keys),
                "n_unique_canon": len(uniq),
                "canon_keys": " ".join(uniq),
            }
        )
    return out


def _quick_stats(summary_rows: List[Dict[str, object]], group_key: str) -> Dict[str, int]:
    total = len(summary_rows)
    only_dept_gr = 0
    for r in summary_rows:
        keys = str(r.get("canon_keys", "")).split()
        if len(keys) != 2:
            continue
        has_dept = any(k.startswith("DEPT|") for k in keys)
        has_gr = any(k.startswith("GR|") for k in keys)
        if has_dept and has_gr:
            only_dept_gr += 1
    return {"total_groups": total, "only_dept_gr": only_dept_gr}


def build_from_manifest(
    manifest_rows: List[Dict[str, str]],
    *,
    las_root: Path,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[QuarantineHit]]:
    """
    Returns:
      - well_index_rows (one per URL row in manifest, with status)
      - curve_index_rows (one per URL per curve)
      - quarantine hits
    """
    well_index: List[Dict[str, object]] = []
    curve_index: List[Dict[str, object]] = []
    qhits: List[QuarantineHit] = []

    for mr in manifest_rows:
        url = (mr.get("URL", "") or "").strip()
        if not url:
            continue

        kgs_id = (mr.get("KGS_ID", "") or "").strip()
        operator = (mr.get("Operator", "") or "").strip()
        lease = (mr.get("Lease", "") or "").strip()
        api = (mr.get("API", "") or "").strip()
        api_nd = api_nodash(api, (mr.get("API_NUM_NODASH", "") or "").strip())
        lat = (mr.get("Latitude", "") or "").strip()
        lon = (mr.get("Longitude", "") or "").strip()
        location = (mr.get("Location", "") or "").strip()
        elev = (mr.get("Elevation", "") or "").strip()
        elev_ref = (mr.get("Elev_Ref", "") or "").strip()
        depth_start = (mr.get("Depth_start", "") or "").strip()
        depth_stop = (mr.get("Depth_stop", "") or "").strip()

        local_path = url_to_local_path(url, las_root)
        status = "OK"
        parse_warnings: List[str] = []
        curves_raw: List[Dict[str, str]] = []

        if not local_path.exists():
            status = "MISSING_LOCAL"
            parse_warnings = [f"Local LAS not found: {local_path}"]
        else:
            try:
                curves_raw, parse_warnings = parse_las_curve_section(local_path)
            except Exception as e:
                status = "PARSE_ERROR"
                parse_warnings = [f"Failed to parse LAS header: {type(e).__name__}: {e}"]
                curves_raw = []

        well_index.append(
            {
                "url": url,
                "file": str(local_path),
                "status": status,
                "parse_warnings": " | ".join(parse_warnings),
                "kgs_id": kgs_id,
                "api": api,
                "api_num_nodash": api_nd,
                "operator": operator,
                "lease": lease,
                "latitude": lat,
                "longitude": lon,
                "location": location,
                "elevation": elev,
                "elev_ref": elev_ref,
                "depth_start": depth_start,
                "depth_stop": depth_stop,
            }
        )

        for c in curves_raw:
            mn = c.get("mnemonic", "") or ""
            un = c.get("unit", "") or ""
            ds = c.get("descr", "") or ""

            mn_n, un_n, key = canon_key(mn, un)

            curve_index.append(
                {
                    "url": url,
                    "file": str(local_path),
                    "kgs_id": kgs_id,
                    "api_num_nodash": api_nd,
                    "operator": operator,
                    "lease": lease,
                    "mnemonic": mn,
                    "unit": un,
                    "descr": ds,
                    "mnemonic_n": mn_n,
                    "unit_n": un_n,
                    "canon_key": key,
                }
            )

            qhits.extend(
                check_quarantine(
                    url=url,
                    file=str(local_path),
                    kgs_id=kgs_id,
                    api_num_nodash=api_nd,
                    operator=operator,
                    lease=lease,
                    mnemonic=mn,
                    unit=un,
                    descr=ds,
                    mnemonic_n=mn_n,
                    unit_n=un_n,
                )
            )

        for w in parse_warnings:
            qhits.append(
                QuarantineHit(
                    url=url,
                    file=str(local_path),
                    kgs_id=kgs_id,
                    api_num_nodash=api_nd,
                    operator=operator,
                    lease=lease,
                    mnemonic="",
                    unit="",
                    descr="",
                    mnemonic_n="",
                    unit_n="",
                    severity="INFO" if status == "OK" else "WARN",
                    reason=f"Header parse note: {w}",
                )
            )

    return well_index, curve_index, qhits


# =============================================================================
# CLI
# =============================================================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Normalize LAS header curve mnemonics/units; build indices; emit quarantine report."
    )

    # Manifest mode (preferred)
    ap.add_argument("--ks-manifest", type=str, default="", help="Path to data/ks_las_files.txt (quoted CSV).")
    ap.add_argument("--las-root", type=str, default="", help="Local directory containing downloaded LAS files.")

    # Legacy CSV normalization mode (kept for backward compatibility)
    ap.add_argument("--curve-index", type=str, default="", help="CSV with columns: file,mnemonic,unit,descr")
    ap.add_argument("--vocal-counts", type=str, default="", help="CSV with columns: mnemonic,unit,count")

    ap.add_argument("--out-dir", type=str, required=True, help="Output directory")
    ap.add_argument("--emit-json", action="store_true", help="Also emit JSON outputs")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------
    # Manifest mode
    # --------------------------
    if args.ks_manifest:
        if not args.las_root:
            raise SystemExit("Manifest mode requires --las-root pointing to local LAS directory.")
        ks_path = Path(args.ks_manifest)
        las_root = Path(args.las_root)

        manifest_rows = read_ks_manifest(ks_path)
        well_index, curve_index, qhits = build_from_manifest(manifest_rows, las_root=las_root)

        # Write well index
        well_fields = [
            "url", "file", "status", "parse_warnings",
            "kgs_id", "api", "api_num_nodash", "operator", "lease",
            "latitude", "longitude", "location", "elevation", "elev_ref",
            "depth_start", "depth_stop",
        ]
        write_csv(out_dir / "las_well_index.csv", well_fields, well_index)

        # Write curve index
        curve_fields = [
            "url", "file", "kgs_id", "api_num_nodash", "operator", "lease",
            "mnemonic", "unit", "descr", "mnemonic_n", "unit_n", "canon_key",
        ]
        write_csv(out_dir / "las_curve_index_normalized.csv", curve_fields, curve_index)

        # Quarantine
        qrows = quarantine_report(qhits)
        qfields = [
            "severity", "reason",
            "url", "file", "kgs_id", "api_num_nodash", "operator", "lease",
            "mnemonic", "unit", "descr", "mnemonic_n", "unit_n",
        ]
        write_csv(out_dir / "las_quarantine_report.csv", qfields, qrows)

        # Summary by URL
        summary = summarize_by_group(curve_index, group_key="url")
        sfields = ["url", "n_curves", "n_unique_canon", "canon_keys"]
        write_csv(out_dir / "las_file_curve_summary.csv", sfields, summary)

        stats = _quick_stats(summary, group_key="url")
        print(f"URLs analyzed: {stats['total_groups']}")
        print(f"URLs with exactly two canonical curves (DEPT + GR): {stats['only_dept_gr']}")

        if args.emit_json:
            (out_dir / "las_well_index.json").write_text(json.dumps(well_index, indent=2), encoding="utf-8")
            (out_dir / "las_curve_index_normalized.json").write_text(json.dumps(curve_index, indent=2), encoding="utf-8")
            (out_dir / "las_quarantine_report.json").write_text(json.dumps(qrows, indent=2), encoding="utf-8")
            (out_dir / "las_file_curve_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        if pd is not None:
            try:
                df = pd.read_csv(out_dir / "las_file_curve_summary.csv")
                print("\nTop coverage patterns (first 20 rows):")
                print(df.head(20).to_string(index=False))
            except Exception:
                pass

        return 0

    # --------------------------
    # Legacy mode (CSV inputs)
    # --------------------------
    if not args.curve_index and not args.vocal_counts:
        raise SystemExit("Provide --ks-manifest (preferred) OR at least one of --curve-index / --vocal-counts")

    if args.curve_index:
        idx_path = Path(args.curve_index)
        idx_rows = read_csv_rows(idx_path)

        norm_rows, qhits = normalize_curve_index_rows(idx_rows)
        qrows = quarantine_report(qhits)
        fsum = summarize_by_group(norm_rows, group_key="file")

        write_csv(
            out_dir / "curve_index_normalized.csv",
            ["file", "mnemonic", "unit", "descr", "mnemonic_n", "unit_n", "canon_key"],
            norm_rows,
        )
        write_csv(
            out_dir / "quarantine_report.csv",
            ["severity", "reason", "url", "file", "kgs_id", "api_num_nodash", "operator", "lease",
             "mnemonic", "unit", "descr", "mnemonic_n", "unit_n"],
            qrows,
        )
        write_csv(
            out_dir / "file_curve_summary.csv",
            ["file", "n_curves", "n_unique_canon", "canon_keys"],
            fsum,
        )

        stats = _quick_stats(fsum, group_key="file")
        print(f"Files analyzed: {stats['total_groups']}")
        print(f"Files with exactly two canonical curves (DEPT + GR): {stats['only_dept_gr']}")

        if args.emit_json:
            (out_dir / "curve_index_normalized.json").write_text(json.dumps(norm_rows, indent=2), encoding="utf-8")
            (out_dir / "quarantine_report.json").write_text(json.dumps(qrows, indent=2), encoding="utf-8")
            (out_dir / "file_curve_summary.json").write_text(json.dumps(fsum, indent=2), encoding="utf-8")

    if args.vocal_counts:
        vc_path = Path(args.vocal_counts)
        vc_rows = read_csv_rows(vc_path)
        vc_norm = normalize_vocal_counts_rows(vc_rows)

        write_csv(out_dir / "vocal_counts_canonical.csv", ["mnemonic_n", "unit_n", "count", "canon_key"], vc_norm)

        if args.emit_json:
            (out_dir / "vocal_counts_canonical.json").write_text(json.dumps(vc_norm, indent=2), encoding="utf-8")

        if pd is not None:
            df = pd.read_csv(out_dir / "vocal_counts_canonical.csv")
            print("\nTop canonical curves (by count):")
            print(df.head(20).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
