# scripts/unzip_las.py
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def extract_las_from_zip(zip_path: Path, *, overwrite: bool) -> int:
    """
    Extract .las files from a single .zip.
    Writes outputs beside the zip file (same directory).
    Returns number of .las files extracted.
    """
    out_dir = zip_path.parent
    extracted = 0

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # iterate members; handle nested paths inside zip
            for info in zf.infolist():
                if info.is_dir():
                    continue

                name = info.filename
                if not name.lower().endswith(".las"):
                    continue

                out_path = out_dir / Path(name).name  # flatten to basename
                if out_path.exists() and not overwrite:
                    continue

                out_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, "r") as src, out_path.open("wb") as dst:
                    dst.write(src.read())

                extracted += 1

    except zipfile.BadZipFile:
        print(f"[WARN] Bad zip (skipping): {zip_path}")
    except Exception as e:
        print(f"[WARN] Failed to process {zip_path}: {e}")

    return extracted


def main() -> None:
    p = argparse.ArgumentParser(description="Extract .las files from .zip archives under a directory.")
    p.add_argument("--root", type=Path, default=Path("data/las"), help="Root folder to scan (default: data/las)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing .las files")
    p.add_argument("--delete-zip", action="store_true", help="Delete zip after successful extraction (only if at least one .las extracted)")
    args = p.parse_args()

    root: Path = args.root
    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")

    zip_files = sorted(root.rglob("*.zip"))
    if not zip_files:
        print(f"No .zip files found under {root}")
        return

    total_extracted = 0
    for zp in zip_files:
        n = extract_las_from_zip(zp, overwrite=args.overwrite)
        if n > 0:
            total_extracted += n
            print(f"[OK] {zp} -> extracted {n} .las")
            if args.delete_zip:
                try:
                    zp.unlink()
                    print(f"     deleted {zp.name}")
                except Exception as e:
                    print(f"[WARN] Could not delete {zp}: {e}")
        else:
            print(f"[SKIP] {zp} (no .las found or already extracted)")

    print(f"Done. Extracted {total_extracted} .las file(s) from {len(zip_files)} zip(s).")


if __name__ == "__main__":
    main()
