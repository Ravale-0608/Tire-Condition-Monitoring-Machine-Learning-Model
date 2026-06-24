"""
Moves flagged images (duplicates + blurry) out of the active dataset
into data/quarantine/ — fully reversible, nothing is deleted.

For YOLO images, moves the matching .txt label file too.

Run:  python quarantine_flagged.py
      python quarantine_flagged.py --dry-run   (preview only, no moves)
"""

import csv
import shutil
import sys
from pathlib import Path

DATA_ROOT  = Path(__file__).parent / "data"
REPORT_CSV = DATA_ROOT / "cleaning_report.csv"
QUARANTINE = DATA_ROOT / "quarantine"

DRY_RUN = "--dry-run" in sys.argv

def yolo_label_path(img_path: Path) -> Path | None:
    """Return the .txt label file for a YOLO image if it exists."""
    if "images" in img_path.parts:
        parts = list(img_path.parts)
        parts[parts.index("images")] = "labels"
        label = Path(*parts).with_suffix(".txt")
        return label if label.exists() else None
    return None

def move(src: Path, dst: Path):
    if DRY_RUN:
        print(f"  [DRY] {src.name}  ->  {dst.relative_to(DATA_ROOT)}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))

def main():
    if not REPORT_CSV.exists():
        print("ERROR: data/cleaning_report.csv not found. Run clean_data.py first.")
        sys.exit(1)

    if DRY_RUN:
        print("DRY RUN — no files will be moved.\n")

    rows = []
    with open(REPORT_CSV, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    duplicates = [r for r in rows if r["duplicate"] == "True"]
    blurry     = [r for r in rows if r["blurry"] == "True" and r["duplicate"] != "True"]

    print(f"Flagged for quarantine:")
    print(f"  Duplicates : {len(duplicates):,}")
    print(f"  Blurry     : {len(blurry):,}  (not already flagged as duplicate)")
    print(f"  Total      : {len(duplicates) + len(blurry):,}\n")

    moved_imgs   = 0
    moved_labels = 0
    missing      = 0

    for reason, group in [("duplicates", duplicates), ("blurry", blurry)]:
        print(f"Moving {reason}...")
        for r in group:
            src = Path(r["path"])
            if not src.exists():
                missing += 1
                continue

            # Mirror the path under quarantine/
            try:
                rel = src.relative_to(DATA_ROOT)
            except ValueError:
                rel = Path(src.name)
            dst = QUARANTINE / reason / rel

            move(src, dst)
            moved_imgs += 1

            # Move matching YOLO label if present
            label_src = yolo_label_path(src)
            if label_src:
                try:
                    label_rel = label_src.relative_to(DATA_ROOT)
                except ValueError:
                    label_rel = Path(label_src.name)
                label_dst = QUARANTINE / reason / label_rel
                move(label_src, label_dst)
                moved_labels += 1

        print(f"  Done.\n")

    if DRY_RUN:
        print(f"DRY RUN complete — {moved_imgs:,} images + {moved_labels:,} labels would be moved.")
    else:
        print(f"Quarantine complete.")
        print(f"  Images moved  : {moved_imgs:,}")
        print(f"  Labels moved  : {moved_labels:,}")
        print(f"  Already gone  : {missing:,}")
        print(f"\nAll moved files are in: {QUARANTINE}")
        print("To restore everything:  move data/quarantine/duplicates/* and data/quarantine/blurry/* back.")

if __name__ == "__main__":
    main()
