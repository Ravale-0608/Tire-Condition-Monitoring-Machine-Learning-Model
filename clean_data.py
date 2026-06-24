"""
Tire dataset cleaning pipeline.
Scans all sub-datasets, flags corrupt / blurry / duplicate / low-res images,
reports class balance, and writes a CSV manifest of every image with its
quality flags so you can review before deleting anything.

Run:  python clean_data.py
Output:
  data/cleaning_report.csv   — per-image flags
  data/cleaning_summary.txt  — human-readable summary
"""

import os
import csv
import json
import hashlib
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import imagehash
from PIL import Image, UnidentifiedImageError

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT = Path(__file__).parent / "data"

# Quality thresholds (tune to your dataset)
MIN_WIDTH        = 64          # pixels
MIN_HEIGHT       = 64
BLUR_THRESHOLD   = 80.0        # Laplacian variance below this = blurry
PHASH_THRESHOLD  = 8           # Hamming distance ≤ this = near-duplicate

# Sub-datasets: (folder, label_type, label_info)
#   label_type "class_folder"  → each subfolder is a class
#   label_type "flat_folder"   → all images in folder share one class name
#   label_type "yolo"          → images/ + labels/ YOLO format
DATASETS = [
    # name                      path                                       type            class/label
    ("tread_train",   DATA_ROOT / "train",                                "yolo",         "BAD_Tyres|BALD_Tyres|NORMAL_Tyres"),
    ("tread_valid",   DATA_ROOT / "valid",                                "yolo",         "BAD_Tyres|BALD_Tyres|NORMAL_Tyres"),
    ("tread_test",    DATA_ROOT / "test",                                  "yolo",         "BAD_Tyres|BALD_Tyres|NORMAL_Tyres"),
    ("condition",     DATA_ROOT / "Tyre_Condition_Dataset",               "class_folder", ""),
    ("defective",     DATA_ROOT / "defective",                            "flat_folder",  "defective"),
    ("good",          DATA_ROOT / "good",                                  "flat_folder",  "good"),
    ("flat",          DATA_ROOT / "flat.class",                            "flat_folder",  "flat"),
    ("full",          DATA_ROOT / "full.class",                            "flat_folder",  "full"),
    ("no_tire",       DATA_ROOT / "no-tire.class",                         "flat_folder",  "no_tire"),
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def laplacian_variance(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def collect_images(dataset_name, path, label_type, class_hint):
    """Yield (image_path, dataset_name, class_label) for every image in a sub-dataset."""
    path = Path(path)
    if not path.exists():
        print(f"  [SKIP] {path} does not exist")
        return

    if label_type == "flat_folder":
        for f in path.iterdir():
            if f.suffix.lower() in IMAGE_EXTS:
                yield f, dataset_name, class_hint

    elif label_type == "class_folder":
        for cls_dir in sorted(path.iterdir()):
            if cls_dir.is_dir():
                for f in cls_dir.iterdir():
                    if f.suffix.lower() in IMAGE_EXTS:
                        yield f, dataset_name, cls_dir.name

    elif label_type == "yolo":
        img_dir = path / "images"
        if not img_dir.exists():
            return
        for f in img_dir.iterdir():
            if f.suffix.lower() in IMAGE_EXTS:
                yield f, dataset_name, "yolo_mixed"


def check_image(img_path):
    """Return dict of quality flags for one image."""
    result = {
        "corrupt":       False,
        "width":         0,
        "height":        0,
        "low_res":       False,
        "blur_score":    None,
        "blurry":        False,
        "phash":         None,
        "md5":           None,
    }

    # MD5 for exact duplicates
    try:
        with open(img_path, "rb") as fh:
            result["md5"] = hashlib.md5(fh.read()).hexdigest()
    except Exception:
        result["corrupt"] = True
        return result

    # Pillow open (catches truncated / unreadable)
    try:
        with Image.open(img_path) as pil_img:
            pil_img.verify()          # detects truncated files
    except (UnidentifiedImageError, Exception):
        result["corrupt"] = True
        return result

    # Re-open for actual pixel access (verify() closes the file)
    try:
        with Image.open(img_path) as pil_img:
            pil_img = pil_img.convert("RGB")
            w, h = pil_img.size
            result["width"]  = w
            result["height"] = h
            result["low_res"] = (w < MIN_WIDTH or h < MIN_HEIGHT)
            result["phash"]  = str(imagehash.phash(pil_img))
    except Exception:
        result["corrupt"] = True
        return result

    # OpenCV for blur
    try:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            result["corrupt"] = True
            return result
        score = laplacian_variance(img_bgr)
        result["blur_score"] = round(score, 2)
        result["blurry"]     = score < BLUR_THRESHOLD
    except Exception:
        pass

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Tire Dataset Cleaning Pipeline")
    print("=" * 60)

    rows = []           # all image records
    class_counts = defaultdict(lambda: defaultdict(int))  # dataset → class → count

    # ── Phase 1: collect + quality check ──────────────────────────────────────
    print("\n[1/3] Scanning images for quality issues...")

    for ds_name, ds_path, label_type, class_hint in DATASETS:
        print(f"  {ds_name} ...", end="", flush=True)
        n = 0
        for img_path, dataset, label in collect_images(ds_name, ds_path, label_type, class_hint):
            flags = check_image(img_path)
            rows.append({
                "path":       str(img_path),
                "dataset":    dataset,
                "label":      label,
                **flags,
                "duplicate":  False,   # filled in phase 2
            })
            class_counts[dataset][label] += 1
            n += 1
        print(f" {n} images")

    # ── Phase 2: near-duplicate detection via pHash ───────────────────────────
    print("\n[2/3] Detecting near-duplicates (pHash)...")

    phash_groups = defaultdict(list)   # hash bucket → list of row indices
    for i, row in enumerate(rows):
        if row["phash"] and not row["corrupt"]:
            phash_groups[row["phash"]].append(i)

    # For each group of identical hashes, mark all but the first as duplicate
    exact_dup_count = 0
    for indices in phash_groups.values():
        if len(indices) > 1:
            for idx in indices[1:]:
                rows[idx]["duplicate"] = True
                exact_dup_count += 1

    # Near-duplicate pass (Hamming distance on pHash)
    # Collect unique hashes only to limit O(n²) cost
    unique_hashes = {}   # hash_str → first row index
    near_dup_count = 0
    for i, row in enumerate(rows):
        if row["corrupt"] or row["duplicate"] or not row["phash"]:
            continue
        h = imagehash.hex_to_hash(row["phash"])
        found_near = False
        for stored_h_str, ref_idx in unique_hashes.items():
            stored_h = imagehash.hex_to_hash(stored_h_str)
            if h - stored_h <= PHASH_THRESHOLD:
                rows[i]["duplicate"] = True
                near_dup_count += 1
                found_near = True
                break
        if not found_near:
            unique_hashes[row["phash"]] = i

    print(f"  Exact duplicates flagged : {exact_dup_count}")
    print(f"  Near-duplicates flagged  : {near_dup_count}")

    # ── Phase 3: Write outputs ────────────────────────────────────────────────
    print("\n[3/3] Writing report...")

    report_csv = DATA_ROOT / "cleaning_report.csv"
    fieldnames = ["path", "dataset", "label", "corrupt", "width", "height",
                  "low_res", "blur_score", "blurry", "duplicate", "phash", "md5"]
    with open(report_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ── Summary ───────────────────────────────────────────────────────────────
    total         = len(rows)
    corrupt_n     = sum(1 for r in rows if r["corrupt"])
    low_res_n     = sum(1 for r in rows if r["low_res"] and not r["corrupt"])
    blurry_n      = sum(1 for r in rows if r["blurry"] and not r["corrupt"])
    duplicate_n   = sum(1 for r in rows if r["duplicate"])
    flagged_n     = sum(1 for r in rows if r["corrupt"] or r["low_res"] or r["blurry"] or r["duplicate"])
    clean_n       = total - flagged_n

    summary_lines = [
        "=" * 60,
        "TIRE DATASET CLEANING SUMMARY",
        "=" * 60,
        f"Total images scanned : {total:,}",
        f"  Corrupt / unreadable : {corrupt_n:,}",
        f"  Low resolution       : {low_res_n:,}",
        f"  Blurry               : {blurry_n:,}  (Laplacian < {BLUR_THRESHOLD})",
        f"  Near-duplicates      : {duplicate_n:,}  (pHash Hamming ≤ {PHASH_THRESHOLD})",
        f"  Total flagged        : {flagged_n:,}",
        f"  Remaining clean      : {clean_n:,}",
        "",
        "─" * 60,
        "CLASS DISTRIBUTION PER DATASET",
        "─" * 60,
    ]

    for ds_name, classes in sorted(class_counts.items()):
        summary_lines.append(f"\n  [{ds_name}]")
        for cls, cnt in sorted(classes.items(), key=lambda x: -x[1]):
            bar = "█" * min(cnt // 10, 50)
            summary_lines.append(f"    {cls:<20} {cnt:>5}  {bar}")

    summary_lines += [
        "",
        "─" * 60,
        "RECOMMENDED ACTIONS",
        "─" * 60,
        "1. Review cleaning_report.csv — filter corrupt=True, then blurry=True",
        "2. Check Tyre_Condition UNUSABLE class (only 73 images — augment heavily)",
        "3. Decide on unified label schema before merging datasets",
        "4. Remove confirmed duplicates before train/val split",
        f"\nFull per-image report: {report_csv}",
    ]

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    summary_path = DATA_ROOT / "cleaning_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write(summary_text)

    print(f"\nDone. Reports saved to:\n  {report_csv}\n  {summary_path}")


if __name__ == "__main__":
    main()
