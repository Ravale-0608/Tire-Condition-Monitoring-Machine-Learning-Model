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

import csv
import hashlib
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT = Path(__file__).parent / "data"

MIN_WIDTH       = 64
MIN_HEIGHT      = 64
BLUR_THRESHOLD  = 80.0
PHASH_THRESHOLD = 8       # Hamming distance ≤ this = near-duplicate
WORKERS         = min(8, (os.cpu_count() or 4))

DATASETS = [
    ("tread_train", DATA_ROOT / "train",                  "yolo",         "yolo_mixed"),
    ("tread_valid", DATA_ROOT / "valid",                  "yolo",         "yolo_mixed"),
    ("tread_test",  DATA_ROOT / "test",                   "yolo",         "yolo_mixed"),
    ("condition",   DATA_ROOT / "Tyre_Condition_Dataset", "class_folder", ""),
    ("defective",   DATA_ROOT / "defective",              "flat_folder",  "defective"),
    ("good",        DATA_ROOT / "good",                   "flat_folder",  "good"),
    ("flat",        DATA_ROOT / "flat.class",             "flat_folder",  "flat"),
    ("full",        DATA_ROOT / "full.class",             "flat_folder",  "full"),
    ("no_tire",     DATA_ROOT / "no-tire.class",          "flat_folder",  "no_tire"),
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def collect_images(ds_name, path, label_type, class_hint):
    path = Path(path)
    if not path.exists():
        return
    if label_type == "flat_folder":
        for f in path.iterdir():
            if f.suffix.lower() in IMAGE_EXTS:
                yield f, ds_name, class_hint
    elif label_type == "class_folder":
        for cls_dir in sorted(path.iterdir()):
            if cls_dir.is_dir():
                for f in cls_dir.iterdir():
                    if f.suffix.lower() in IMAGE_EXTS:
                        yield f, ds_name, cls_dir.name
    elif label_type == "yolo":
        img_dir = path / "images"
        if img_dir.exists():
            for f in img_dir.iterdir():
                if f.suffix.lower() in IMAGE_EXTS:
                    yield f, ds_name, class_hint


def check_image(args):
    img_path, ds_name, label = args
    result = {
        "path": str(img_path), "dataset": ds_name, "label": label,
        "corrupt": False, "width": 0, "height": 0,
        "low_res": False, "blur_score": None, "blurry": False,
        "phash": None, "md5": None, "duplicate": False,
    }

    try:
        with open(img_path, "rb") as fh:
            raw = fh.read()
        result["md5"] = hashlib.md5(raw).hexdigest()
    except Exception:
        result["corrupt"] = True
        return result

    try:
        with Image.open(img_path) as pil_img:
            pil_img.load()
            pil_img = pil_img.convert("RGB")
            result["width"]   = pil_img.width
            result["height"]  = pil_img.height
            result["low_res"] = pil_img.width < MIN_WIDTH or pil_img.height < MIN_HEIGHT
            result["phash"]   = str(imagehash.phash(pil_img))
    except Exception:
        result["corrupt"] = True
        return result

    try:
        img_bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is not None:
            gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            result["blur_score"] = round(score, 2)
            result["blurry"]     = score < BLUR_THRESHOLD
    except Exception:
        pass

    return result


def fast_dedup(rows):
    """
    Mark duplicates using:
    1. Exact MD5 match (O(n))
    2. Vectorized pHash Hamming distance via numpy (O(n²) but runs in C)
    Returns (exact_count, near_count).
    """
    # Exact MD5 dedup
    md5_seen = {}
    exact = 0
    for r in rows:
        if r["corrupt"] or not r["md5"]:
            continue
        if r["md5"] in md5_seen:
            r["duplicate"] = True
            exact += 1
        else:
            md5_seen[r["md5"]] = True

    # Gather valid (non-duplicate, non-corrupt) rows with pHash
    valid = [r for r in rows if not r["corrupt"] and not r["duplicate"] and r["phash"]]
    n = len(valid)
    if n == 0:
        return exact, 0

    print(f"  Building hash matrix for {n:,} images...", flush=True)

    # Convert hex pHashes → numpy bit matrix  shape (n, 64)  dtype bool
    bits = np.array(
        [imagehash.hex_to_hash(r["phash"]).hash.flatten() for r in valid],
        dtype=np.bool_
    )

    # Vectorized Hamming: D[i,j] = number of differing bits
    # hamming = rowsums[i] + rowsums[j] - 2 * dot(bits[i], bits[j])
    # = A + A.T - 2 * (bits @ bits.T)   where A = rowsums broadcast
    print(f"  Computing pairwise Hamming distances...", flush=True)
    b = bits.astype(np.int8)
    dot = b @ b.T                                # (n, n)  int
    rowsums = b.sum(axis=1)                      # (n,)
    D = rowsums[:, None] + rowsums[None, :] - 2 * dot  # (n, n) Hamming distances

    # Mark duplicates: upper triangle only, keep first occurrence
    near = 0
    dup_mask = np.zeros(n, dtype=bool)
    for i in range(n):
        if dup_mask[i]:
            continue
        # Find all j > i within threshold
        close = np.where((D[i, i+1:] <= PHASH_THRESHOLD))[0]
        for j in close + i + 1:
            if not dup_mask[j]:
                dup_mask[j] = True
                valid[j]["duplicate"] = True
                near += 1

    return exact, near


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Tire Dataset Cleaning Pipeline", flush=True)
    print("=" * 60, flush=True)

    tasks = []
    for ds_name, ds_path, label_type, class_hint in DATASETS:
        for img_path, ds, label in collect_images(ds_name, ds_path, label_type, class_hint):
            tasks.append((str(img_path), ds, label))

    total = len(tasks)
    print(f"\n[1/3] Scanning {total:,} images ({WORKERS} threads)...", flush=True)

    rows = []
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(check_image, t): t for t in tasks}
        for fut in as_completed(futures):
            rows.append(fut.result())
            done += 1
            if done % 1000 == 0 or done == total:
                print(f"  {done:,}/{total:,}  ({done/total*100:.0f}%)", flush=True)

    print("  Done.", flush=True)

    print("\n[2/3] Detecting duplicates...", flush=True)
    exact, near = fast_dedup(rows)
    print(f"  Exact duplicates : {exact:,}", flush=True)
    print(f"  Near-duplicates  : {near:,}", flush=True)

    print("\n[3/3] Writing report...", flush=True)
    fieldnames = ["path", "dataset", "label", "corrupt", "width", "height",
                  "low_res", "blur_score", "blurry", "duplicate", "phash", "md5"]
    report_csv = DATA_ROOT / "cleaning_report.csv"
    with open(report_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    corrupt_n  = sum(1 for r in rows if r["corrupt"])
    low_res_n  = sum(1 for r in rows if r["low_res"] and not r["corrupt"])
    blurry_n   = sum(1 for r in rows if r["blurry"] and not r["corrupt"])
    dup_n      = sum(1 for r in rows if r["duplicate"])
    flagged_n  = sum(1 for r in rows if r["corrupt"] or r["low_res"] or r["blurry"] or r["duplicate"])
    clean_n    = total - flagged_n

    class_counts = defaultdict(lambda: defaultdict(int))
    for r in rows:
        class_counts[r["dataset"]][r["label"]] += 1

    lines = [
        "=" * 60,
        "TIRE DATASET CLEANING SUMMARY",
        "=" * 60,
        f"Total images scanned : {total:,}",
        f"  Corrupt            : {corrupt_n:,}",
        f"  Low resolution     : {low_res_n:,}",
        f"  Blurry             : {blurry_n:,}  (Laplacian < {BLUR_THRESHOLD})",
        f"  Near-duplicates    : {dup_n:,}  (pHash Hamming <= {PHASH_THRESHOLD})",
        f"  Total flagged      : {flagged_n:,}",
        f"  Remaining clean    : {clean_n:,}",
        "",
        "-" * 60,
        "CLASS DISTRIBUTION",
        "-" * 60,
    ]
    for ds, classes in sorted(class_counts.items()):
        lines.append(f"\n  [{ds}]")
        for cls, cnt in sorted(classes.items(), key=lambda x: -x[1]):
            bar = "#" * min(cnt // 20, 40)
            lines.append(f"    {cls:<22} {cnt:>5}  {bar}")

    lines += [
        "",
        "-" * 60,
        "NEXT STEPS",
        "-" * 60,
        "1. Open data/cleaning_report.csv — filter corrupt=True first",
        "2. Review blurry=True images (blur_score column)",
        "3. Tyre_Condition UNUSABLE only has 73 images — augment heavily",
        "4. Remove duplicates before train/val split",
        f"\nReport: {report_csv}",
    ]

    summary = "\n".join(lines)
    print("\n" + summary, flush=True)
    (DATA_ROOT / "cleaning_summary.txt").write_text(summary, encoding="utf-8")
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
