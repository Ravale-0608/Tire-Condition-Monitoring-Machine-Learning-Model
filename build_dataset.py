"""
Reorganises all sub-datasets into a single YOLOv8-classify directory layout:

  data/unified/
    train/  val/  test/
      no_tire / flat / defective / worn / good / new

Label mapping
─────────────
no_tire   ← no-tire.class/
flat      ← flat.class/
defective ← defective/  +  Tyre_Condition_Dataset/UNUSABLE
worn      ← BALD_Tyres images  +  BAD_Tyres images  (from YOLO labels)
good      ← good/  +  full.class/  +  SERVICEABLE  +  NORMAL_Tyres images
new       ← Tyre_Condition_Dataset/NEW

Split: 80 % train / 10 % val / 10 % test  (stratified, random)

Run:  python build_dataset.py
"""

import random
import shutil
from collections import defaultdict
from pathlib import Path

DATA_ROOT = Path(__file__).parent / "data"
NEW_DATA  = Path(__file__).parent / "new data"
OUT_ROOT  = DATA_ROOT / "unified"

SPLITS = {"train": 0.80, "val": 0.10, "test": 0.10}
MAX_PER_CLASS = 1000   # raised cap — more data now available
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42

# YOLO class index → unified label
YOLO_CLASS_MAP = {
    0: "worn",   # BAD_Tyres
    1: "worn",   # BALD_Tyres
    2: "good",   # NORMAL_Tyres
}

random.seed(SEED)


def collect_flat_folder(folder: Path, label: str):
    if not folder.exists():
        return []
    return [(f, label) for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTS]


def collect_yolo_by_dominant_class(images_dir: Path, labels_dir: Path):
    """
    For each image, read its YOLO label file, find the dominant class,
    and assign that as the unified label.
    Images with no label file are skipped.
    """
    if not images_dir.exists():
        return []
    results = []
    for img in images_dir.iterdir():
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        lbl_file = labels_dir / (img.stem + ".txt")
        if not lbl_file.exists():
            continue
        class_votes = defaultdict(int)
        with open(lbl_file) as fh:
            for line in fh:
                parts = line.strip().split()
                if parts:
                    try:
                        cls_idx = int(parts[0])
                        unified = YOLO_CLASS_MAP.get(cls_idx)
                        if unified:
                            class_votes[unified] += 1
                    except ValueError:
                        pass
        if class_votes:
            dominant = max(class_votes, key=class_votes.get)
            results.append((img, dominant))
    return results


def split_and_copy(items, label_override=None):
    """
    items: list of (Path, label)
    Shuffle, split, copy into OUT_ROOT/split/label/filename
    Returns count per split.
    """
    random.shuffle(items)
    n = len(items)
    n_val  = max(1, int(n * SPLITS["val"]))
    n_test = max(1, int(n * SPLITS["test"]))
    n_train = n - n_val - n_test

    buckets = (
        [("train", items[:n_train]),
         ("val",   items[n_train:n_train + n_val]),
         ("test",  items[n_train + n_val:])]
    )

    counts = defaultdict(int)
    for split, group in buckets:
        for img_path, label in group:
            lbl = label_override or label
            dst_dir = OUT_ROOT / split / lbl
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / img_path.name
            # Handle name collisions by appending parent folder name
            if dst.exists():
                dst = dst_dir / f"{img_path.parent.name}_{img_path.name}"
            shutil.copy2(str(img_path), str(dst))
            counts[split] += 1
    return counts


def main():
    if OUT_ROOT.exists():
        print(f"Removing existing {OUT_ROOT} ...", flush=True)
        shutil.rmtree(OUT_ROOT)

    print("Building unified YOLOv8-classify dataset ...\n", flush=True)

    all_items = defaultdict(list)  # label -> [(path, label)]

    # ── no_tire ───────────────────────────────────────────────────────────────
    all_items["no_tire"] += collect_flat_folder(DATA_ROOT / "no-tire.class", "no_tire")

    # ── flat ──────────────────────────────────────────────────────────────────
    all_items["flat"] += collect_flat_folder(DATA_ROOT / "flat.class", "flat")

    # ── defective ─────────────────────────────────────────────────────────────
    all_items["defective"] += collect_flat_folder(DATA_ROOT / "defective", "defective")
    all_items["defective"] += collect_flat_folder(
        DATA_ROOT / "Tyre_Condition_Dataset" / "UNUSABLE", "defective"
    )
    # New data — labeled Defective and cracked
    all_items["defective"] += collect_flat_folder(NEW_DATA / "Defective",                         "defective")
    all_items["defective"] += collect_flat_folder(NEW_DATA / "training_data" / "cracked",         "defective")

    # ── worn ──────────────────────────────────────────────────────────────────
    for split in ("train", "valid", "test"):
        img_dir = DATA_ROOT / split.replace("valid", "valid") / "images"
        lbl_dir = DATA_ROOT / split.replace("valid", "valid") / "labels"
        # Normalise split name
        actual_split_dir = DATA_ROOT / split
        if not actual_split_dir.exists():
            continue
        items = collect_yolo_by_dominant_class(
            actual_split_dir / "images",
            actual_split_dir / "labels",
        )
        for img, label in items:
            all_items[label].append((img, label))

    # ── good ──────────────────────────────────────────────────────────────────
    all_items["good"] += collect_flat_folder(DATA_ROOT / "good", "good")
    all_items["good"] += collect_flat_folder(DATA_ROOT / "full.class", "good")
    all_items["good"] += collect_flat_folder(
        DATA_ROOT / "Tyre_Condition_Dataset" / "SERVICEABLE", "good"
    )
    # New data — labeled Good and normal
    all_items["good"] += collect_flat_folder(NEW_DATA / "Good",                                  "good")
    all_items["good"] += collect_flat_folder(NEW_DATA / "training_data" / "normal",              "good")

    # ── new ───────────────────────────────────────────────────────────────────
    all_items["new"] += collect_flat_folder(
        DATA_ROOT / "Tyre_Condition_Dataset" / "NEW", "new"
    )

    # ── Cap large classes ─────────────────────────────────────────────────────
    for label in all_items:
        if len(all_items[label]) > MAX_PER_CLASS:
            random.shuffle(all_items[label])
            all_items[label] = all_items[label][:MAX_PER_CLASS]

    # ── Summary before split ──────────────────────────────────────────────────
    print("Image counts per class (after cap):")
    total = 0
    for label in ["no_tire", "flat", "defective", "worn", "good", "new"]:
        n = len(all_items[label])
        bar = "#" * (n // 10)
        print(f"  {label:<12} {n:>5}  {bar}")
        total += n
    print(f"  {'TOTAL':<12} {total:>5}\n")

    # ── Split & copy ──────────────────────────────────────────────────────────
    split_totals = defaultdict(int)
    for label, items in all_items.items():
        counts = split_and_copy(items)
        for split, n in counts.items():
            split_totals[split] += n
        print(f"  {label:<12} -> train:{counts['train']}  val:{counts['val']}  test:{counts['test']}", flush=True)

    print(f"\nDataset written to: {OUT_ROOT}")
    print(f"  train : {split_totals['train']:,} images")
    print(f"  val   : {split_totals['val']:,} images")
    print(f"  test  : {split_totals['test']:,} images")
    print(f"  TOTAL : {sum(split_totals.values()):,} images")

    # ── Holdout test set (never seen during training or val) ─────────────────
    holdout_root = OUT_ROOT / "holdout"
    holdout_map  = {
        "defective": [NEW_DATA / "testing_data" / "cracked"],
        "good":      [NEW_DATA / "testing_data" / "normal"],
    }
    holdout_counts = defaultdict(int)
    for label, folders in holdout_map.items():
        dst_dir = holdout_root / label
        dst_dir.mkdir(parents=True, exist_ok=True)
        for folder in folders:
            for img in folder.iterdir():
                if img.suffix.lower() in IMAGE_EXTS:
                    dst = dst_dir / img.name
                    if dst.exists():
                        dst = dst_dir / f"{img.parent.name}_{img.name}"
                    shutil.copy2(str(img), str(dst))
                    holdout_counts[label] += 1

    print(f"\nHoldout test set (completely unseen):")
    for lbl, cnt in holdout_counts.items():
        print(f"  {lbl:<12} {cnt:>4}")

    # Write a data.yaml for reference
    yaml_text = (
        f"path: {OUT_ROOT}\n"
        f"nc: 6\n"
        f"names: ['no_tire', 'flat', 'defective', 'worn', 'good', 'new']\n"
    )
    (OUT_ROOT / "data.yaml").write_text(yaml_text)
    print("\ndata.yaml written.")


if __name__ == "__main__":
    main()
